"""
Stock Watchlist — Streamlit App  (free-tier-friendly, fundamentals-ranked)

Why this rewrite exists
-----------------------
The previous version fetched 3 separate FMP calls **per ticker** (profile +
key-metrics + growth). With ~89 tickers that's ~267 calls fired in an 8-thread
burst — more than the FMP free tier's entire 250-requests/day budget, in a single
page load. Result: prices loaded, but the quota ran out mid-run and almost every
fundamentals column came back blank.

What changed
------------
1. **Batch prices.** All quotes come from one (or two) ``batch-quote`` calls
   instead of ~89 ``profile`` calls. 52-week range, market cap and PE come along
   for free. (~89 calls -> ~2.)
2. **Cheaper, correct fundamentals.** 2 calls/ticker, derived from raw statement
   numbers (see fundamentals.py), so the previously-blank Gross Margin / Rule-40
   columns now populate.
3. **Disk cache + long TTL.** Fundamentals change quarterly, so they're cached
   for 14 days on disk and survive reruns/restarts. Prices cache for 6h.
4. **Per-run budget + rate-limit abort.** Each load refreshes at most
   ``FMP_FUND_BUDGET`` (default 45) tickers' fundamentals and stops instantly on a
   "Limit Reach" response — so a cold board fills over a few refreshes and a load
   never blows the daily quota. Already-loaded values keep showing.

Net effect: a warm board costs ~2 calls/refresh; a cold board fills within a day
on the free tier without ever blanking out.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import logging
import os
import tempfile
import threading
import time
from typing import Optional

import pandas as pd
import requests
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# ---------------------------------------------------------------------------
# Config (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Watchlist",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Watchlist import — guarded so a bad/stale watchlist_data.py shows a clear
# message instead of a raw traceback.
# ---------------------------------------------------------------------------
try:
    from watchlist_data import WATCHLIST as _BASE_WATCHLIST, SECTOR_ORDER as _BASE_SECTORS
except ImportError:
    try:
        import watchlist_data as _wd
        _found = [n for n in dir(_wd) if not n.startswith("_")]
    except Exception as _e:  # noqa: BLE001
        _found = [f"(watchlist_data.py failed to import: {_e})"]
    st.error(
        "**`watchlist_data.py` is missing `WATCHLIST` / `SECTOR_ORDER`.**\n\n"
        f"**Names actually found:** `{_found}`\n\n"
        "**Fix:** re-commit the complete `watchlist_data.py`, then redeploy."
    )
    st.stop()

from watchlist_growth import GROWTH_WATCHLIST
from fundamentals import (
    fetch_fundamentals,
    compute_composite_scores,
    target_is_sane,
    has_any,
    FMPRateLimitError,
    FACTORS,
    GROWTH_FACTORS,
)
import market_risk

# Merge the additive growth picks onto the original list (nothing removed).
_seen = {item["ticker"] for item in _BASE_WATCHLIST}
WATCHLIST = _BASE_WATCHLIST + [g for g in GROWTH_WATCHLIST if g["ticker"] not in _seen]
SECTOR_ORDER = _BASE_SECTORS + [s for s in ["Hypergrowth"] if s not in _BASE_SECTORS]
SYMBOLS = [item["ticker"] for item in WATCHLIST]

# FMP stable API (legacy /api/v3/ was deprecated for new accounts after Aug 31, 2025)
FMP_BASE = "https://financialmodelingprep.com/stable"

# Cache TTLs (seconds)
QUOTE_TTL = 6 * 60 * 60            # prices: 6h
FUND_TTL = 14 * 24 * 60 * 60       # fundamentals: 14d (they change quarterly)
MARKET_TTL = 6 * 60 * 60           # market context: 6h

# Free-tier guards.
# Default budget fills the whole board in ONE cold load when it fits the daily
# cap (2 calls/ticker + a few overhead, kept under ~225 so there's headroom under
# the 250/day free limit), and caps per-load calls for bigger lists so they fill
# over a couple of loads instead. The 14-day disk cache means a full fill happens
# at most ~once; warm refreshes cost ~2 calls. Override with FMP_FUND_BUDGET.
_DEFAULT_BUDGET = min(len(WATCHLIST), 110)
FUND_BUDGET = int(os.getenv("FMP_FUND_BUDGET", str(_DEFAULT_BUDGET)))
BATCH_SIZE = 50                                          # symbols per batch-quote call
MAX_WORKERS = 3                                          # gentle concurrency (avoids per-minute bursts)


def get_api_key() -> Optional[str]:
    try:
        return st.secrets["FMP_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.getenv("FMP_API_KEY")


# ---------------------------------------------------------------------------
# Disk-persistent cache (survives reruns AND container restarts within a deploy)
# ---------------------------------------------------------------------------
def _cache_dir() -> str:
    for cand in (os.path.join(os.path.dirname(os.path.abspath(__file__)), ".fmp_cache"),
                 os.path.join(tempfile.gettempdir(), "fmp_watchlist_cache")):
        try:
            os.makedirs(cand, exist_ok=True)
            return cand
        except Exception:  # noqa: BLE001
            continue
    return tempfile.gettempdir()


class PersistentStore:
    """ticker -> {"data": ..., "ts": epoch}. JSON-backed, write-through on flush."""

    def __init__(self, path: str):
        self.path = path
        try:
            with open(path, "r") as fh:
                self.data = json.load(fh)
        except Exception:  # noqa: BLE001 — missing/corrupt cache: start fresh
            self.data = {}

    def get(self, key: str) -> Optional[dict]:
        return self.data.get(key)

    def set(self, key: str, value: dict) -> None:
        self.data[key] = value

    def flush(self) -> None:
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(self.data, fh)
            os.replace(tmp, self.path)
        except Exception as e:  # noqa: BLE001 — cache is best-effort
            logger.warning("cache flush failed (%s): %s", self.path, e)


@st.cache_resource
def quote_store() -> PersistentStore:
    return PersistentStore(os.path.join(_cache_dir(), "quotes.json"))


@st.cache_resource
def fund_store() -> PersistentStore:
    return PersistentStore(os.path.join(_cache_dir(), "fundamentals.json"))


@st.cache_resource
def market_store() -> PersistentStore:
    return PersistentStore(os.path.join(_cache_dir(), "market.json"))


def _fresh(entry: Optional[dict], ttl: int) -> bool:
    return bool(entry) and (time.time() - entry.get("ts", 0)) < ttl


# ---------------------------------------------------------------------------
# Price layer — batch-quote (with per-ticker fallback)
# ---------------------------------------------------------------------------
def _looks_rate_limited(text: str) -> bool:
    t = (text or "").lower()
    return ("limit reach" in t or "upgrade your plan" in t
            or "too many requests" in t or "bandwidth" in t)


def _num(rec: dict, *keys):
    for k in keys:
        if k in rec and rec[k] is not None:
            try:
                return float(rec[k])
            except (TypeError, ValueError):
                continue
    return None


def _norm_quote(row: dict) -> dict:
    return {
        "symbol": row.get("symbol"),
        "price": _num(row, "price"),
        "changePercentage": _num(row, "changePercentage", "changesPercentage"),
        "marketCap": _num(row, "marketCap", "mktCap"),
        "yearLow": _num(row, "yearLow"),
        "yearHigh": _num(row, "yearHigh"),
        "pe": _num(row, "pe", "peTTM", "priceEarningsRatioTTM"),
        "eps": _num(row, "eps", "epsTTM"),
    }


def _batch_quote(symbols: list[str], api_key: str):
    """One call for many symbols. Returns list of rows, [] if unsupported, raises
    FMPRateLimitError on quota exhaustion."""
    r = requests.get(f"{FMP_BASE}/batch-quote",
                     params={"symbols": ",".join(symbols), "apikey": api_key}, timeout=25)
    if r.status_code == 429 or _looks_rate_limited(r.text):
        raise FMPRateLimitError(r.text[:120])
    if r.status_code != 200:
        logger.info("batch-quote HTTP %s — will fall back to per-ticker", r.status_code)
        return []
    try:
        data = r.json()
    except ValueError:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and data.get("symbol"):
        return [data]
    return []


def _single_quote(symbol: str, api_key: str):
    r = requests.get(f"{FMP_BASE}/quote",
                     params={"symbol": symbol, "apikey": api_key}, timeout=12)
    if r.status_code == 429 or _looks_rate_limited(r.text):
        raise FMPRateLimitError(r.text[:120])
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except ValueError:
        return None
    if isinstance(data, list) and data:
        return data[0]
    if isinstance(data, dict) and data.get("symbol"):
        return data
    return None


def fetch_quotes(api_key: str, symbols: list[str], force: bool) -> tuple[dict, int, int, bool]:
    """Return (quotes_by_symbol, n_fresh_fetched, n_from_cache, rate_limited)."""
    store = quote_store()
    now = time.time()
    quotes: dict = {}
    need: list[str] = []
    from_cache = 0

    for s in symbols:
        e = store.get(s)
        if _fresh(e, QUOTE_TTL) and not force:
            quotes[s] = e["data"]
            from_cache += 1
        else:
            if e:
                quotes[s] = e["data"]          # provisional stale value
            need.append(s)

    fetched = 0
    rate_limited = False
    if need:
        got: dict = {}
        try:
            for chunk in (need[i:i + BATCH_SIZE] for i in range(0, len(need), BATCH_SIZE)):
                rows = _batch_quote(chunk, api_key)
                for row in rows:
                    sym = row.get("symbol")
                    if sym:
                        got[sym] = _norm_quote(row)
                # If batch isn't supported on this plan, rows is [] -> per-ticker fallback
                missing = [s for s in chunk if s not in got]
                if missing and not rows:
                    for sym in missing:
                        row = _single_quote(sym, api_key)
                        if row:
                            got[sym] = _norm_quote(row)
        except FMPRateLimitError:
            rate_limited = True

        for sym, d in got.items():
            store.set(sym, {"data": d, "ts": now})
            quotes[sym] = d
            fetched += 1
        store.flush()

    return quotes, fetched, from_cache, rate_limited


# ---------------------------------------------------------------------------
# Fundamentals layer — disk-cached, per-run budget, rate-limit abort
# ---------------------------------------------------------------------------
def fetch_fundamentals_all(api_key: str, symbols: list[str], market_caps: dict, force: bool):
    """Return (funds_by_symbol, n_refreshed, pending_symbols, n_loaded_total, rate_limited).

    funds_by_symbol includes cached (possibly stale) values for everything we know.
    Only up to FUND_BUDGET uncached/oldest tickers are refreshed this run.
    """
    store = fund_store()
    now = time.time()
    funds: dict = {}
    candidates: list[tuple[float, str]] = []   # (age-key, symbol) for things needing refresh

    for s in symbols:
        e = store.get(s)
        if e:
            funds[s] = e["data"]
        if force or not _fresh(e, FUND_TTL):
            # never-fetched (ts 0) sort first, then oldest-first
            candidates.append((e.get("ts", 0.0) if e else 0.0, s))

    candidates.sort(key=lambda kv: kv[0])
    to_fetch = [s for _, s in candidates[:FUND_BUDGET]]
    pending = [s for _, s in candidates[FUND_BUDGET:]]

    stop = threading.Event()
    rate_limited = False
    refreshed = 0
    lock = threading.Lock()

    def work(sym: str):
        if stop.is_set():
            return sym, "SKIP"
        try:
            return sym, fetch_fundamentals(sym, api_key, market_caps.get(sym))
        except FMPRateLimitError:
            stop.set()
            return sym, "RATE"

    if to_fetch:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = [ex.submit(work, s) for s in to_fetch]
            for fut in as_completed(futs):
                sym, res = fut.result()
                if res == "RATE":
                    rate_limited = True
                elif res not in ("SKIP", None) and has_any(res):
                    with lock:
                        store.set(sym, {"data": res, "ts": now})
                        funds[sym] = res
                        refreshed += 1
        store.flush()

    # Anything that was only "pending" because of the budget but never had data
    # is still pending; anything skipped due to rate-limit is pending too.
    pending = [s for s in symbols if s not in funds]
    n_loaded = len([s for s in symbols if s in funds and has_any(funds[s])])
    return funds, refreshed, pending, n_loaded, rate_limited


@st.cache_data(ttl=MARKET_TTL, show_spinner=False)
def cached_market_context(api_key: str) -> dict:
    return market_risk.get_market_context(api_key)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
CURRENCY_SYMBOL = {"USD": "$", "INR": "₹", "JPY": "¥", "EUR": "€", "GBP": "£", "CAD": "C$"}
REGION_CURRENCY = {"US": "USD", "IN": "INR", "JP": "JPY", "UK": "GBP", "CA": "CAD", "EU": "EUR"}


def fmt_price(v, currency="USD"):
    if v is None or pd.isna(v):
        return "—"
    sym = CURRENCY_SYMBOL.get(currency, currency + " ")
    return f"{sym}{v:,.2f}"


def fmt_mcap(v, currency="USD"):
    if v is None or pd.isna(v):
        return "—"
    sym = CURRENCY_SYMBOL.get(currency, currency + " ")
    if v >= 1e12: return f"{sym}{v/1e12:.2f}T"
    if v >= 1e9:  return f"{sym}{v/1e9:.2f}B"
    if v >= 1e6:  return f"{sym}{v/1e6:.0f}M"
    return f"{sym}{v:,.0f}"


def build_dataframe(quotes: dict, fundamentals: dict, scores: dict, growth_scores: dict) -> pd.DataFrame:
    rows = []
    for item in WATCHLIST:
        tk = item["ticker"]
        q = quotes.get(tk) or {}
        price = q.get("price")
        day_pct = q.get("changePercentage")
        mcap = q.get("marketCap")
        lo, hi = q.get("yearLow"), q.get("yearHigh")
        currency = REGION_CURRENCY.get(item.get("region", "US"), "USD")

        target = item.get("target")
        target_ok = target_is_sane(target, lo, hi)
        upside = ((target - price) / price) * 100 if (price and target and target_ok) else None

        sc = scores.get(tk) or {}
        gsc = growth_scores.get(tk) or {}
        fund = fundamentals.get(tk) or {}

        def pct(key):
            v = fund.get(key)
            return v * 100 if v is not None else None

        rows.append({
            "Ticker": tk,
            "Region": item["region"],
            "Name": item["name"],
            "Sector": item["sector"],
            "Price": price,
            "Day %": day_pct,
            "52w Low": lo,
            "52w High": hi,
            "52w Pos %": ((price - lo) / (hi - lo) * 100) if (price and lo and hi and hi > lo) else None,
            "Mkt Cap": mcap,
            "Currency": currency,
            "Score": (sc.get("score") * 100) if sc.get("score") is not None else None,
            "Coverage": sc.get("coverage", 0),
            "Growth Score": (gsc.get("score") * 100) if gsc.get("score") is not None else None,
            "Growth Cov": gsc.get("coverage", 0),
            "Earnings Yld %": pct("earnings_yield"),
            "ROIC %": pct("roic"),
            "Rev Growth %": pct("rev_growth"),
            "Gross Mgn %": pct("gross_margin"),
            "Rule40": fund.get("rule_of_40"),
            "FCF Yld %": pct("fcf_yield"),
            "Target": target,
            "Target OK": target_ok,
            "Upside %": upside,
            "Thesis": item["thesis"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("📈 Stock Watchlist")
st.caption(
    f"{len(WATCHLIST)} tickers across {len(SECTOR_ORDER)} sectors. Ranked by **reported "
    f"fundamentals** (earnings yield · ROIC · revenue growth · FCF yield), not analyst targets. "
    f"Live data from FMP — batched quotes + a 14-day fundamentals cache to stay inside the free tier."
)

api_key = get_api_key()
if not api_key:
    st.error("Missing FMP API key. Set `FMP_API_KEY` in Streamlit secrets (production) or as "
             "an environment variable (local). See README.md.")
    st.stop()

if "force_refresh" not in st.session_state:
    st.session_state.force_refresh = False

col_a, col_b, col_c = st.columns([1.2, 1.4, 4])
with col_a:
    if st.button("🔄 Refresh prices", use_container_width=True,
                 help="Refetch live prices now (1–2 API calls). Fundamentals use the 14-day cache."):
        st.session_state.force_refresh = "prices"
        st.rerun()
with col_b:
    if st.button("📊 Fetch more fundamentals", use_container_width=True,
                 help=f"Pull the next {FUND_BUDGET} tickers' fundamentals into the cache. "
                      "Cold boards fill over a few clicks; cached values change quarterly."):
        st.session_state.force_refresh = "funds"
        st.rerun()
with col_c:
    st.caption(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
               f"prices cached {QUOTE_TTL//3600}h · fundamentals cached {FUND_TTL//86400}d · "
               f"budget {FUND_BUDGET} tickers/refresh")

_force = st.session_state.force_refresh
st.session_state.force_refresh = False
force_prices = _force in ("prices", "funds", True)
force_funds = _force in ("funds", True)

with st.spinner("Loading prices…"):
    quotes, q_fetched, q_cached, q_rl = fetch_quotes(api_key, SYMBOLS, force_prices)

market_caps = {s: (quotes.get(s) or {}).get("marketCap") for s in SYMBOLS}

with st.spinner(f"Loading fundamentals (up to {FUND_BUDGET} this refresh)…"):
    fundamentals, f_refreshed, f_pending, f_loaded, f_rl = fetch_fundamentals_all(
        api_key, SYMBOLS, market_caps, force_funds)

# Earnings-yield fallback from quote PE (costs no extra API calls). Works even
# when deep fundamentals are rate-limited/uncached: we create a minimal record so
# every ticker with a P/E still gets an earnings-yield signal and a (thin) Value
# score from the single batch-quote call, instead of a fully blank row.
for s in SYMBOLS:
    f = fundamentals.get(s)
    if (not f) or f.get("earnings_yield") is None:
        pe = (quotes.get(s) or {}).get("pe")
        if pe and pe > 0:
            if not f:
                f = {}
                fundamentals[s] = f
            f["earnings_yield"] = 1.0 / pe

scores = compute_composite_scores(fundamentals, FACTORS)
growth_scores = compute_composite_scores(fundamentals, GROWTH_FACTORS)

# ---------------------------------------------------------------------------
# Market-conditions dashboard — CONTEXT, not buy/sell signals.
# ---------------------------------------------------------------------------
mkt = cached_market_context(api_key)
with st.container():
    st.subheader("🌡️ Market conditions")
    st.caption(
        "Context for **how** to deploy (position size · scaling in/out · rebalancing) — "
        "**not** a buy/sell signal. This describes the environment you're investing into."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("S&P 500", f"{mkt['index_price']:,.0f}" if mkt["index_price"] else "—",
              mkt["trend"] if mkt["trend"] != "unknown" else None)
    c2.metric("vs 200-day avg",
              f"{mkt['pct_from_ma200']:+.1f}%" if mkt["pct_from_ma200"] is not None else "—")
    c3.metric("Index 52w position",
              f"{mkt['index_52w_pos']:.0f}%" if mkt["index_52w_pos"] is not None else "—")
    c4.metric("VIX (volatility)", f"{mkt['vix']:.1f}" if mkt["vix"] else "—",
              mkt["vix_label"] if mkt["vix_label"] != "unknown" else None)
    if mkt.get("vix_context"):
        st.caption(f"**Volatility regime — {mkt['vix_label']}:** {mkt['vix_context']}")
    for note in mkt.get("notes", []):
        st.info(note)
st.divider()

# ---------------------------------------------------------------------------
# Status banner — clear about what's live, cached, pending, or rate-limited.
# ---------------------------------------------------------------------------
n = len(SYMBOLS)
if q_rl or f_rl or mkt.get("rate_limited"):
    st.warning(
        f"⏳ **FMP quota reached** — showing cached values. "
        f"{f_loaded}/{n} tickers have fundamentals; **{len(f_pending)} pending** will fill once the "
        f"daily quota resets (or on a paid plan). Prices/fundamentals already cached are unaffected."
    )
elif f_pending:
    st.info(
        f"📊 Fundamentals: **{f_loaded}/{n} loaded** "
        f"({f_refreshed} refreshed this load) · **{len(f_pending)} pending**. "
        f"Click **Fetch more fundamentals** to pull the next batch — cold boards fill in a few clicks, "
        f"then stay cached for {FUND_TTL//86400} days."
    )
else:
    st.success(f"✅ All {n} tickers loaded — prices live, fundamentals cached "
               f"({q_cached} prices from cache, {f_loaded} with fundamentals).")

df = build_dataframe(quotes, fundamentals, scores, growth_scores)

# Flag any static target the sanity guard rejected (split/stale catcher).
flagged = df[(df["Target"].notna()) & (~df["Target OK"])]
if len(flagged):
    def _fmt_flag(row):
        hi = row["52w High"]
        hi_str = f"{hi:g}" if pd.notna(hi) else "n/a"
        return f"{row['Ticker']} (target {row['Target']:g} vs 52w high {hi_str})"
    names = ", ".join(_fmt_flag(r) for _, r in flagged.iterrows())
    st.error(
        f"⚠️ {len(flagged)} static target(s) look implausible vs the live 52-week range and were "
        f"excluded from upside — likely a split or stale entry in `watchlist_data.py`: {names}"
    )

# Summary metrics
loaded = df["Price"].notna().sum()
scored = df[df["Score"].notna()]
avg_score = scored["Score"].mean() if len(scored) else 0
top_row = scored.sort_values("Score", ascending=False).head(1)
top_name = top_row["Ticker"].iloc[0] if len(top_row) else "—"
top_score = top_row["Score"].iloc[0] if len(top_row) else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Prices loaded", f"{loaded}/{n}")
m2.metric("With fundamentals", f"{len(scored)}/{n}")
m3.metric("Avg fundamental score", f"{avg_score:.0f}/100")
m4.metric("Top score", f"{top_name} {top_score:.0f}")
st.divider()

available_sectors = [s for s in SECTOR_ORDER if s in df["Sector"].unique()]
fcol1, fcol2 = st.columns([2, 3])
with fcol1:
    rank_mode = st.radio(
        "Rank by",
        options=["Value / Quality", "Growth / Asymmetric", "Blended"],
        horizontal=True,
        help="Value = cheap + high return-on-capital (avoids value traps). "
             "Growth = hypergrowth characteristics (higher risk). Blended = average of both.",
    )
with fcol2:
    selected_sectors = st.multiselect(
        "Filter by sector", options=available_sectors, default=available_sectors,
        label_visibility="collapsed", placeholder="Filter by sector...",
    )

view = df[df["Sector"].isin(selected_sectors)].copy() if selected_sectors else df.copy()
view["Blend"] = view[["Score", "Growth Score"]].mean(axis=1, skipna=True)
sort_key = {"Value / Quality": "Score", "Growth / Asymmetric": "Growth Score", "Blended": "Blend"}[rank_mode]
view = view.sort_values(sort_key, ascending=False, na_position="last")


def render_display(row):
    target_str = fmt_price(row["Target"], row["Currency"]) if pd.notna(row["Target"]) else "—"
    if pd.notna(row["Target"]) and not row["Target OK"]:
        target_str += " ⚠️"
    return pd.Series({
        "Ticker": f"{row['Ticker']}  ({row['Region']})",
        "Name": row["Name"],
        "Sector": row["Sector"],
        "Value": f"{row['Score']:.0f}" if pd.notna(row["Score"]) else "—",
        "Growth": f"{row['Growth Score']:.0f}" if pd.notna(row["Growth Score"]) else "—",
        "Cov": f"{int(row['Coverage'])}/{len(FACTORS)}" if pd.notna(row["Coverage"]) else "—",
        "Price": fmt_price(row["Price"], row["Currency"]),
        "Day %": f"{row['Day %']:+.2f}%" if pd.notna(row["Day %"]) else "—",
        "Earn Yld": f"{row['Earnings Yld %']:.1f}%" if pd.notna(row["Earnings Yld %"]) else "—",
        "ROIC": f"{row['ROIC %']:.1f}%" if pd.notna(row["ROIC %"]) else "—",
        "Rev Grw": f"{row['Rev Growth %']:+.1f}%" if pd.notna(row["Rev Growth %"]) else "—",
        "Gross Mgn": f"{row['Gross Mgn %']:.0f}%" if pd.notna(row["Gross Mgn %"]) else "—",
        "Rule40": f"{row['Rule40']:.0f}" if pd.notna(row["Rule40"]) else "—",
        "FCF Yld": f"{row['FCF Yld %']:.1f}%" if pd.notna(row["FCF Yld %"]) else "—",
        "Mkt Cap": fmt_mcap(row["Mkt Cap"], row["Currency"]),
        "Target": target_str,
        "Upside %": f"{row['Upside %']:+.1f}%" if pd.notna(row["Upside %"]) else "—",
        "Thesis": row["Thesis"],
    })


display = view.apply(render_display, axis=1)


def color_score(val: str) -> str:
    if val == "—": return "color: #6b7280;"
    try:
        nval = float(val)
    except ValueError:
        return ""
    if nval >= 70: return "color: #047857; font-weight: 600;"
    if nval >= 45: return "color: #0369a1;"
    return "color: #6b7280;"


def color_signed(val: str) -> str:
    if val == "—": return "color: #6b7280;"
    try:
        nval = float(val.replace("%", "").replace("+", ""))
    except ValueError:
        return ""
    return "color: #059669;" if nval >= 0 else "color: #dc2626;"


styler = (
    display.style
    .map(color_score, subset=["Value"])
    .map(color_score, subset=["Growth"])
    .map(color_signed, subset=["Day %"])
    .map(color_signed, subset=["Rev Grw"])
)

st.dataframe(
    styler, use_container_width=True, hide_index=True,
    height=min(60 + 36 * len(display), 900),
    column_config={
        "Thesis": st.column_config.TextColumn(width="large"),
        "Name": st.column_config.TextColumn(width="medium"),
    },
)

st.divider()
csv = view.drop(columns=["Target OK", "Blend"], errors="ignore").to_csv(index=False).encode()
st.download_button(
    "📥 Download current view as CSV", data=csv,
    file_name=f"watchlist_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv",
)

with st.expander("ℹ️ Notes & caveats", expanded=False):
    st.markdown(
        f"""
- **Not financial advice.** Ranking reflects reported fundamentals, which describe the past;
  they don't predict returns. A high score is a starting point for research, not a buy signal.
- **Two scores, pick your lens.** *Value/Quality* = cheap + high return-on-capital (Greenblatt-style).
  *Growth/Asymmetric* = hypergrowth characteristics. The growth score does NOT find "the next Nvidia" —
  use it as a small, diversified basket to research, sized so any single zero doesn't hurt you.
- **How metrics are derived.** Revenue growth, gross/operating margin and Rule-of-40 are computed from
  raw income-statement figures (stable across FMP plans). ROIC / FCF yield come from `key-metrics-ttm`;
  earnings yield falls back to net income ÷ market cap, then to 1 ÷ P/E, so the column rarely blanks.
  *Rule-of-40 here = revenue growth % + (FCF margin if available, else operating margin) %.*
- **Free-tier budget.** Prices come from one `batch-quote` call; fundamentals are cached
  {FUND_TTL//86400} days and refreshed at most {FUND_BUDGET} tickers per load, stopping instantly if FMP
  returns "Limit Reach". A cold board fills over a few **Fetch more fundamentals** clicks, then a normal
  refresh costs ~2 calls. Raise the per-load budget with the `FMP_FUND_BUDGET` env var on a paid plan.
- **Pending vs blank.** "Pending" means *not fetched yet* (budget/quota), not "no data" — it fills on the
  next refresh. Genuinely unavailable metrics (e.g. some non-US tickers) stay "—".
"""
    )
