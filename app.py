"""
Stock Watchlist — Streamlit App  (free-tier-friendly, multi-factor scored)

Data layer (why this is free-tier safe)
---------------------------------------
The old version fired 3 calls/ticker (~267 for 89 tickers) in a burst — past the
FMP free tier's 250/day cap, so fundamentals came back blank. Now:
  • Prices: ONE batch-quote call (52w range, market cap, P/E, 50/200-day avgs).
  • Fundamentals: 2 calls/ticker, derived from raw statements (robust field names).
  • Disk cache (prices 6h, fundamentals 14d) + per-run budget + 429-abort, so a
    load never blows the quota and a warm refresh costs ~2 calls.

Scoring layer (sector-neutral, multi-factor)
--------------------------------------------
Five family sub-scores — Value, Quality, Growth, Momentum, Safety — each built by
percentile-ranking its metrics WITHIN sector (so banks rank against banks), then
equal-weighted and blended per a selectable lens. Momentum is free (52-week
position + price vs 200-day average from the quote); Safety is a Piotroski-style
health score from the income statement we already fetch. See fundamentals.py.
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

st.set_page_config(page_title="Stock Watchlist", page_icon="📈",
                   layout="wide", initial_sidebar_state="collapsed")

# ---------------------------------------------------------------------------
# Watchlist import (guarded)
# ---------------------------------------------------------------------------
try:
    from watchlist_data import WATCHLIST as _BASE_WATCHLIST, SECTOR_ORDER as _BASE_SECTORS
except ImportError:
    try:
        import watchlist_data as _wd
        _found = [n for n in dir(_wd) if not n.startswith("_")]
    except Exception as _e:  # noqa: BLE001
        _found = [f"(watchlist_data.py failed to import: {_e})"]
    st.error("**`watchlist_data.py` is missing `WATCHLIST` / `SECTOR_ORDER`.**\n\n"
             f"**Names found:** `{_found}`\n\n**Fix:** re-commit the complete file, then redeploy.")
    st.stop()

from watchlist_growth import GROWTH_WATCHLIST
from fundamentals import (
    fetch_fundamentals, compute_family_scores, composite_for_lens,
    compute_flags, target_is_sane, has_any, FMPRateLimitError,
    FAMILIES, LENSES, DEFAULT_LENS, CALLS_PER_TICKER,
)
from yahoo_fallback import fetch_quotes_yahoo, fetch_fundamentals_yahoo, HAVE_YF
import market_risk

_seen = {item["ticker"] for item in _BASE_WATCHLIST}
WATCHLIST = _BASE_WATCHLIST + [g for g in GROWTH_WATCHLIST if g["ticker"] not in _seen]
SECTOR_ORDER = _BASE_SECTORS + [s for s in ["Hypergrowth"] if s not in _BASE_SECTORS]
SYMBOLS = [item["ticker"] for item in WATCHLIST]
SECTORS = {item["ticker"]: item["sector"] for item in WATCHLIST}

# Family -> short column label shown in the table.
FAM_ABBR = {"Value": "V", "Quality": "Q", "Growth": "G", "Momentum": "M", "Safety": "S"}

FMP_BASE = "https://financialmodelingprep.com/stable"
QUOTE_TTL = 6 * 60 * 60
FUND_TTL = 14 * 24 * 60 * 60
YF_FUND_TTL = 3 * 24 * 60 * 60   # Yahoo-sourced fundamentals expire sooner so FMP can replace them
MARKET_TTL = 6 * 60 * 60

# Budget is in TICKERS per refresh; each ticker costs CALLS_PER_TICKER FMP calls.
# Default keeps one full refresh comfortably inside the 250-calls/day free tier
# (70 × 3 = 210, plus a handful of quote/market calls).
_DEFAULT_BUDGET = min(len(WATCHLIST), max(10, (250 - 10) // CALLS_PER_TICKER))
FUND_BUDGET = int(os.getenv("FMP_FUND_BUDGET", str(_DEFAULT_BUDGET)))
YF_BUDGET = int(os.getenv("YF_FUND_BUDGET", "60"))  # Yahoo fallback fills per run
BATCH_SIZE = 50
MAX_WORKERS = 3


def get_api_key() -> Optional[str]:
    try:
        return st.secrets["FMP_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.getenv("FMP_API_KEY")


# ---------------------------------------------------------------------------
# Disk-persistent cache
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
        except Exception:  # noqa: BLE001
            self.data = {}

    def get(self, key): return self.data.get(key)
    def set(self, key, value): self.data[key] = value

    def flush(self):
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(self.data, fh)
            os.replace(tmp, self.path)
        except Exception as e:  # noqa: BLE001
            logger.warning("cache flush failed (%s): %s", self.path, e)


@st.cache_resource
def quote_store(): return PersistentStore(os.path.join(_cache_dir(), "quotes.json"))
@st.cache_resource
def fund_store(): return PersistentStore(os.path.join(_cache_dir(), "fundamentals.json"))


def _fresh(entry, ttl): return bool(entry) and (time.time() - entry.get("ts", 0)) < ttl


def _fund_fresh(entry):
    """Yahoo-sourced entries use a shorter TTL so FMP data replaces them sooner."""
    ttl = YF_FUND_TTL if (entry or {}).get("src") == "yahoo" else FUND_TTL
    return _fresh(entry, ttl)


# ---------------------------------------------------------------------------
# Price layer — batch-quote (per-ticker fallback)
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
        "ma200": _num(row, "priceAvg200", "priceAverage200"),
        "pe": _num(row, "pe", "peTTM", "priceEarningsRatioTTM"),
        "eps": _num(row, "eps", "epsTTM"),
    }


def _batch_quote(symbols, api_key):
    r = requests.get(f"{FMP_BASE}/batch-quote",
                     params={"symbols": ",".join(symbols), "apikey": api_key}, timeout=25)
    if r.status_code == 429 or _looks_rate_limited(r.text):
        raise FMPRateLimitError(r.text[:120])
    if r.status_code != 200:
        logger.info("batch-quote HTTP %s — falling back to per-ticker", r.status_code)
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


def _single_quote(symbol, api_key):
    r = requests.get(f"{FMP_BASE}/quote", params={"symbol": symbol, "apikey": api_key}, timeout=12)
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


def fetch_quotes(api_key, symbols, force):
    """(quotes_by_symbol, n_fetched, n_from_cache, rate_limited)."""
    store = quote_store()
    now = time.time()
    quotes, need, from_cache = {}, [], 0
    for s in symbols:
        e = store.get(s)
        if _fresh(e, QUOTE_TTL) and not force:
            quotes[s] = e["data"]; from_cache += 1
        else:
            if e:
                quotes[s] = e["data"]
            need.append(s)

    fetched, rate_limited = 0, False
    if need:
        got = {}
        try:
            for chunk in (need[i:i + BATCH_SIZE] for i in range(0, len(need), BATCH_SIZE)):
                rows = _batch_quote(chunk, api_key)
                for row in rows:
                    if row.get("symbol"):
                        got[row["symbol"]] = _norm_quote(row)
                if not rows:  # batch unsupported on this plan -> per-ticker
                    for sym in chunk:
                        row = _single_quote(sym, api_key)
                        if row:
                            got[sym] = _norm_quote(row)
        except FMPRateLimitError:
            rate_limited = True
        except requests.RequestException as e:
            logger.warning("FMP quote fetch failed (%s) — will try Yahoo fallback", str(e)[:120])
        for sym, d in got.items():
            store.set(sym, {"data": d, "ts": now}); quotes[sym] = d; fetched += 1

        # Yahoo fallback: any symbol STILL without a price (FMP quota spent,
        # plan doesn't include it, etc.) gets filled keylessly and cached.
        missing = [s for s in need if (quotes.get(s) or {}).get("price") is None]
        if missing and HAVE_YF:
            for sym, d in fetch_quotes_yahoo(missing).items():
                store.set(sym, {"data": d, "ts": now, "src": "yahoo"})
                quotes[sym] = d; fetched += 1
        store.flush()
    return quotes, fetched, from_cache, rate_limited


# ---------------------------------------------------------------------------
# Fundamentals layer — disk-cached, per-run budget, rate-limit abort
# ---------------------------------------------------------------------------
def fetch_fundamentals_all(api_key, symbols, market_caps, force):
    """(funds_by_symbol, n_refreshed, pending_symbols, n_loaded, rate_limited)."""
    store = fund_store()
    now = time.time()
    funds = {}
    candidates = []
    for s in symbols:
        e = store.get(s)
        if e:
            funds[s] = e["data"]
        if force or not _fund_fresh(e):
            candidates.append((e.get("ts", 0.0) if e else 0.0, s))

    candidates.sort(key=lambda kv: kv[0])
    to_fetch = [s for _, s in candidates[:FUND_BUDGET]]

    stop = threading.Event()
    rate_limited = False
    refreshed = 0
    lock = threading.Lock()

    def work(sym):
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

    # ---- Yahoo fallback: fill tickers FMP couldn't deliver (quota, plan) ----
    yahoo_missing = [s for s in symbols
                     if s not in funds or not has_any(funds.get(s) or {})]
    if yahoo_missing and HAVE_YF:
        def yf_work(sym):
            return sym, fetch_fundamentals_yahoo(sym)
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(yf_work, s) for s in yahoo_missing[:YF_BUDGET]]
            for fut in as_completed(futs):
                sym, res = fut.result()
                if res and has_any(res):
                    store.set(sym, {"data": res, "ts": now, "src": "yahoo"})
                    funds[sym] = res
                    refreshed += 1
        store.flush()

    pending = [s for s in symbols if s not in funds]
    n_loaded = len([s for s in symbols if s in funds and has_any(funds[s])])
    return funds, refreshed, pending, n_loaded, rate_limited


@st.cache_data(ttl=MARKET_TTL, show_spinner=False)
def cached_market_context(api_key): return market_risk.get_market_context(api_key)


# ---------------------------------------------------------------------------
# Formatting
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


def build_dataframe(quotes, fundamentals, fam_scores) -> pd.DataFrame:
    rows = []
    for item in WATCHLIST:
        tk = item["ticker"]
        q = quotes.get(tk) or {}
        fund = fundamentals.get(tk) or {}
        fs = fam_scores.get(tk) or {}
        price, lo, hi = q.get("price"), q.get("yearLow"), q.get("yearHigh")
        currency = REGION_CURRENCY.get(item.get("region", "US"), "USD")

        target = item.get("target")
        target_ok = target_is_sane(target, lo, hi)
        upside = ((target - price) / price) * 100 if (price and target and target_ok) else None

        def pct(key):
            v = fund.get(key)
            return v * 100 if v is not None else None

        def sub(fam):
            s = (fs.get(fam) or {}).get("score")
            return s * 100 if s is not None else None

        flag_adj, flag_list = compute_flags(fund)

        rows.append({
            "Ticker": tk, "Region": item["region"], "Name": item["name"], "Sector": item["sector"],
            "Price": price, "Day %": q.get("changePercentage"),
            "52w Low": lo, "52w High": hi,
            "52w Pos %": ((price - lo) / (hi - lo) * 100) if (price and lo and hi and hi > lo) else None,
            "Mkt Cap": q.get("marketCap"), "Currency": currency,
            # family sub-scores (lens-independent)
            "V": sub("Value"), "Q": sub("Quality"), "G": sub("Growth"),
            "M": sub("Momentum"), "S": sub("Safety"),
            # raw metrics
            "Earnings Yld %": pct("earnings_yield"), "ROIC %": pct("roic"),
            "Rev Growth %": pct("rev_growth"), "Gross Mgn %": pct("gross_margin"),
            "Rule40": fund.get("rule_of_40"), "FCF Yld %": pct("fcf_yield"),
            "Safety %": pct("safety"),
            "P/S": fund.get("ps_ratio"), "PEG": fund.get("peg"),
            "D/E": fund.get("debt_equity"),
            "FlagAdj": flag_adj, "Flags": " · ".join(flag_list) if flag_list else "",
            "Target": target, "Target OK": target_ok, "Upside %": upside,
            "Thesis": item["thesis"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("📈 Stock Watchlist")
st.caption(
    f"{len(WATCHLIST)} tickers, {len(SECTOR_ORDER)} sectors. Scored on five **sector-neutral** factor "
    f"families — Value · Quality · Growth · Momentum · Safety — blended by the lens you pick, then "
    f"adjusted by absolute red/green flags (debt, falling revenue, PEG, net cash). "
    f"Data: FMP first (cached to stay inside the free tier)"
    + (", Yahoo Finance fills anything FMP can't deliver — no more empty cells." if HAVE_YF
       else ". Install `yfinance` to auto-fill cells when the FMP quota is hit.")
)

api_key = get_api_key()
if not api_key:
    st.error("Missing FMP API key. Set `FMP_API_KEY` in Streamlit secrets or as an env var. See README.md.")
    st.stop()

if "force_refresh" not in st.session_state:
    st.session_state.force_refresh = False

col_a, col_b, col_c = st.columns([1.2, 1.4, 4])
with col_a:
    if st.button("🔄 Refresh prices", use_container_width=True,
                 help="Refetch live prices now (1–2 API calls). Fundamentals use the 14-day cache."):
        st.session_state.force_refresh = "prices"; st.rerun()
with col_b:
    if st.button("📊 Fetch more fundamentals", use_container_width=True,
                 help=f"Pull the next {FUND_BUDGET} tickers' fundamentals into the cache."):
        st.session_state.force_refresh = "funds"; st.rerun()
with col_c:
    st.caption(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · prices cached {QUOTE_TTL//3600}h · "
               f"fundamentals cached {FUND_TTL//86400}d · budget {FUND_BUDGET}/refresh")

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

# Merge in cheap, quote-derived signals: earnings-yield fallback + Momentum.
for s in SYMBOLS:
    f = fundamentals.get(s)
    if f is None:
        f = {}; fundamentals[s] = f
    q = quotes.get(s) or {}
    price, lo, hi, ma200, pe = (q.get("price"), q.get("yearLow"), q.get("yearHigh"),
                                q.get("ma200"), q.get("pe"))
    if f.get("earnings_yield") is None and pe and pe > 0:
        f["earnings_yield"] = 1.0 / pe
    if price and lo is not None and hi and hi > lo:
        f["mom_52w"] = (price - lo) / (hi - lo)          # 0..1 position in 52w range
    if price and ma200:
        f["mom_ma200"] = price / ma200 - 1               # trend vs 200-day average
    # PEG fallback from the quote's P/E + statement EPS growth (both must be > 0).
    eg = f.get("eps_growth")
    if f.get("peg") is None and pe and pe > 0 and eg and eg > 0:
        f["peg"] = pe / (eg * 100.0)
    # Safety fallback for Yahoo-sourced rows (no 2-year statement history there):
    if f.get("safety") is None:
        checks = []
        for key, ok in (("net_margin", lambda v: v > 0), ("operating_margin", lambda v: v > 0),
                        ("fcf_yield", lambda v: v > 0), ("rev_growth", lambda v: v > 0)):
            v = f.get(key)
            if v is not None:
                checks.append(ok(v))
        if checks:
            f["safety"] = sum(1 for c in checks if c) / len(checks)

fam_scores = compute_family_scores(fundamentals, SECTORS)

# ---------------------------------------------------------------------------
# Market conditions
# ---------------------------------------------------------------------------
mkt = cached_market_context(api_key)
with st.container():
    st.subheader("🌡️ Market conditions")
    st.caption("Context for **how** to deploy (position size · scaling · rebalancing) — **not** a buy/sell signal.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("S&P 500", f"{mkt['index_price']:,.0f}" if mkt["index_price"] else "—",
              mkt["trend"] if mkt["trend"] != "unknown" else None)
    c2.metric("vs 200-day avg", f"{mkt['pct_from_ma200']:+.1f}%" if mkt["pct_from_ma200"] is not None else "—")
    c3.metric("Index 52w position", f"{mkt['index_52w_pos']:.0f}%" if mkt["index_52w_pos"] is not None else "—")
    c4.metric("VIX (volatility)", f"{mkt['vix']:.1f}" if mkt["vix"] else "—",
              mkt["vix_label"] if mkt["vix_label"] != "unknown" else None)
    if mkt.get("vix_context"):
        st.caption(f"**Volatility regime — {mkt['vix_label']}:** {mkt['vix_context']}")
    for note in mkt.get("notes", []):
        st.info(note)
st.divider()

# ---------------------------------------------------------------------------
# Status banner
# ---------------------------------------------------------------------------
n = len(SYMBOLS)
if q_rl or f_rl or mkt.get("rate_limited"):
    extra = ("Yahoo Finance fallback filled the gaps where possible." if HAVE_YF
             else "Install `yfinance` to auto-fill gaps when this happens.")
    st.warning(f"⏳ **FMP quota reached** — showing cached values. {f_loaded}/{n} tickers have fundamentals; "
               f"**{len(f_pending)} pending**. {extra}")
elif f_pending:
    st.info(f"📊 Fundamentals: **{f_loaded}/{n} loaded** ({f_refreshed} this load) · **{len(f_pending)} pending**. "
            f"Click **Fetch more fundamentals** to pull the next batch — then they stay cached {FUND_TTL//86400} days.")
else:
    st.success(f"✅ All {n} tickers loaded — prices live, fundamentals cached.")

df = build_dataframe(quotes, fundamentals, fam_scores)

flagged = df[(df["Target"].notna()) & (~df["Target OK"])]
if len(flagged):
    def _fmt_flag(row):
        hi = row["52w High"]; hi_str = f"{hi:g}" if pd.notna(hi) else "n/a"
        return f"{row['Ticker']} (target {row['Target']:g} vs 52w high {hi_str})"
    names = ", ".join(_fmt_flag(r) for _, r in flagged.iterrows())
    st.error(f"⚠️ {len(flagged)} static target(s) implausible vs the live 52-week range — excluded from upside "
             f"(likely a split or stale entry in `watchlist_data.py`): {names}")

# ---------------------------------------------------------------------------
# Controls — lens + sector filter
# ---------------------------------------------------------------------------
available_sectors = [s for s in SECTOR_ORDER if s in df["Sector"].unique()]
fcol1, fcol2 = st.columns([2.2, 3])
with fcol1:
    lens_names = list(LENSES.keys())
    rank_mode = st.radio(
        "Lens (how the headline Score is blended)",
        options=lens_names, index=lens_names.index(DEFAULT_LENS), horizontal=True,
        help="All five sub-scores (V/Q/G/M/S) always show. The lens only sets the blended **Score** and sort. "
             "Value=cheap · Quality=high return-on-capital · Growth=fast top-line · Momentum=trending up · Safety=financially healthy.",
    )
with fcol2:
    selected_sectors = st.multiselect("Filter by sector", options=available_sectors, default=available_sectors,
                                      label_visibility="collapsed", placeholder="Filter by sector...")

view = df[df["Sector"].isin(selected_sectors)].copy() if selected_sectors else df.copy()

# Composite Score for the chosen lens = weighted mean of the family sub-scores
# present, then adjusted by the absolute red/green flags (clipped to 0–100).
weights = LENSES[rank_mode]
def _composite(row):
    num = den = 0.0; present = 0
    for fam, w in weights.items():
        v = row[FAM_ABBR[fam]]
        if pd.notna(v):
            num += w * v; den += w; present += 1
    base = (num / den) if den > 0 else float("nan")
    adj = row["FlagAdj"] if pd.notna(row.get("FlagAdj", float("nan"))) else 0.0
    score = min(100.0, max(0.0, base + adj)) if pd.notna(base) else base
    return pd.Series({"Score": score, "Cov": present, "CovN": len(weights)})

view[["Score", "Cov", "CovN"]] = view.apply(_composite, axis=1)
view = view.sort_values("Score", ascending=False, na_position="last")

# Summary (reflects the chosen lens)
loaded = df["Price"].notna().sum()
with_data = int(df[["V", "Q", "G", "M", "S"]].notna().any(axis=1).sum())
scored = view[view["Score"].notna()]
avg_score = scored["Score"].mean() if len(scored) else 0
top = scored.head(1)
top_name = top["Ticker"].iloc[0] if len(top) else "—"
top_score = top["Score"].iloc[0] if len(top) else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Prices loaded", f"{loaded}/{n}")
m2.metric("With fundamentals", f"{with_data}/{n}")
m3.metric(f"Avg score ({rank_mode.split(' ')[0]})", f"{avg_score:.0f}/100")
m4.metric("Top", f"{top_name} {top_score:.0f}")
st.divider()


def _i(v):  # 0–100 integer or em-dash
    return f"{v:.0f}" if pd.notna(v) else "—"


def render_display(row):
    target_str = fmt_price(row["Target"], row["Currency"]) if pd.notna(row["Target"]) else "—"
    if pd.notna(row["Target"]) and not row["Target OK"]:
        target_str += " ⚠️"
    return pd.Series({
        "Ticker": f"{row['Ticker']}  ({row['Region']})",
        "Name": row["Name"],
        "Sector": row["Sector"],
        "Score": _i(row["Score"]),
        "Cov": f"{int(row['Cov'])}/{int(row['CovN'])}" if pd.notna(row["Cov"]) else "—",
        "V": _i(row["V"]), "Q": _i(row["Q"]), "G": _i(row["G"]), "M": _i(row["M"]), "S": _i(row["S"]),
        "Price": fmt_price(row["Price"], row["Currency"]),
        "Day %": f"{row['Day %']:+.2f}%" if pd.notna(row["Day %"]) else "—",
        "Earn Yld": f"{row['Earnings Yld %']:.1f}%" if pd.notna(row["Earnings Yld %"]) else "—",
        "ROIC": f"{row['ROIC %']:.1f}%" if pd.notna(row["ROIC %"]) else "—",
        "Rev Grw": f"{row['Rev Growth %']:+.1f}%" if pd.notna(row["Rev Growth %"]) else "—",
        "Gross Mgn": f"{row['Gross Mgn %']:.0f}%" if pd.notna(row["Gross Mgn %"]) else "—",
        "FCF Yld": f"{row['FCF Yld %']:.1f}%" if pd.notna(row["FCF Yld %"]) else "—",
        "P/S": f"{row['P/S']:.1f}" if pd.notna(row["P/S"]) else "—",
        "PEG": f"{row['PEG']:.2f}" if pd.notna(row["PEG"]) else "—",
        "D/E": f"{row['D/E']:.2f}" if pd.notna(row["D/E"]) else "—",
        "Flags": row["Flags"] if row["Flags"] else "—",
        "52w": f"{row['52w Pos %']:.0f}%" if pd.notna(row["52w Pos %"]) else "—",
        "Mkt Cap": fmt_mcap(row["Mkt Cap"], row["Currency"]),
        "Target": target_str,
        "Upside %": f"{row['Upside %']:+.1f}%" if pd.notna(row["Upside %"]) else "—",
        "Thesis": row["Thesis"],
    })


display = view.apply(render_display, axis=1)


def color_score(val):
    if val == "—": return "color: #6b7280;"
    try:
        nv = float(val)
    except ValueError:
        return ""
    if nv >= 70: return "color: #047857; font-weight: 600;"
    if nv >= 45: return "color: #0369a1;"
    return "color: #6b7280;"


def color_signed(val):
    if val == "—": return "color: #6b7280;"
    try:
        nv = float(val.replace("%", "").replace("+", ""))
    except ValueError:
        return ""
    return "color: #059669;" if nv >= 0 else "color: #dc2626;"


styler = display.style
for c in ["Score", "V", "Q", "G", "M", "S"]:
    styler = styler.map(color_score, subset=[c])
styler = styler.map(color_signed, subset=["Day %"]).map(color_signed, subset=["Rev Grw"])

st.dataframe(
    styler, use_container_width=True, hide_index=True,
    height=min(60 + 36 * len(display), 900),
    column_config={"Thesis": st.column_config.TextColumn(width="large"),
                   "Flags": st.column_config.TextColumn(width="medium"),
                   "Name": st.column_config.TextColumn(width="medium")},
)

st.divider()
csv = view.drop(columns=["Target OK"], errors="ignore").to_csv(index=False).encode()
st.download_button("📥 Download current view as CSV", data=csv,
                   file_name=f"watchlist_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")

with st.expander("ℹ️ How the score works & caveats", expanded=False):
    st.markdown(
        f"""
**Five sub-scores, each 0–100 (percentile vs peers):**

- **V — Value:** earnings yield + FCF yield + **low P/S** + **low PEG** (cheapness, growth-adjusted).
- **Q — Quality:** ROIC + gross margin + operating margin (profitability / returns on capital).
- **G — Growth:** revenue growth + EPS growth + Rule-of-40.
- **M — Momentum:** position in the 52-week range + price vs the 200-day average (free from the quote).
- **S — Safety:** Piotroski-style health check + **low debt/equity** + **low net-debt/EBITDA** + interest coverage.

**Logic:** each metric is percentile-ranked **within its sector** (so a bank ranks against banks, not against
software), then equal-weighted into its family. The headline **Score** blends families per the **lens** you
pick — equal weights on purpose, since tuned weights overfit. Sub-scores renormalize over only the metrics a
ticker actually has; **Cov** shows how many of the lens's families were available.

**Flags (absolute checks, applied on top of the relative ranks):** percentiles only say "better than peers" —
flags catch what's bad or great in absolute terms. Red flags subtract points: falling revenue (−6 to −12),
D/E > 1 or 2 (−4/−8), net debt > 3–4× EBITDA (−4/−8), interest coverage < 2× (−6), negative FCF (−5),
unprofitable (−4), and high-debt-plus-falling-revenue (−5, the classic value trap). Green flags add points:
PEG < 1 (+6), P/S < 2 with growing revenue (+4), ROIC > 20% (+5), net cash (+4). Capped at −25/+15.

**Why QARP is the default lens:** cheapness alone finds value traps (cheap because dying); quality alone
overpays. The most evidence-backed simple recipe — Greenblatt's Magic Formula and the academic
quality-minus-junk literature — is to demand **both**: high earnings/FCF yield AND high ROIC, with a clean
balance sheet. QARP = Value 1.0 × Quality 1.0 × Safety 0.75 × Growth 0.25, flag-adjusted.

**Caveats.**
- Not financial advice. This ranks **past reported data** — factor premia are real but noisy and can
  underperform for years. A high score is a research starting point, not a buy signal.
- **Banks / REITs:** sector-neutral ranking compares them fairly to peers, but ROIC/FCF-yield are imperfect for
  them — read those rows with extra care.
- **Thin sectors** (fewer than {5} rated names) fall back to whole-universe ranking.
- **Free-tier budget:** prices = 1 batch call; fundamentals = {CALLS_PER_TICKER} calls/ticker, cached
  {FUND_TTL//86400}d, ≤{FUND_BUDGET} tickers refreshed per load, stopping on "Limit Reach".
  Momentum adds **no** API calls. Anything FMP can't deliver is filled from **Yahoo Finance**
  (keyless, cached {YF_FUND_TTL//86400}d so FMP data replaces it when quota allows).
"""
    )
