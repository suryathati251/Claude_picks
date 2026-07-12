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
    compute_flags, safety_gate, target_is_sane, has_any, FMPRateLimitError,
    FAMILIES, LENSES, DEFAULT_LENS, CALLS_PER_TICKER,
)
from yahoo_fallback import (
    fetch_quotes_yahoo, fetch_fundamentals_yahoo, fetch_momentum_yahoo, HAVE_YF,
)
from tenx_universe import SCAN_SYMBOLS, SCAN_NAMES, SCAN_SECTORS
from tenx_radar import (fetch_quarterly_yahoo, compute_tenx_metrics, tenx_score,
                        fetch_next_earnings)
from entry_meter import fetch_entry_history_yahoo, compute_entry_meter
from insider import fetch_insider, insider_display
import market_risk

_seen = {item["ticker"] for item in _BASE_WATCHLIST}
WATCHLIST = _BASE_WATCHLIST + [g for g in GROWTH_WATCHLIST if g["ticker"] not in _seen]
SECTOR_ORDER = _BASE_SECTORS + [s for s in ["Hypergrowth"] if s not in _BASE_SECTORS]
SYMBOLS = [item["ticker"] for item in WATCHLIST]
SECTORS = {item["ticker"]: item["sector"] for item in WATCHLIST}

# Family -> short column label shown in the table.
FAM_ABBR = {"Value": "V", "Quality": "Q", "Growth": "G", "Momentum": "M", "Safety": "S", "Moat": "Moat"}

FMP_BASE = "https://financialmodelingprep.com/stable"
QUOTE_TTL = 6 * 60 * 60
FUND_TTL = 14 * 24 * 60 * 60
YF_FUND_TTL = 3 * 24 * 60 * 60   # Yahoo-sourced fundamentals expire sooner so FMP can replace them
MARKET_TTL = 6 * 60 * 60
MOM_TTL = 24 * 60 * 60           # 12-1 momentum moves slowly; refresh once a day
TENX_TTL = 21 * 24 * 60 * 60     # quarterly revenue data — a new print lands ~every 90d
TENX_RETRY_TTL = 24 * 60 * 60    # failed quarterly fetches retry after a day
SCAN_QUOTE_TTL = 24 * 60 * 60    # scan-universe quotes only need daily freshness

# Budget is in TICKERS per refresh; each ticker costs CALLS_PER_TICKER FMP calls.
# Default keeps one full refresh comfortably inside the 250-calls/day free tier
# (70 × 3 = 210, plus a handful of quote/market calls).
_DEFAULT_BUDGET = min(len(WATCHLIST), max(10, (250 - 10) // CALLS_PER_TICKER))
FUND_BUDGET = int(os.getenv("FMP_FUND_BUDGET", str(_DEFAULT_BUDGET)))
YF_BUDGET = int(os.getenv("YF_FUND_BUDGET", "60"))  # Yahoo fallback fills per run
BATCH_SIZE = 50
MAX_WORKERS = 3

# ---- 10x Radar universe: watchlist + S&P 500 + Nasdaq-100 + curated extras ----
# Quarterly data comes from Yahoo (keyless — zero FMP quota); scan-universe quotes
# use cheap FMP batch calls (~1 per 50 tickers per day) with a Yahoo bulk fallback.
SCAN_ONLY_SYMBOLS = [s for s in SCAN_SYMBOLS if s not in set(SYMBOLS)]
TENX_SYMBOLS = SYMBOLS + SCAN_ONLY_SYMBOLS
TENX_NAMES = {**SCAN_NAMES, **{it["ticker"]: it["name"] for it in WATCHLIST}}
TENX_SECTORS = {**SCAN_SECTORS, **SECTORS}
TENX_AUTO_BUDGET = int(os.getenv("TENX_SCAN_BUDGET", "25"))   # quarterly fetches per load
TENX_CLICK_BUDGET = 75                                        # per "Scan next batch" click
EARN_TTL = 3 * 24 * 60 * 60          # next-earnings dates (radar's shown rows only)
INSIDER_TTL = 14 * 24 * 60 * 60      # insider summaries change slowly
INSIDER_BUDGET = int(os.getenv("INSIDER_BUDGET", "8"))   # FMP calls/load (watchlist only)
HISTORY_RETAIN_DAYS = 180            # daily score snapshots kept on disk


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
@st.cache_resource
def momentum_store(): return PersistentStore(os.path.join(_cache_dir(), "momentum.json"))
@st.cache_resource
def tenx_store(): return PersistentStore(os.path.join(_cache_dir(), "tenx.json"))
@st.cache_resource
def earnings_store(): return PersistentStore(os.path.join(_cache_dir(), "earnings.json"))
@st.cache_resource
def insider_store(): return PersistentStore(os.path.join(_cache_dir(), "insider.json"))
@st.cache_resource
def history_store(): return PersistentStore(os.path.join(_cache_dir(), "history.json"))


# ---------------------------------------------------------------------------
# Daily score history (risers, 🆕 entered-top-10 flags, entry-meter trend).
# Note: on Streamlit Cloud the disk survives normal traffic but resets on
# redeploys/reboots, so history re-accumulates from there — it's a convenience
# trail, not an audit log.
# ---------------------------------------------------------------------------
def record_history(kind: str, values):
    """Write today's snapshot for `kind` once per day (values: dict or float)."""
    store = history_store()
    h = store.data.setdefault(kind, {})
    today = datetime.now().strftime("%Y-%m-%d")
    if today in h:
        return
    h[today] = values
    for d in sorted(h)[:-HISTORY_RETAIN_DAYS]:
        del h[d]
    store.flush()


def history_snapshot(kind: str, min_age_days: int = 5):
    """(date, values) of the newest snapshot at least `min_age_days` old,
    else (None, None). Used for Δ columns and 'a week ago' comparisons."""
    h = history_store().data.get(kind) or {}
    cutoff = (datetime.now() - pd.Timedelta(days=min_age_days)).strftime("%Y-%m-%d")
    olds = [d for d in h if d <= cutoff]
    if not olds:
        return None, None
    d = max(olds)
    return d, h[d]


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
            return sym, fetch_fundamentals_yahoo(sym, market_caps.get(sym))
        with ThreadPoolExecutor(max_workers=2) as ex:
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


def fetch_momentum(symbols, force):
    """Disk-cached 12-1 month momentum from Yahoo (free, no FMP calls). Returns
    {symbol: mom_12_1|None}. Refreshes only stale entries (daily)."""
    if not HAVE_YF:
        return {}
    store = momentum_store()
    now = time.time()
    out, need = {}, []
    for s in symbols:
        e = store.get(s)
        if _fresh(e, MOM_TTL) and not force:
            out[s] = e["data"]
        else:
            need.append(s)
    if need:
        try:
            res = fetch_momentum_yahoo(need)
        except Exception as e:  # noqa: BLE001
            logger.warning("momentum fetch failed: %s", str(e)[:120]); res = {}
        for sym, d in res.items():
            val = (d or {}).get("mom_12_1")
            store.set(sym, {"data": val, "ts": now})
            out[sym] = val
        store.flush()
    return out


def fetch_scan_quotes(api_key, symbols):
    """Quotes for the SCAN universe (market cap + 52w range for the 10x Radar).
    Deliberately batch-only on FMP — no per-ticker fallback, so a plan that
    rejects batch quotes can never burn hundreds of calls — with a keyless
    Yahoo bulk download filling whatever is missing. Cached 24h."""
    store = quote_store()
    now = time.time()
    quotes, need = {}, []
    for s in symbols:
        e = store.get(s)
        if _fresh(e, SCAN_QUOTE_TTL):
            quotes[s] = e["data"]
        else:
            if e:
                quotes[s] = e["data"]
            need.append(s)
    if not need:
        return quotes
    got = {}
    try:
        for chunk in (need[i:i + BATCH_SIZE] for i in range(0, len(need), BATCH_SIZE)):
            for row in _batch_quote(chunk, api_key):
                if row.get("symbol"):
                    got[row["symbol"]] = _norm_quote(row)
    except FMPRateLimitError:
        logger.info("scan quotes: FMP rate-limited — Yahoo will fill")
    except requests.RequestException as e:
        logger.warning("scan quotes failed (%s) — Yahoo will fill", str(e)[:120])
    missing = [s for s in need
               if s not in got and (quotes.get(s) or {}).get("price") is None]
    if missing and HAVE_YF:
        got.update(fetch_quotes_yahoo(missing))
    for sym, d in got.items():
        store.set(sym, {"data": d, "ts": now})
        quotes[sym] = d
    if got:
        store.flush()
    return quotes


def fetch_tenx_all(symbols, budget):
    """Quarterly-revenue radar metrics, disk-cached, stalest-first, budgeted.
    Yahoo-only (keyless) so the radar never touches the FMP quota.
    Returns (metrics_by_symbol, n_refreshed, n_with_data)."""
    store = tenx_store()
    now = time.time()
    metrics, candidates = {}, []
    for s in symbols:
        e = store.get(s)
        if e:
            metrics[s] = e["data"]
        ttl = TENX_RETRY_TTL if (e or {}).get("fail") else TENX_TTL
        if not _fresh(e, ttl):
            candidates.append((e.get("ts", 0.0) if e else 0.0, s))
    candidates.sort(key=lambda kv: kv[0])
    to_fetch = [s for _, s in candidates[:max(0, budget)]] if HAVE_YF else []

    refreshed = 0
    if to_fetch:
        def work(sym):
            return sym, compute_tenx_metrics(fetch_quarterly_yahoo(sym))
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = [ex.submit(work, s) for s in to_fetch]
            for fut in as_completed(futs):
                sym, res = fut.result()
                ok = res.get("q_rev_yoy") is not None or (res.get("n_quarters") or 0) > 0
                store.set(sym, {"data": res, "ts": now, "fail": not ok})
                metrics[sym] = res
                refreshed += 1
        store.flush()

    n_have = sum(1 for s in symbols
                 if (metrics.get(s) or {}).get("q_rev_yoy") is not None)
    return metrics, refreshed, n_have


def fetch_earnings_dates(symbols):
    """{sym: 'YYYY-MM-DD'|None} for the radar's DISPLAYED rows only (≤ ~40),
    via Yahoo's calendar (keyless), disk-cached 3 days."""
    if not HAVE_YF:
        return {}
    store = earnings_store()
    now = time.time()
    out, need = {}, []
    for s in symbols:
        e = store.get(s)
        if _fresh(e, EARN_TTL):
            out[s] = e["data"]
        else:
            need.append(s)
    for s in need:
        val = fetch_next_earnings(s)
        store.set(s, {"data": val, "ts": now})
        out[s] = val
    if need:
        store.flush()
    return out


def fmt_earnings(iso: Optional[str]) -> str:
    """'Jul 29 · 17d' with ⚠️ inside 7 days; '—' when unknown."""
    if not iso:
        return "—"
    try:
        d = datetime.strptime(iso, "%Y-%m-%d").date()
    except ValueError:
        return "—"
    days = (d - datetime.now().date()).days
    label = d.strftime("%b %d")
    if days < 0:
        return f"{label} (past)"
    warn = " ⚠️" if days <= 7 else ""
    return f"{label} · {days}d{warn}"


def fetch_insider_all(api_key, symbols, budget):
    """Insider summaries for WATCHLIST tickers, stalest-first, small FMP budget,
    cached 14d. Returns (by_symbol, plan_unsupported)."""
    store = insider_store()
    now = time.time()
    data, candidates = {}, []
    for s in symbols:
        e = store.get(s)
        if e:
            data[s] = e["data"]
        if not _fresh(e, INSIDER_TTL):
            candidates.append((e.get("ts", 0.0) if e else 0.0, s))
    candidates.sort(key=lambda kv: kv[0])
    unsupported = any((d or {}).get("unsupported") for d in data.values())
    if not unsupported:
        fetched = 0
        for _, s in candidates:
            if fetched >= max(0, budget):
                break
            try:
                res = fetch_insider(s, api_key)
            except FMPRateLimitError:
                break
            store.set(s, {"data": res, "ts": now})
            data[s] = res
            fetched += 1
            if res.get("unsupported"):
                unsupported = True
                break
        if fetched:
            store.flush()
    return data, unsupported


@st.cache_data(ttl=MARKET_TTL, show_spinner=False)
def cached_market_context(api_key): return market_risk.get_market_context(api_key)


@st.cache_data(ttl=MARKET_TTL, show_spinner=False)
def cached_entry_history(): return fetch_entry_history_yahoo()


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
            "M": sub("Momentum"), "S": sub("Safety"), "Moat": sub("Moat"),
            # moat raw signals (CSV / detail)
            "GM 5y %": pct("gross_margin_avg"), "Margin Stab %": pct("margin_stability"),
            "Growth Consist %": pct("growth_consistency"),
            # raw metrics
            "Earnings Yld %": pct("earnings_yield"), "ROIC %": pct("roic"),
            "Rev Growth %": pct("rev_growth"), "Gross Mgn %": pct("gross_margin"),
            "Rule40": fund.get("rule_of_40"), "FCF Yld %": pct("fcf_yield"),
            "Safety %": pct("safety"),
            "Mom 12-1 %": pct("mom_12_1"), "Gross Prof %": pct("gross_profitability"),
            "P/S": fund.get("ps_ratio"), "PEG": fund.get("peg"),
            "EV/EBIT": fund.get("ev_ebit"), "D/E": fund.get("debt_equity"),
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
    + f" The **🚀 10x Radar** tab scans {len(TENX_SYMBOLS)} names (watchlist + S&P 500 + "
      f"Nasdaq-100) for exploding, accelerating quarterly revenues."
)

api_key = get_api_key()
if not api_key:
    st.error("Missing FMP API key. Set `FMP_API_KEY` in Streamlit secrets or as an env var. See README.md.")
    st.stop()

if "force_refresh" not in st.session_state:
    st.session_state.force_refresh = False

col_a, col_b, col_c = st.columns([1.2, 1.4, 4])
with col_a:
    if st.button("🔄 Refresh prices", width="stretch",
                 help="Refetch live prices now (1–2 API calls). Fundamentals use the 14-day cache."):
        st.session_state.force_refresh = "prices"; st.rerun()
with col_b:
    if st.button("📊 Fetch more fundamentals", width="stretch",
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

# Real 12-1 month momentum (free, Yahoo, cached daily). Empty if yfinance absent.
with st.spinner("Loading momentum…"):
    mom_12_1 = fetch_momentum(SYMBOLS, force_funds)

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
    if mom_12_1.get(s) is not None:
        f["mom_12_1"] = mom_12_1[s]                       # canonical 12-1m momentum
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

mkt = cached_market_context(api_key)

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
        nv = float(val.replace("%", "").replace("+", "").replace("pt", ""))
    except ValueError:
        return ""
    return "color: #059669;" if nv >= 0 else "color: #dc2626;"


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
NAV_WATCH, NAV_RADAR, NAV_MARKET, NAV_LOOKUP = (
    "📊 Watchlist", "🚀 10x Radar", "🎯 Market & Entry", "🔎 Lookup")
nav = st.radio("section", [NAV_WATCH, NAV_RADAR, NAV_MARKET, NAV_LOOKUP],
               horizontal=True, key="nav", label_visibility="collapsed")

if nav == NAV_WATCH:
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
            options=lens_names, index=lens_names.index(DEFAULT_LENS), horizontal=True, key="lens",
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
        # Safety gate: a gentle multiplier from the absolute health fraction, so a
        # fragile balance sheet caps the score even in lenses that ignore Safety.
        s_raw = row.get("Safety %")
        gate = safety_gate(s_raw / 100.0) if pd.notna(s_raw) else 1.0
        score = min(100.0, max(0.0, base * gate + adj)) if pd.notna(base) else base
        return pd.Series({"Score": score, "Cov": present, "CovN": len(weights)})

    view[["Score", "Cov", "CovN"]] = view.apply(_composite, axis=1)
    view = view.sort_values("Score", ascending=False, na_position="last")

    # Daily composite-score snapshot (powers future risers/fallers views).
    _w_scores = {r["Ticker"]: round(r["Score"], 1) for _, r in view.iterrows() if pd.notna(r["Score"])}
    if _w_scores:
        record_history("watch", _w_scores)

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


    # Insider signal (small separate FMP budget, cached 14d, watchlist only).
    insider_data, insider_unsupported = fetch_insider_all(api_key, SYMBOLS, INSIDER_BUDGET)
    if insider_unsupported:
        st.caption("ℹ️ Insider data isn't included in this FMP plan — the **Insider** column stays "
                   "blank and no quota is spent retrying.")


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
            "Moat": _i(row["Moat"]),
            "Price": fmt_price(row["Price"], row["Currency"]),
            "Day %": f"{row['Day %']:+.2f}%" if pd.notna(row["Day %"]) else "—",
            "Earn Yld": f"{row['Earnings Yld %']:.1f}%" if pd.notna(row["Earnings Yld %"]) else "—",
            "ROIC": f"{row['ROIC %']:.1f}%" if pd.notna(row["ROIC %"]) else "—",
            "Rev Grw": f"{row['Rev Growth %']:+.1f}%" if pd.notna(row["Rev Growth %"]) else "—",
            "Gross Mgn": f"{row['Gross Mgn %']:.0f}%" if pd.notna(row["Gross Mgn %"]) else "—",
            "FCF Yld": f"{row['FCF Yld %']:.1f}%" if pd.notna(row["FCF Yld %"]) else "—",
            "P/S": f"{row['P/S']:.1f}" if pd.notna(row["P/S"]) else "—",
            "PEG": f"{row['PEG']:.2f}" if pd.notna(row["PEG"]) else "—",
            "EV/EBIT": ("n/m" if (pd.notna(row["EV/EBIT"]) and row["EV/EBIT"] >= 999)
                        else (f"{row['EV/EBIT']:.1f}" if pd.notna(row["EV/EBIT"]) else "—")),
            "D/E": f"{row['D/E']:.2f}" if pd.notna(row["D/E"]) else "—",
            "Flags": row["Flags"] if row["Flags"] else "—",
            "Insider": insider_display(insider_data.get(row["Ticker"]), row["52w Pos %"]),
            "12-1m": f"{row['Mom 12-1 %']:+.0f}%" if pd.notna(row["Mom 12-1 %"]) else "—",
            "52w": f"{row['52w Pos %']:.0f}%" if pd.notna(row["52w Pos %"]) else "—",
            "Mkt Cap": fmt_mcap(row["Mkt Cap"], row["Currency"]),
            "Target": target_str,
            "Upside %": f"{row['Upside %']:+.1f}%" if pd.notna(row["Upside %"]) else "—",
            "Thesis": row["Thesis"],
        })


    display = view.apply(render_display, axis=1)


    styler = display.style
    for c in ["Score", "V", "Q", "G", "M", "S", "Moat"]:
        styler = styler.map(color_score, subset=[c])
    styler = (styler.map(color_signed, subset=["Day %"])
                    .map(color_signed, subset=["Rev Grw"])
                    .map(color_signed, subset=["12-1m"]))

    st.dataframe(
        styler, width="stretch", hide_index=True,
        # Render at full height so ALL rows show and the whole PAGE scrolls, instead
        # of trapping the rows inside a fixed-height box with its own scrollbar.
        height=(len(display) + 1) * 36 + 3,
        column_config={"Thesis": st.column_config.TextColumn(width="large"),
                       "Flags": st.column_config.TextColumn(width="medium"),
                       "Name": st.column_config.TextColumn(width="medium")},
    )
    st.caption("🟢 = positive signal · 🔴 = risk. Flags adjust the Score (±, capped at −25/+15). "
               "**Insider** = open-market buys/sells by officers & directors, last 90 days "
               "(🟢 net buying · 🟣 cluster buying near the 52-week low — the classic contrarian tell; "
               "display-only, doesn't move the Score). "
               "Full list under **ℹ️ How the score works & caveats** below.")

    st.divider()
    csv = view.drop(columns=["Target OK"], errors="ignore").to_csv(index=False).encode()
    st.download_button("📥 Download current view as CSV", data=csv,
                       file_name=f"watchlist_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")

    with st.expander("ℹ️ How the score works & caveats", expanded=False):
        st.markdown(
            f"""
**Six sub-scores, each 0–100 (percentile vs peers):**

- **V — Value:** earnings yield + **EV/EBIT** (capital-structure-aware) · FCF yield · low P/S · low PEG.
- **Q — Quality:** ROIC + **gross profitability (gross profit ÷ assets, Novy-Marx)** · margins.
- **G — Growth:** revenue growth · EPS growth · Rule-of-40.
- **M — Momentum:** **real 12-minus-1-month return** · 52-week range position · price vs 200-day avg.
- **S — Safety:** Piotroski-style health · leverage (D/E + net-debt/EBITDA) · interest coverage.
- **Moat:** the *quantitative fingerprint* of a durable edge — sustained high ROIC + a high **5-year
  average gross margin** (pricing power) + **margin stability** and **revenue-growth consistency** over
  ~5 years. A true moat is qualitative; this captures what it leaves in the numbers. Rank by it with the
  **Wide-Moat Compounders** lens. (Uses 5 years of statements — no extra API calls.)

**Logic (how a sub-score is built):**

1. Each metric is percentile-ranked **within its sector** (a bank ranks against banks, not software).
2. Correlated metrics are grouped into a **concept** and averaged, then concepts are averaged equally — so
   adding a third margin metric can't drown ROIC (no double-counting).
3. The family score is **shrunk toward neutral by missing coverage**, so a stock with only 1 of 3 concepts
   can't fluke a 100. **Cov** shows how many of the lens's families were available.

**Headline Score** blends families per the **lens** (equal-ish weights on purpose — tuned weights overfit),
then applies a gentle **Safety gate** (fragile balance sheets keep ≥85% of their score, so a cheap-but-shaky
name can't top a momentum or growth lens) and the absolute red/green **flags** below.

**Flags — what the icons mean.** Percentile scores only say "better than peers"; flags catch what's good or
bad in *absolute* terms and nudge the Score (total capped at −25/+15). **🟢 = positive, 🔴 = risk.**

*🟢 positives (add points):*

- **🟢 cheap vs growth (PEG < 1)** — price is low relative to its earnings-growth rate. *(+6)*
- **🟢 cheap on sales, still growing (P/S < 2)** — inexpensive on revenue while revenue is still rising. *(+4)*
- **🟢 high returns on capital (ROIC > 20%)** — earns unusually high returns — a quality/moat sign. *(+5)*
- **🟢 strong balance sheet (net cash)** — more cash than debt. *(+4)*

*🔴 risks (subtract points):*

- **🔴 revenue declining / falling >10%** — the top line is shrinking. *(−6 / −12)*
- **🔴 elevated / high debt (D/E > 1 / > 2)** — a lot of debt relative to equity. *(−4 / −8)*
- **🔴 leveraged / heavy leverage (net debt > 3× / 4× EBITDA)** — debt is large vs. earnings. *(−4 / −8)*
- **🔴 thin interest coverage (< 2×)** — profits barely cover interest payments. *(−6)*
- **🔴 burning cash (negative FCF)** — spending more cash than it generates. *(−5)*
- **🔴 unprofitable (negative margin)** — losing money at the bottom line. *(−4)*
- **🔴 value-trap risk (high debt + falling revenue)** — the dangerous combo of leverage and shrinking sales. *(−5)*

**Why QARP is the default lens:** cheapness alone finds value traps (cheap because dying); quality alone
overpays. The most evidence-backed simple recipe — Greenblatt's Magic Formula and the academic
quality-minus-junk literature — is to demand **both**: high earnings/FCF yield AND high ROIC, with a clean
balance sheet. QARP = Value 1.0 × Quality 1.0 × Safety 0.75 × Growth 0.25 × Moat 0.5, flag-adjusted.
*Moat is now folded into every lens* (0.5 in quality/value/safety lenses, 0.25 in growth/momentum, 1.0 in
the Wide-Moat lens), so durable franchises get credit no matter how you rank.

**Caveats.**
- Not financial advice. This ranks **past reported data** — factor premia are real but noisy and can
  underperform for years. A high score is a research starting point, not a buy signal.
- **Banks / REITs:** sector-neutral ranking compares them fairly to peers, but ROIC/FCF-yield are imperfect for
  them — read those rows with extra care.
- **Thin sectors** (fewer than {5} rated names) fall back to whole-universe ranking.
- **Free-tier budget:** prices = 1 batch call; fundamentals = {CALLS_PER_TICKER} calls/ticker, cached
  {FUND_TTL//86400}d, ≤{FUND_BUDGET} tickers refreshed per load, stopping on "Limit Reach".
  Momentum adds **no FMP calls** (one free Yahoo price download/day, cached). Anything FMP can't
  deliver is filled from **Yahoo Finance** (keyless, cached {YF_FUND_TTL//86400}d so FMP replaces it
  when quota allows).
"""
        )

elif nav == NAV_RADAR:
    # ---------------------------------------------------------------------------
    # 🚀 10x Radar — exploding revenues (watchlist + S&P 500 + Nasdaq-100 scan)
    # ---------------------------------------------------------------------------
    st.subheader("🚀 10x Radar — exploding revenues")
    st.caption(
        f"Hunts the reported-numbers profile that past 10-baggers (Nvidia '23, Supermicro, memory-cycle turns "
        f"like Micron/SanDisk) showed **early**: quarterly revenue growth that is **big and getting bigger**, "
        f"margins inflecting up (operating leverage), a market cap with room to 10x, and price momentum "
        f"confirming. Scans **{len(TENX_SYMBOLS)}** names — your {len(SYMBOLS)}-ticker watchlist **plus the "
        f"S&P 500 / Nasdaq-100** — using free quarterly data (zero FMP quota). "
        f"**Not a prediction:** most hypergrowth names never 10x; treat hits as research candidates, sized small."
    )
    if not HAVE_YF:
        st.warning("Install `yfinance` to enable the 10x Radar — it powers the free quarterly-revenue scan.")
    else:
        if "tenx_scan_more" not in st.session_state:
            st.session_state.tenx_scan_more = False
        tenx_budget = TENX_CLICK_BUDGET if st.session_state.tenx_scan_more else TENX_AUTO_BUDGET
        st.session_state.tenx_scan_more = False

        with st.spinner("Scanning quarterly revenues (cached 3 weeks; new tickers fill in batches)…"):
            scan_quotes = fetch_scan_quotes(api_key, SCAN_ONLY_SYMBOLS)
            tenx_metrics, t_ref, t_have = fetch_tenx_all(TENX_SYMBOLS, tenx_budget)

        rc1, rc2, rc3 = st.columns([1.5, 2.6, 1.4])
        with rc1:
            if st.button(f"🔍 Scan next {TENX_CLICK_BUDGET} tickers", width="stretch",
                         help="Fetch quarterly revenue for the next unscanned batch (Yahoo — free, keyless)."):
                st.session_state.tenx_scan_more = True
                st.rerun()
        with rc2:
            min_yoy = st.slider("Min quarterly revenue growth (YoY %)", 0, 100, 25, 5,
                                help="Only show names whose latest-quarter revenue grew at least this "
                                     "fast vs the same quarter last year.")
        with rc3:
            show_all_tenx = st.checkbox("Show all matches", value=False, help="Off = top 20 by 10x score.")

        st.caption(f"Quarterly data: **{t_have}/{len(TENX_SYMBOLS)}** tickers scanned · {t_ref} refreshed this "
                   f"load (budget {tenx_budget}/load — coverage builds over a few loads, then stays cached "
                   f"{TENX_TTL//86400} days).")

        tenx_rows = []
        tenx_scores_all = {}
        for s_ in TENX_SYMBOLS:
            tm = tenx_metrics.get(s_)
            if not tm:
                continue
            q_ = quotes.get(s_) or scan_quotes.get(s_) or {}
            mcap_ = q_.get("marketCap")
            p_, lo_, hi_ = q_.get("price"), q_.get("yearLow"), q_.get("yearHigh")
            m52_ = ((p_ - lo_) / (hi_ - lo_)) if (p_ and lo_ is not None and hi_ and hi_ > lo_) else None
            score_, _sub, tags_ = tenx_score(tm, mcap_, mom_12_1.get(s_), m52_)
            if score_ is None:
                continue
            tenx_scores_all[s_] = round(score_, 1)
            accel_ = tm.get("rev_accel") if tm.get("rev_accel") is not None else tm.get("seq_accel")
            ttm_ = tm.get("ttm_rev")
            tenx_rows.append({
                "Sym": s_,
                "Ticker": ("★ " if s_ in SECTORS else "") + s_,
                "Name": TENX_NAMES.get(s_, s_), "Sector": TENX_SECTORS.get(s_, "—"),
                "10x": score_,
                "Rev YoY (Q) %": tm["q_rev_yoy"] * 100,
                "Accel ppt": accel_ * 100 if accel_ is not None else None,
                "QoQ %": tm["q_rev_qoq"] * 100 if tm.get("q_rev_qoq") is not None else None,
                "GM Δ ppt": tm["gm_delta"] * 100 if tm.get("gm_delta") is not None else None,
                "OM Δ ppt": tm["om_delta"] * 100 if tm.get("om_delta") is not None else None,
                "12-1m %": mom_12_1.get(s_) * 100 if mom_12_1.get(s_) is not None else None,
                "P/S": (mcap_ / ttm_) if (mcap_ and ttm_ and ttm_ > 0) else None,
                "Mkt Cap": mcap_,
                "Latest Q": tm.get("latest_q") or "—",
                "Signals": " · ".join(tags_) if tags_ else "—",
            })

        # Daily history: Δ vs the last snapshot ≥5 days old + 🆕 for names that
        # ENTERED the top 10 since then (entering the radar is the signal).
        if tenx_scores_all:
            record_history("tenx", tenx_scores_all)
        hist_date, old_scores = history_snapshot("tenx", 5)
        top10_now = set(sorted(tenx_scores_all, key=tenx_scores_all.get, reverse=True)[:10])
        old_top10 = (set(sorted(old_scores, key=old_scores.get, reverse=True)[:10])
                     if old_scores else set())
        for row in tenx_rows:
            sym = row["Sym"]
            row["10x Δ"] = (row["10x"] - old_scores[sym]) if (old_scores and sym in old_scores) else None
            if old_scores and sym in top10_now and sym not in old_top10:
                row["Signals"] = "🆕 entered top 10 · " + (row["Signals"] if row["Signals"] != "—" else "")

        tdf = pd.DataFrame(tenx_rows)
        if tdf.empty:
            st.info("No quarterly data scanned yet — it fills automatically each load, or click "
                    "**Scan next batch** to speed it up.")
        else:
            tdf = tdf[tdf["Rev YoY (Q) %"] >= float(min_yoy)].sort_values("10x", ascending=False)
            shown_tenx = tdf if show_all_tenx else tdf.head(20)
            if shown_tenx.empty:
                st.info(f"No scanned name clears {min_yoy}% quarterly YoY revenue growth — lower the "
                        f"slider or scan more tickers.")
            else:
                with st.spinner("Checking earnings dates…"):
                    earn_dates = fetch_earnings_dates(list(shown_tenx["Sym"])[:40])

                def _tenx_disp(row):
                    return pd.Series({
                        "Ticker": row["Ticker"], "Name": row["Name"], "Sector": row["Sector"],
                        "10x": f"{row['10x']:.0f}",
                        "Δ": (f"{row['10x Δ']:+.0f}" if pd.notna(row.get("10x Δ")) else "—"),
                        "Rev YoY (Q)": f"{row['Rev YoY (Q) %']:+.0f}%",
                        "Accel": f"{row['Accel ppt']:+.0f}pt" if pd.notna(row["Accel ppt"]) else "—",
                        "QoQ": f"{row['QoQ %']:+.1f}%" if pd.notna(row["QoQ %"]) else "—",
                        "GM Δ": f"{row['GM Δ ppt']:+.1f}pt" if pd.notna(row["GM Δ ppt"]) else "—",
                        "OM Δ": f"{row['OM Δ ppt']:+.1f}pt" if pd.notna(row["OM Δ ppt"]) else "—",
                        "12-1m": f"{row['12-1m %']:+.0f}%" if pd.notna(row["12-1m %"]) else "—",
                        "P/S": f"{row['P/S']:.1f}" if pd.notna(row["P/S"]) else "—",
                        "Mkt Cap": fmt_mcap(row["Mkt Cap"]),
                        "Earnings": fmt_earnings(earn_dates.get(row["Sym"])),
                        "Latest Q": row["Latest Q"],
                        "Signals": row["Signals"],
                    })
                tenx_display = shown_tenx.apply(_tenx_disp, axis=1)
                tstyler = (tenx_display.style
                           .map(color_score, subset=["10x"])
                           .map(color_signed, subset=["Δ", "Rev YoY (Q)", "Accel", "QoQ",
                                                      "GM Δ", "OM Δ", "12-1m"]))
                st.dataframe(
                    tstyler, width="stretch", hide_index=True,
                    height=(len(tenx_display) + 1) * 36 + 3,
                    column_config={"Signals": st.column_config.TextColumn(width="large"),
                                   "Name": st.column_config.TextColumn(width="medium")},
                )
                st.caption("★ = also in your watchlist · sorted by 10x score · "
                           "**Δ** = 10x-score change" + (f" since {hist_date}" if hist_date else " (needs ~5 days of history)") + " · "
                           "**Rev YoY (Q)** = latest quarter vs same quarter last year · "
                           "**Accel** = change in that YoY rate vs the prior quarter (percentage points) · "
                           "**GM/OM Δ** = gross/operating-margin change vs year-ago quarter · "
                           "**P/S** = market cap ÷ TTM revenue — how much of the explosion is already priced in · "
                           "**Earnings** = next report (⚠️ within 7 days).")
                tenx_csv = tdf.to_csv(index=False).encode()
                st.download_button("📥 Download 10x Radar as CSV", data=tenx_csv,
                                   file_name=f"tenx_radar_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                   mime="text/csv")

        with st.expander("ℹ️ How the 10x score works — and why most hits still won't 10x"):
            st.markdown(
                f"""
**The idea.** Stocks that went on to 10x — Nvidia into the AI buildout, Supermicro, Micron/SanDisk in
memory upcycles — tended to share a *reported-numbers* fingerprint **before** the biggest gains: quarterly
revenue growth that was already large **and still accelerating**, margins expanding at the same time
(operating leverage), a market cap small enough that a 10x was arithmetically possible, and price momentum
turning up as the market caught on. This radar scores how closely each name's **latest reported quarter**
matches that fingerprint. Annual data is too slow for this job — a memory-cycle turn shows up in quarterly
prints two or three quarters before the annual growth number moves — so the radar runs on **quarterly**
income statements (Yahoo Finance, free, cached {TENX_TTL//86400} days).

**The five components (weights):**

- **Revenue explosion (35%)** — latest-quarter revenue vs the same quarter last year. 0% scores 0; +100%
  or more scores full marks. This is the "revenues exploding" core.
- **Acceleration (25%)** — is the YoY growth rate itself rising vs the prior quarter's YoY rate? Catching
  the *second derivative* is what finds cycle turns (SanDisk/Micron) early. Falls back to sequential
  quarter-over-quarter acceleration when only five quarters of history exist.
- **Operating leverage (15%)** — gross & operating margin change vs the year-ago quarter. Exploding revenue
  with *expanding* margins is the profit-inflection double-whammy that re-rates stocks.
- **10x headroom (15%)** — market-cap tiers: under $2B scores 1.0, $2–10B ≈ 0.9, $10–50B ≈ 0.7, $50–200B ≈
  0.5, $200B–1T ≈ 0.3, above $1T ≈ 0.1. A $4T company 10x-ing would be a $40T company — the math matters.
- **Momentum confirmation (10%)** — 12-minus-1-month return (or 52-week-range position for scan-only
  names). A hot quarter the market ignores deserves a look; one it's chasing deserves confirmation.

Missing components shrink the score toward 50 rather than being skipped, so thin data can't fluke a 95.
Thresholds are **absolute**, not peer-relative — "exploding" should mean exploding.

**Read the ⚠️ tags.** *Tiny revenue base* means the percentages are easy (a $30M company doubling is not
Nvidia doubling); *margins compressing* means growth is being bought with profitability.

**Caveats — please read.** Not financial advice. This screen looks **backwards** at reported data; it cannot
see guidance cuts, competition, dilution, or cycle peaks (the same memory names that 10x also fall 70% in
downcycles). Most names that light up here will *not* 10x — the historical base rate for 10-baggers is low
even among hypergrowers. Use it to generate research candidates, size positions small, and do the
qualitative work the numbers can't.
"""
            )
    st.divider()

elif nav == NAV_MARKET:
    with st.container():
        st.subheader("📊 Market conditions")
        st.caption("Context for **how** to deploy (position size · scaling · rebalancing) — **not** a buy/sell signal.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("S&P 500", f"{mkt['index_price']:,.0f}" if mkt["index_price"] else "—",
                  help="Current level of the S&P 500 index.")
        c2.metric("vs 200-day avg", f"{mkt['pct_from_ma200']:+.1f}%" if mkt["pct_from_ma200"] is not None else "—",
                  help="How far the index sits above (+) or below (−) its 200-day moving average. "
                       "Above = uptrend regime; the further above, the more 'stretched' it is.")
        c3.metric("Index 52w position", f"{mkt['index_52w_pos']:.0f}%" if mkt["index_52w_pos"] is not None else "—",
                  help="Where the index sits in its 52-week range: 0% = 52-week LOW, 100% = 52-week HIGH. "
                       "So 94% means it's near the top of the past year's range.")
        c4.metric("VIX (volatility)", f"{mkt['vix']:.1f}" if mkt["vix"] else "—",
                  help="The 'fear gauge' — expected 30-day volatility. Under ~15 = calm, 20–28 = elevated, 40+ = extreme.")

        # Plain-language regime line (clearer than st.metric's up/down arrow, and kept
        # neutral — this describes the environment, it is not a buy/sell signal).
        trend = mkt.get("trend", "unknown")
        if trend != "unknown":
            above = "above" in trend
            st.caption(
                f"**Trend regime:** {'📈' if above else '📉'} the S&P 500 is **{'above' if above else 'below'}** "
                f"its 200-day average ({'uptrend' if above else 'downtrend'}). Describes the environment — not a buy/sell call."
            )
        if mkt.get("vix_context"):
            st.caption(f"**Volatility — {mkt['vix_label']}:** {mkt['vix_context']}")
        for note in mkt.get("notes", []):
            st.info(note)

        # -----------------------------------------------------------------------
        # 🎯 S&P 500 entry meter — buy fear, not greed
        # -----------------------------------------------------------------------
        st.markdown("#### 🎯 Time to add to the S&P 500? — fear/greed entry meter")

        # Breadth: % of tracked tickers above their own 200-day average (free —
        # derived from the batch quotes already in memory).
        _above = _with_ma = 0
        for _s in SYMBOLS:
            _q = quotes.get(_s) or {}
            _p, _ma = _q.get("price"), _q.get("ma200")
            if _p and _ma:
                _with_ma += 1
                if _p > _ma:
                    _above += 1
        breadth_pct = (_above / _with_ma * 100) if _with_ma >= 20 else None

        _hist = cached_entry_history() if HAVE_YF else {}
        meter = compute_entry_meter(mkt, breadth_pct,
                                    _hist.get("spx_rsi14"), _hist.get("vix_pctile_1y"))

        if meter["score"] is None:
            st.caption("Entry meter unavailable — market data didn't load this refresh.")
        else:
            e1, e2 = st.columns([1, 3])
            e1.metric("Fear ↔ Greed", f"{meter['score']:.0f}/100",
                      help="0 = extreme fear (historically the better entry zone) · "
                           "100 = extreme greed (historically the worse entry zone). "
                           "Blend of drawdown, VIX level + 1y percentile, trend stretch, "
                           "52-week position, RSI(14), and breadth.")
            with e2:
                st.progress(min(1.0, max(0.0, meter["score"] / 100.0)),
                            text=f"**{meter['zone']}** — 0 = extreme fear (buy zone by your rule) · "
                                 f"100 = extreme greed")
            st.markdown(f"**Stance:** {meter['stance']}")
            record_history("meter", round(meter["score"], 1))
            m_date, m_old = history_snapshot("meter", 5)
            if m_old is not None:
                drift = meter["score"] - float(m_old)
                st.caption(f"**Trend:** {m_old:.0f} on {m_date} → {meter['score']:.0f} now "
                           f"({drift:+.0f}; {'moving toward fear' if drift < 0 else 'moving toward greed' if drift > 0 else 'unchanged'}).")
            comp_line = "  ·  ".join(f"{name} **{g:.0f}** ({detail})"
                                     for name, g, detail in meter["components"])
            st.caption(f"Components (each 0 = fear → 100 = greed): {comp_line}")
            with st.expander("ℹ️ How the entry meter works — and the honest caveats"):
                st.markdown(
                    """
**Your rule, quantified.** "Invest during fear, not greed" is the contrarian discipline this meter
encodes. Each signal is scored 0 (maximum fear) to 100 (maximum greed) and averaged:

- **Drawdown from 52-week high** — at the high scores 100; a −20% bear-market drawdown scores 0.
  Deeper discounts to the high have historically offered better forward entry prices.
- **VIX level** — panic (40+) scores 0; complacency (≤12) scores 100. Volatility spikes cluster
  near lows; calm clusters near tops.
- **VIX vs its own 1-year range** — the same signal, but relative: today's VIX percentile over the
  past year, inverted.
- **Stretch vs the 200-day average** — more than ~10% above the long-term trend line scores fully
  greedy; more than ~10% below scores fully fearful.
- **52-week range position** — where the index sits between its 1-year low (0) and high (100).
- **RSI(14)** — the classic oversold (≤30 → 0) / overbought (≥70 → 100) oscillator on the index.
- **Breadth** — the share of this app's tracked tickers above their own 200-day average; narrow
  participation near highs is a greed tell, washed-out breadth a fear tell.

**The honest caveats.** Market timing does not reliably work, and the meter knows nothing about
the future: extreme fear got MORE extreme in 2008 and March 2020 before it paid off, and greed
readings can persist for years in strong bull markets (1995–1999, 2023–2024) while the index
compounds. Historically a lump sum invested immediately has beaten waiting for a pullback roughly
two-thirds of the time. So use the meter the way the stance text does: **never stop a scheduled
DCA because of greed; use fear to deploy pre-committed extra cash faster.** It calibrates the
price you pay when you were investing anyway — it is not a prediction, and not financial advice.
"""
                )
    st.divider()

elif nav == NAV_LOOKUP:
    # Lens picked on the Watchlist tab (session default if not visited yet).
    rank_mode = st.session_state.get("lens", DEFAULT_LENS)
    weights = LENSES[rank_mode]

    # ---------------------------------------------------------------------------
    # Single-ticker lookup — type ANY symbol (watchlist or not) to pull its metrics
    # ---------------------------------------------------------------------------
    st.subheader("🔎 Look up a ticker")
    query = st.text_input(
        "ticker lookup", "", label_visibility="collapsed",
        placeholder="Type any symbol — AAPL, MSFT, RELIANCE.NS — and press Enter",
    ).strip().upper()

    if query:
        is_member = query in SYMBOLS
        look_sector = SECTORS.get(query, "Lookup")
        look_name = next((it["name"] for it in WATCHLIST if it["ticker"] == query), query)
        look_region = next((it["region"] for it in WATCHLIST if it["ticker"] == query), "US")
        have = (is_member and has_any(fundamentals.get(query) or {})
                and (quotes.get(query) or {}).get("price") is not None)

        look_fund, look_quote = (fundamentals.get(query) or {}), (quotes.get(query) or {})
        if not have:
            try:
                with st.spinner(f"Fetching {query}…"):
                    lq, *_ = fetch_quotes(api_key, [query], False)
                    look_quote = lq.get(query) or {}
                    lf, *_ = fetch_fundamentals_all(api_key, [query],
                                                    {query: look_quote.get("marketCap")}, False)
                    look_fund = lf.get(query) or {}
                    price, lo, hi, ma200, pe = (look_quote.get("price"), look_quote.get("yearLow"),
                                                look_quote.get("yearHigh"), look_quote.get("ma200"),
                                                look_quote.get("pe"))
                    if look_fund.get("earnings_yield") is None and pe and pe > 0:
                        look_fund["earnings_yield"] = 1.0 / pe
                    if price and lo is not None and hi and hi > lo:
                        look_fund["mom_52w"] = (price - lo) / (hi - lo)
                    if price and ma200:
                        look_fund["mom_ma200"] = price / ma200 - 1
                    mom = fetch_momentum([query], False).get(query)
                    if mom is not None:
                        look_fund["mom_12_1"] = mom
            except Exception as _e:  # noqa: BLE001
                look_quote = {}
                logger.warning("lookup %s failed: %s", query, str(_e)[:120])

        if not look_quote or look_quote.get("price") is None:
            st.warning(f"Couldn't find data for **{query}**. Check the symbol — non-US tickers need an "
                       f"exchange suffix (e.g. `RELIANCE.NS`, `7203.T`, `BARC.L`).")
        else:
            # Score it against the full universe so V/Q/G/M/S are real percentiles.
            combined = {**fundamentals, query: look_fund}
            combined_sec = {**SECTORS, query: look_sector}
            look_sub = {f: ((compute_family_scores(combined, combined_sec).get(query, {}).get(f) or {}).get("score"))
                        for f in FAMILIES} if not is_member else \
                       {f: ((fam_scores.get(query, {}).get(f) or {}).get("score")) for f in FAMILIES}
            num = den = 0.0
            for f, w in weights.items():
                s = look_sub.get(f)
                if s is not None:
                    num += w * s; den += w
            base = (num / den * 100) if den > 0 else None
            flag_adj, flag_list = compute_flags(look_fund)
            gate = safety_gate(look_fund.get("safety"))
            comp = min(100.0, max(0.0, base * gate + flag_adj)) if base is not None else None
            ccy = REGION_CURRENCY.get(look_region, "USD")

            tag = f"_{look_sector}_" if is_member else "_(not in watchlist — ranked vs whole universe)_"
            st.markdown(f"### {look_name} · {query}  ·  {tag}")

            cols = st.columns(8)
            def _sub(f):
                v = look_sub.get(f); return f"{v*100:.0f}" if v is not None else "—"
            cols[0].metric(f"Score · {rank_mode.split(' ')[0]}", f"{comp:.0f}" if comp is not None else "—")
            cols[1].metric("Value", _sub("Value"))
            cols[2].metric("Quality", _sub("Quality"))
            cols[3].metric("Growth", _sub("Growth"))
            cols[4].metric("Momentum", _sub("Momentum"))
            cols[5].metric("Safety", _sub("Safety"))
            cols[6].metric("Moat", _sub("Moat"))
            cols[7].metric("Price", fmt_price(look_quote.get("price"), ccy),
                           f"{look_quote.get('changePercentage'):+.2f}%"
                           if look_quote.get("changePercentage") is not None else None)

            def _p(k):
                v = look_fund.get(k); return f"{v*100:.1f}%" if v is not None else "—"
            def _r(k, d=2):
                v = look_fund.get(k); return f"{v:.{d}f}" if v is not None else "—"
            raw = {
                "Earn Yld": _p("earnings_yield"), "ROIC": _p("roic"), "FCF Yld": _p("fcf_yield"),
                "Rev Grw": _p("rev_growth"), "EPS Grw": _p("eps_growth"),
                "Gross Mgn": _p("gross_margin"), "Gross Prof": _p("gross_profitability"),
                "Op Mgn": _p("operating_margin"),
                "Rule40": _r("rule_of_40", 0), "12-1m": _p("mom_12_1"),
                "P/S": _r("ps_ratio", 1), "PEG": _r("peg", 2),
                "EV/EBIT": ("n/m" if (look_fund.get("ev_ebit") or 0) >= 999 else _r("ev_ebit", 1)),
                "D/E": _r("debt_equity", 2),
                "Net Debt/EBITDA": _r("net_debt_ebitda", 1), "Int Cov": _r("interest_coverage", 1),
                "GM 5y": _p("gross_margin_avg"), "Margin Stab": _p("margin_stability"),
                "Growth Consist": _p("growth_consistency"),
                "Mkt Cap": fmt_mcap(look_quote.get("marketCap"), ccy),
            }
            st.dataframe(pd.DataFrame([raw]), width="stretch", hide_index=True)
            if flag_list:
                st.caption("**Flags:** " + " · ".join(flag_list))
            # 10x Radar read for the looked-up ticker (1 free Yahoo call, cached).
            if HAVE_YF:
                tq_metrics, _, _ = fetch_tenx_all([query], budget=1)
                tqm = tq_metrics.get(query)
                if tqm and tqm.get("q_rev_yoy") is not None:
                    t_sc, _, t_tags = tenx_score(tqm, look_quote.get("marketCap"),
                                                 look_fund.get("mom_12_1"), look_fund.get("mom_52w"))
                    if t_sc is not None:
                        st.caption(f"**🚀 10x Radar:** {t_sc:.0f}/100 · Rev YoY (Q) "
                                   f"{tqm['q_rev_yoy']*100:+.0f}%"
                                   + (" · " + " · ".join(t_tags) if t_tags else ""))
            if is_member:
                thesis = next((it["thesis"] for it in WATCHLIST if it["ticker"] == query), "")
                if thesis:
                    st.caption(f"**Thesis:** {thesis}")
    st.divider()
