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

# ---------------------------------------------------------------------------
# Streamlit Cloud stale-module guard.
# On a git push, Cloud hot-reloads THIS file but keeps previously imported
# local modules in the running Python process — so a new app.py importing a
# just-added function raises ImportError until someone reboots the app
# (this bit us twice: tenx_radar.fetch_next_earnings, options_income.live_spot).
# Reloading our own modules at script start guarantees app.py and its helpers
# always come from the same commit. Parents reload before dependents.
# ---------------------------------------------------------------------------
import importlib as _importlib
import sys as _sys
for _m in ("fundamentals", "yahoo_fallback", "market_risk",
           "watchlist_data", "watchlist_growth", "tenx_universe",
           "tenx_radar", "entry_meter", "insider", "options_income"):
    if _m in _sys.modules:
        try:
            _importlib.reload(_sys.modules[_m])
        except Exception:  # noqa: BLE001 — never let the guard itself kill the app
            pass

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
    compute_flags, safety_gate, value_gate, target_is_sane, has_any,
    FMPRateLimitError, FAMILIES, LENSES, DEFAULT_LENS, CALLS_PER_TICKER,
)
from yahoo_fallback import (
    fetch_quotes_yahoo, fetch_fundamentals_yahoo, fetch_momentum_yahoo, HAVE_YF,
)
from tenx_universe import SCAN_SYMBOLS, SCAN_NAMES, SCAN_SECTORS
from tenx_radar import (fetch_quarterly_yahoo, compute_tenx_metrics, tenx_score,
                        fetch_next_earnings, fetch_last_earnings_surprise)
from entry_meter import fetch_entry_history_yahoo, compute_entry_meter
from insider import fetch_insider, insider_display
from options_income import (fetch_put_candidates, fetch_call_candidates,
                            fetch_leaps_candidates, fetch_wheel_candidates,
                            fetch_put_spread_candidates, live_spot)
import market_risk

_seen = {item["ticker"] for item in _BASE_WATCHLIST}
WATCHLIST = _BASE_WATCHLIST + [g for g in GROWTH_WATCHLIST if g["ticker"] not in _seen]
SECTOR_ORDER = _BASE_SECTORS + [s for s in ["Hypergrowth"] if s not in _BASE_SECTORS]
SYMBOLS = [item["ticker"] for item in WATCHLIST]
SECTORS = {item["ticker"]: item["sector"] for item in WATCHLIST}

# Family -> short column label shown in the table.
FAM_ABBR = {"Value": "V", "Quality": "Q", "Growth": "G", "Momentum": "M", "Safety": "S", "Moat": "Moat"}

FMP_BASE = "https://financialmodelingprep.com/stable"
QUOTE_TTL = int(os.getenv("QUOTE_TTL_MIN", "60")) * 60   # fresh-ish prices in market hours
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
# Reserve ~45 daily FMP calls for quotes (hourly batches), market context,
# scan-universe batches and the insider budget, so fundamentals can never
# starve the price layer into staleness.
_DEFAULT_BUDGET = min(len(WATCHLIST), max(10, (250 - 45) // CALLS_PER_TICKER))
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
OPTIONS_TTL = 4 * 60 * 60            # put-screener chains (research freshness, not execution)
PUTS_MAX_TICKERS = 12                # options chains fetched per load (~2-3 calls each)
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
@st.cache_resource
def options_store(): return PersistentStore(os.path.join(_cache_dir(), "options.json"))


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
    """True only for GENUINE quota exhaustion (daily limit / 429 / bandwidth).
    FMP's plan-restriction message ('Exclusive Endpoint ... upgrade your plan')
    must NOT match: it used to, and one premium-only endpoint reply made the
    app abort all fetching and show 'quota reached' at 32/250 real calls."""
    t = (text or "").lower()
    return "limit reach" in t or "too many requests" in t or "bandwidth" in t


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
                if not rows and len(need) <= 20:
                    # Batch unsupported on this plan -> per-ticker, but ONLY for
                    # small needs (single-ticker lookups). A big stale set would
                    # burn 100+ calls here; Yahoo's bulk refresh below handles it.
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

        # Yahoo fallback for EVERYTHING FMP couldn't refresh this round — both
        # symbols with no price at all AND symbols with a stale cached price.
        # (The old "missing only" check let a stale price block its own refresh:
        # CRWV sat on a Friday close for days once the FMP quota was spent.)
        still_stale = [s for s in need if s not in got]
        if still_stale and HAVE_YF:
            lite = fetch_quotes_yahoo(still_stale, with_mcap=False)
            for sym, d in lite.items():
                old = quotes.get(sym) or {}
                if d.get("marketCap") is None:
                    d["marketCap"] = old.get("marketCap")   # keep last known mcap
                if d.get("pe") is None:
                    d["pe"] = old.get("pe")
                store.set(sym, {"data": d, "ts": now, "src": "yahoo"})
                quotes[sym] = d; fetched += 1
            # brand-new symbols Yahoo priced but that never had an FMP mcap:
            no_mcap = [s for s in lite if (quotes.get(s) or {}).get("marketCap") is None]
            if no_mcap and len(no_mcap) <= 25:
                for sym, d in fetch_quotes_yahoo(no_mcap, with_mcap=True).items():
                    store.set(sym, {"data": d, "ts": now, "src": "yahoo"})
                    quotes[sym] = d
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
    # Yahoo-refresh every scan symbol FMP couldn't deliver (stale OR missing) —
    # one bulk download, no per-ticker mcap calls; old market caps carry over.
    still_stale = [s for s in need if s not in got]
    if still_stale and HAVE_YF:
        for sym, d in fetch_quotes_yahoo(still_stale, with_mcap=False).items():
            old = quotes.get(sym) or {}
            if d.get("marketCap") is None:
                d["marketCap"] = old.get("marketCap")
            if d.get("pe") is None:
                d["pe"] = old.get("pe")
            got[sym] = d
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


_OPT_FETCHERS = {"csp": fetch_put_candidates, "cc": fetch_call_candidates,
                 "leaps": fetch_leaps_candidates, "wheel": fetch_wheel_candidates,
                 "pcs": fetch_put_spread_candidates}

# Index underlyings for defined-risk spreads: §1256 contracts — mark-to-market,
# NO wash-sale rules, 60/40 tax treatment, cash-settled (no assignment).
INDEX_UNDERLYINGS = {"^XSP": "XSP — Mini-SPX (1/10th S&P 500)",
                     "^SPX": "SPX — S&P 500 index"}


def fetch_surprises(symbols):
    """Last-quarter EPS surprise % per ticker (negative = missed estimates),
    Yahoo, cached 3 days in the earnings store under 'surprise:' keys."""
    if not HAVE_YF:
        return {}
    store = earnings_store()
    now = time.time()
    out, need = {}, []
    for s in symbols:
        e = store.get(f"surprise:{s}")
        if _fresh(e, EARN_TTL):
            out[s] = e["data"]
        else:
            need.append(s)
    for s in need:
        out[s] = fetch_last_earnings_surprise(s)
        store.set(f"surprise:{s}", {"data": out[s], "ts": now})
    if need:
        store.flush()
    return out


def fetch_live_spots(symbols):
    """Near-live spots (yfinance fast_info, keyless) for a SMALL list — the
    options screens must never price contracts off a stale cached quote.
    Returns only the symbols Yahoo answered for; callers fall back to cache."""
    if not (HAVE_YF and symbols):
        return {}
    out = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(live_spot, s): s for s in symbols}
        for fut in as_completed(futs):
            try:
                v = fut.result()
            except Exception:  # noqa: BLE001
                v = None
            if v:
                out[futs[fut]] = v
    return out


def fetch_options_all(spots_by_symbol, strategy, budget=PUTS_MAX_TICKERS):
    """Option candidates per ticker for one strategy ('csp'/'cc'/'leaps'/...).
    Yahoo chains (keyless), disk-cached 4h per (strategy, ticker) — but a cache
    entry is also invalidated when the underlying has MOVED >2% since it was
    built, so strikes/cushions never reflect a stale spot."""
    fetcher = _OPT_FETCHERS[strategy]
    store = options_store()
    now = time.time()
    out, need = {}, []
    for s, spot in spots_by_symbol.items():
        e = store.get(f"{strategy}:{s}")
        cached_spot = (e or {}).get("spot")
        drift_ok = (cached_spot and spot
                    and abs(spot / cached_spot - 1.0) <= 0.02)
        if _fresh(e, OPTIONS_TTL) and drift_ok:
            out[s] = e["data"]
        else:
            need.append((s, spot))
    if need and HAVE_YF:
        def work(sym, spot):
            return sym, spot, fetcher(sym, spot)
        fetched = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = [ex.submit(work, s, sp) for s, sp in need[:max(0, budget)]]
            for fut in as_completed(futs):
                sym, sp, res = fut.result()
                store.set(f"{strategy}:{sym}", {"data": res, "ts": now, "spot": sp})
                out[sym] = res
                fetched += 1
        if fetched:
            store.flush()
    return out


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
    st.caption(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · prices cached {QUOTE_TTL//60}min · "
               f"fundamentals cached {FUND_TTL//86400}d · budget {FUND_BUDGET}/refresh")

_force = st.session_state.force_refresh
st.session_state.force_refresh = False
force_prices = _force in ("prices", "funds", True)
force_funds = _force in ("funds", True)

with st.spinner("Loading prices…"):
    quotes, q_fetched, q_cached, q_rl = fetch_quotes(api_key, SYMBOLS, force_prices)

# Surface price AGE honestly — quota exhaustion used to freeze prices silently.
_qstore = quote_store()
_ages = [time.time() - (_qstore.get(s) or {}).get("ts", 0)
         for s in SYMBOLS if _qstore.get(s)]
_stale_h = (max(_ages) / 3600) if _ages else None
if _stale_h and _stale_h > 24:
    st.warning(f"⚠️ Some cached prices are up to **{_stale_h:.0f}h old** — the FMP quota was likely "
               f"exhausted before they could refresh (Yahoo fills what it can). The Options tab "
               f"fetches near-live spots independently; treat watchlist/radar prices as indicative "
               f"and hit **🔄 Refresh prices**.")

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

def color_score(v):
    """Text color for 0-100 score cells. Cells stay NUMERIC (so header-click
    sorting works); pretty formatting happens via st.column_config."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "color: #6b7280;"
    try:
        nv = float(v)
    except (TypeError, ValueError):
        return ""
    if nv >= 70: return "color: #047857; font-weight: 600;"
    if nv >= 45: return "color: #0369a1;"
    return "color: #6b7280;"


def color_signed(v):
    """Green/red text for signed numeric cells (numeric so sorting works)."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "color: #6b7280;"
    try:
        nv = float(v)
    except (TypeError, ValueError):
        return ""
    return "color: #059669;" if nv >= 0 else "color: #dc2626;"


def ncol(fmt, help_=None):
    """Shorthand: numeric column with printf formatting (keeps sorting numeric)."""
    return st.column_config.NumberColumn(format=fmt, help=help_)


def pos_row_bg(pos):
    """Row background from the 52-week range position (0 = at the low, 100 = at
    the high). Buy-fear visual: bottom 30% of the range shades GREEN (darkest at
    the 52w low), top 30% shades RED (darkest at the 52w high), middle is plain.
    rgba keeps text readable in light and dark themes."""
    if pos is None or pd.isna(pos):
        return ""
    if pos <= 30:
        alpha = 0.15 + (30.0 - pos) / 30.0 * 0.40      # 0% -> 0.55 · 30% -> 0.15
        css = f"background-color: rgba(16, 185, 129, {alpha:.2f});"
    elif pos >= 70:
        alpha = 0.12 + (pos - 70.0) / 30.0 * 0.33      # 70% -> 0.12 · 100% -> 0.45
        css = f"background-color: rgba(239, 68, 68, {alpha:.2f});"
    else:
        return ""
    # On the darkest rows, force white text so cell values stay readable even
    # over their own semantic colors (applied last, so it wins the cascade).
    if alpha >= 0.40:
        css += " color: #ffffff; font-weight: 600;"
    return css


def row_bg_styler(pos_series):
    """Row-wise Styler.apply function bound to a {row index -> 52w pos} series."""
    def _style(row):
        return [pos_row_bg(pos_series.get(row.name))] * len(row)
    return _style


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
NAV_WATCH, NAV_RADAR, NAV_MARKET, NAV_PUTS, NAV_LOOKUP = (
    "📊 Watchlist", "🚀 10x Radar", "🎯 Market & Entry", "💰 Options", "🔎 Lookup")
nav = st.radio("section", [NAV_WATCH, NAV_RADAR, NAV_MARKET, NAV_PUTS, NAV_LOOKUP],
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
        # Value gate: lenses promising a reasonable price (Value weight >= 1)
        # get an ABSOLUTE cheapness check — sector-relative Value ranks alone
        # let a 24x-sales stock top QARP as "cheap vs tech peers".
        vgate = 1.0
        if weights.get("Value", 0) >= 1.0:
            _ey, _ps, _rg = row.get("Earnings Yld %"), row.get("P/S"), row.get("Rev Growth %")
            vgate = value_gate(_ey / 100.0 if pd.notna(_ey) else None,
                               _ps if pd.notna(_ps) else None,
                               _rg / 100.0 if pd.notna(_rg) else None)
        score = min(100.0, max(0.0, base * gate * vgate + adj)) if pd.notna(base) else base
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

    # Insider signal (small separate FMP budget, cached 14d, watchlist only).
    insider_data, insider_unsupported = fetch_insider_all(api_key, SYMBOLS, INSIDER_BUDGET)
    if insider_unsupported:
        st.caption("ℹ️ Insider data isn't included in this FMP plan — the **Insider** column stays "
                   "blank and no quota is spent retrying.")


    def render_display(row):
        """Display row: NUMERIC cells stay numeric (header-click sorting works);
        formatting is applied by st.column_config below. Text only where a cell
        is inherently text (currency symbols, flags, ratios with sentinels)."""
        target_str = fmt_price(row["Target"], row["Currency"]) if pd.notna(row["Target"]) else "—"
        if pd.notna(row["Target"]) and not row["Target OK"]:
            target_str += " ⚠️"
        return pd.Series({
            "Ticker": f"{row['Ticker']}  ({row['Region']})",
            "Name": row["Name"],
            "Sector": row["Sector"],
            "Score": row["Score"],
            "Cov": f"{int(row['Cov'])}/{int(row['CovN'])}" if pd.notna(row["Cov"]) else "—",
            "V": row["V"], "Q": row["Q"], "G": row["G"], "M": row["M"], "S": row["S"],
            "Moat": row["Moat"],
            "Price": fmt_price(row["Price"], row["Currency"]),
            "Day %": row["Day %"],
            "Earn Yld": row["Earnings Yld %"],
            "ROIC": row["ROIC %"],
            "Rev Grw": row["Rev Growth %"],
            "Gross Mgn": row["Gross Mgn %"],
            "FCF Yld": row["FCF Yld %"],
            "P/S": row["P/S"],
            "PEG": row["PEG"],
            "EV/EBIT": (None if (pd.notna(row["EV/EBIT"]) and row["EV/EBIT"] >= 999)
                        else row["EV/EBIT"]),
            "D/E": row["D/E"],
            "Flags": row["Flags"] if row["Flags"] else "—",
            "Insider": insider_display(insider_data.get(row["Ticker"]), row["52w Pos %"]),
            "12-1m": row["Mom 12-1 %"],
            "52w": row["52w Pos %"],
            "Mkt Cap": row["Mkt Cap"],
            "Target": target_str,
            "Upside %": row["Upside %"],
            "Thesis": row["Thesis"],
        })


    display = view.apply(render_display, axis=1)

    styler = display.style
    for c in ["Score", "V", "Q", "G", "M", "S", "Moat"]:
        styler = styler.map(color_score, subset=[c])
    styler = (styler.map(color_signed, subset=["Day %"])
                    .map(color_signed, subset=["Rev Grw"])
                    .map(color_signed, subset=["12-1m"]))
    # Row shading LAST so the white-text override wins on the darkest rows.
    styler = styler.apply(row_bg_styler(view["52w Pos %"]), axis=1)

    st.dataframe(
        styler, width="stretch", hide_index=True,
        # Render at full height so ALL rows show and the whole PAGE scrolls, instead
        # of trapping the rows inside a fixed-height box with its own scrollbar.
        height=(len(display) + 1) * 36 + 3,
        column_config={
            "Thesis": st.column_config.TextColumn(width="large"),
            "Flags": st.column_config.TextColumn(width="medium"),
            "Name": st.column_config.TextColumn(width="medium"),
            "Score": ncol("%.0f"), "V": ncol("%.0f"), "Q": ncol("%.0f"), "G": ncol("%.0f"),
            "M": ncol("%.0f"), "S": ncol("%.0f"), "Moat": ncol("%.0f"),
            "Day %": ncol("%+.2f%%"),
            "Earn Yld": ncol("%.1f%%"), "ROIC": ncol("%.1f%%"), "Rev Grw": ncol("%+.1f%%"),
            "Gross Mgn": ncol("%.0f%%"), "FCF Yld": ncol("%.1f%%"),
            "P/S": ncol("%.1f"), "PEG": ncol("%.2f"),
            "EV/EBIT": ncol("%.1f", "Blank when EBIT ≤ 0 (not meaningful)."),
            "D/E": ncol("%.2f"),
            "12-1m": ncol("%+.0f%%"), "52w": ncol("%.0f%%"),
            "Mkt Cap": st.column_config.NumberColumn(
                format="compact", help="Native currency for non-US listings."),
            "Upside %": ncol("%+.1f%%"),
        },
    )
    st.caption("**Row shading:** 🟩 green = in the bottom 30% of its 52-week range (darker = closer "
               "to the 52w low — the buy-fear zone) · 🟥 red = top 30% of the range (darker = closer "
               "to the 52w high). "
               "🟢 = positive signal · 🔴 = risk. Flags adjust the Score (±, capped at −25/+15). "
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
- **🔴 rich / very rich on sales for its growth (P/S > 15 / > 25)** — the absolute-price guard: sector-relative
  Value ranks can make an expensive stock look "cheap vs peers", so multiples far ahead of the growth paying
  for them subtract points — this is what keeps QARP's *reasonable price* honest. *(−4 / −8)*

**Why QARP is the default lens:** cheapness alone finds value traps (cheap because dying); quality alone
overpays. The most evidence-backed simple recipe — Greenblatt's Magic Formula and the academic
quality-minus-junk literature — is to demand **both**: high earnings/FCF yield AND high ROIC, with a clean
balance sheet. QARP = Value 1.0 × Quality 1.0 × Safety 0.75 × Growth 0.25 × Moat 0.5, flag-adjusted.

**The absolute value gate (QARP / Blended / Value-Quality only).** Value sub-scores rank *within sector*,
so in an expensive sector a 24×-sales stock can rank "cheap vs peers". Lenses that promise a reasonable
price therefore apply a second, **absolute** check — earnings yield (5%+ untouched, P/E≈67 maxes the
penalty) and the sales multiple vs the growth paying for it (P/S ÷ growth% ≤ 0.3 untouched, ≥ 1.0 maxes
it). The most expensively priced names keep at most **80%** of their score, on top of the 🔴 rich-on-sales
flags. Growth/Momentum lenses are deliberately NOT gated — hunting expensive hypergrowth is their job.
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
            show_n = st.selectbox("Rows shown", ["Top 20", "Top 50", "Top 100", "All matches"],
                                  index=0, help="How many of the highest-scoring matches to display. "
                                  "All matches = every scanned name above the growth slider.")

        rv1, rv2, _rv3 = st.columns([1.9, 1.5, 2.6])
        with rv1:
            tenx_value_aware = st.toggle(
                "💸 Valuation-aware score", value=False,
                help="Adds a 6th component (20% weight): P/S ÷ YoY growth — the price paid per unit "
                     "of growth. A 24× sales name growing 68% scores worse than a 64× name growing "
                     "680%. Off by default: many past 10x-baggers looked expensive the whole way up. "
                     "The Δ column keeps tracking the base score.")
        with rv2:
            max_ps_choice = st.selectbox(
                "Max P/S", ["Any", "≤ 5", "≤ 10", "≤ 20"], index=0,
                help="Hard price-to-sales filter. Names with unknown P/S (no market cap yet) are kept.")

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
            ttm_ = tm.get("ttm_rev")
            ps_ = (mcap_ / ttm_) if (mcap_ and ttm_ and ttm_ > 0) else None
            base_, _sub, tags_ = tenx_score(tm, mcap_, mom_12_1.get(s_), m52_, ps=ps_)
            if base_ is None:
                continue
            tenx_scores_all[s_] = round(base_, 1)   # history always tracks the base score
            if tenx_value_aware:
                score_, _sub, tags_ = tenx_score(tm, mcap_, mom_12_1.get(s_), m52_,
                                                 ps=ps_, value_aware=True)
            else:
                score_ = base_
            accel_ = tm.get("rev_accel") if tm.get("rev_accel") is not None else tm.get("seq_accel")
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
                "52w Pos %": m52_ * 100 if m52_ is not None else None,
                "P/S": ps_,
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
            tdf = tdf[tdf["Rev YoY (Q) %"] >= float(min_yoy)]
            if max_ps_choice != "Any":
                _ps_lim = float(max_ps_choice.replace("≤", "").strip())
                tdf = tdf[tdf["P/S"].isna() | (tdf["P/S"] <= _ps_lim)]
            tdf = tdf.sort_values("10x", ascending=False)
            _n_map = {"Top 20": 20, "Top 50": 50, "Top 100": 100}
            shown_tenx = tdf.head(_n_map[show_n]) if show_n in _n_map else tdf
            st.caption(f"Showing **{len(shown_tenx)}** of **{len(tdf)}** names clearing the "
                       f"{min_yoy}% growth bar — use **Rows shown** to see more.")
            if shown_tenx.empty:
                st.info(f"No scanned name clears {min_yoy}% quarterly YoY revenue growth — lower the "
                        f"slider or scan more tickers.")
            else:
                with st.spinner("Checking earnings dates…"):
                    # First 100 displayed rows (1 free Yahoo call each on first
                    # load, then cached 3 days); beyond that the column shows —.
                    earn_dates = fetch_earnings_dates(list(shown_tenx["Sym"])[:100])

                # Numeric columns (formatting via column_config) so header-click
                # sorting is numeric, not alphabetical.
                tenx_display = pd.DataFrame({
                    "Ticker": shown_tenx["Ticker"], "Name": shown_tenx["Name"],
                    "Sector": shown_tenx["Sector"],
                    "10x": shown_tenx["10x"], "Δ": shown_tenx["10x Δ"],
                    "Rev YoY (Q)": shown_tenx["Rev YoY (Q) %"],
                    "Accel": shown_tenx["Accel ppt"], "QoQ": shown_tenx["QoQ %"],
                    "GM Δ": shown_tenx["GM Δ ppt"], "OM Δ": shown_tenx["OM Δ ppt"],
                    "12-1m": shown_tenx["12-1m %"], "52w": shown_tenx["52w Pos %"],
                    "P/S": shown_tenx["P/S"], "Mkt Cap": shown_tenx["Mkt Cap"],
                    "Earnings": shown_tenx["Sym"].map(lambda s: fmt_earnings(earn_dates.get(s))),
                    "Latest Q": shown_tenx["Latest Q"], "Signals": shown_tenx["Signals"],
                })
                tstyler = (tenx_display.style
                           .map(color_score, subset=["10x"])
                           .map(color_signed, subset=["Δ", "Rev YoY (Q)", "Accel", "QoQ",
                                                      "GM Δ", "OM Δ", "12-1m"])
                           .apply(row_bg_styler(shown_tenx["52w Pos %"]), axis=1))
                st.dataframe(
                    tstyler, width="stretch", hide_index=True,
                    height=(len(tenx_display) + 1) * 36 + 3,
                    column_config={
                        "Signals": st.column_config.TextColumn(width="large"),
                        "Name": st.column_config.TextColumn(width="medium"),
                        "10x": ncol("%.0f"), "Δ": ncol("%+.0f"),
                        "Rev YoY (Q)": ncol("%+.0f%%"), "Accel": ncol("%+.0fpt"),
                        "QoQ": ncol("%+.1f%%"), "GM Δ": ncol("%+.1fpt"), "OM Δ": ncol("%+.1fpt"),
                        "12-1m": ncol("%+.0f%%"), "52w": ncol("%.0f%%"), "P/S": ncol("%.1f"),
                        "Mkt Cap": st.column_config.NumberColumn(format="compact"),
                    },
                )
                st.caption("**Row shading:** 🟩 green = bottom 30% of the 52-week range (darker = closer to "
                           "the 52w low — exploding revenue AND a beaten-down price is the radar's dream setup) · "
                           "🟥 red = top 30% (darker = closer to the 52w high). "
                           "★ = also in your watchlist · sorted by 10x score · "
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
            vgate = (value_gate(look_fund.get("earnings_yield"), look_fund.get("ps_ratio"),
                                look_fund.get("rev_growth"))
                     if weights.get("Value", 0) >= 1.0 else 1.0)
            comp = min(100.0, max(0.0, base * gate * vgate + flag_adj)) if base is not None else None
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

elif nav == NAV_PUTS:
    # -----------------------------------------------------------------------
    # 💰 Options strategies — income + defined-risk expressions of the same
    # buy-fear / 10x theses the rest of the app surfaces.
    # -----------------------------------------------------------------------
    st.subheader("💰 Options strategies")
    st.caption(
        "Three ways to express what the other tabs find: **cash-secured puts** get you paid to wait "
        "for a green-zone entry price · **covered calls** rent out red-zone names you already own · "
        "**LEAPS calls** turn a 10x-radar thesis into a defined-risk position. Yahoo chains, ~15-min "
        "delayed — research, not execution. **Not advice.**"
    )
    if not HAVE_YF:
        st.warning("Install `yfinance` to enable the options screeners.")
    else:
        OPT_CSP = "🛡️ Cash-secured puts"
        OPT_CC = "🏠 Covered calls"
        OPT_LEAPS = "🚀 LEAPS calls"
        OPT_WHEEL = "🎡 Wheel screen"
        OPT_PCS = "📐 Put credit spreads"
        strategy = st.radio("strategy", [OPT_CSP, OPT_CC, OPT_LEAPS, OPT_WHEEL, OPT_PCS],
                            horizontal=True, key="opt_strategy", label_visibility="collapsed")
        us_df = df[(df["Region"] == "US") & df["Price"].notna()]
        today_iso = datetime.now().date().isoformat()

        # ------------------------------------------------------------------
        # 🛡️ Cash-secured puts — get PAID to wait for your buy-fear entry
        # ------------------------------------------------------------------
        if strategy == OPT_CSP:
            st.markdown("**Sell an OTM put on a stock you want cheaper** (ideally green-zone, near its "
                        "52-week low), with cash reserved for 100 shares: keep the premium, or get "
                        "assigned at the entry you wanted *minus* the premium.")
            green_df = us_df[us_df["52w Pos %"].notna() & (us_df["52w Pos %"] <= 40)] \
                .sort_values("52w Pos %")
            pc1, pc2 = st.columns([3, 1.6])
            with pc1:
                picked = st.multiselect(
                    "Stocks to screen (pre-filled with green-zone names, nearest 52-week low first)",
                    options=sorted(us_df["Ticker"]), default=list(green_df["Ticker"].head(8)),
                    max_selections=PUTS_MAX_TICKERS,
                    help=f"Each ticker ≈3 free Yahoo calls (cached {OPTIONS_TTL//3600}h). "
                         f"Max {PUTS_MAX_TICKERS} at a time.")
            with pc2:
                min_cushion = st.slider(
                    "Min cushion %", 0, 15, 5,
                    help="How far the stock can fall before the position loses at expiry "
                         "(spot → breakeven). Higher = safer, smaller premium.")
            if not picked:
                st.info("Pick at least one ticker — green-zone names usually carry the richest, "
                        "most rational premiums for this strategy.")
            else:
                _live = fetch_live_spots(picked)
                spots = {s: (_live.get(s) or (quotes.get(s) or {}).get("price")) for s in picked}
                spots = {s: p for s, p in spots.items() if p}
                with st.spinner(f"Fetching put chains for {len(spots)} tickers (15-65 days out)…"):
                    by_sym = fetch_options_all(spots, "csp")
                    earn_p = fetch_earnings_dates(list(spots))
                put_rows = []
                for s_, rows_ in by_sym.items():
                    m_ = us_df[us_df["Ticker"] == s_]
                    pos_ = m_["52w Pos %"].iloc[0] if len(m_) else None
                    for r_ in (rows_ or []):
                        ed = earn_p.get(s_)
                        put_rows.append({
                            "Ticker": s_, "52w Pos %": pos_,
                            "Spot": r_["spot"], "Strike": r_["strike"], "OTM %": r_["otm_pct"] * 100,
                            "Expiry": r_["expiry"], "DTE": r_["dte"],
                            "Premium": r_["premium"], "Src": r_["premium_src"],
                            "Yield %": r_["yield"] * 100, "Annualized %": r_["annualized"] * 100,
                            "Breakeven": r_["breakeven"], "Cushion %": r_["cushion"] * 100,
                            "IV %": r_["iv"] * 100 if r_["iv"] is not None else None,
                            "OI": r_["oi"], "Cash needed": r_["cash_needed"],
                            "Earnings pre-expiry": (f"⚠️ {ed}" if ed and today_iso <= ed <= r_["expiry"]
                                                    else "—"),
                        })
                pdf_ = pd.DataFrame(put_rows)
                if pdf_.empty:
                    st.info("No put candidates found — these names may not have listed options, or "
                            "quotes are empty outside US market hours.")
                else:
                    pdf_ = pdf_[pdf_["Cushion %"] >= float(min_cushion)] \
                        .sort_values("Annualized %", ascending=False).head(40)
                    if pdf_.empty:
                        st.info(f"Nothing clears a {min_cushion}% cushion — lower the slider, or "
                                f"accept that premiums are thin right now.")
                    else:
                        put_display = pd.DataFrame({
                            "Ticker": pdf_["Ticker"], "52w": pdf_["52w Pos %"],
                            "Spot": pdf_["Spot"], "Strike": pdf_["Strike"], "OTM": pdf_["OTM %"],
                            "Expiry": pdf_["Expiry"], "DTE": pdf_["DTE"],
                            "Premium": pdf_["Premium"], "Src": pdf_["Src"],
                            "Yield": pdf_["Yield %"], "Annualized": pdf_["Annualized %"],
                            "Breakeven": pdf_["Breakeven"], "Cushion": pdf_["Cushion %"],
                            "IV": pdf_["IV %"], "OI": pdf_["OI"],
                            "Cash/contract": pdf_["Cash needed"],
                            "Earnings": pdf_["Earnings pre-expiry"],
                        })
                        pstyler = (put_display.style
                                   .map(color_signed, subset=["Cushion", "Yield"])
                                   .apply(row_bg_styler(pdf_["52w Pos %"]), axis=1))
                        st.dataframe(
                            pstyler, width="stretch", hide_index=True,
                            height=(len(put_display) + 1) * 36 + 3,
                            column_config={
                                "52w": ncol("%.0f%%"), "Spot": ncol("$%.2f"),
                                "Strike": ncol("$%.2f"), "OTM": ncol("%.0f%% below"),
                                "DTE": ncol("%dd"), "Premium": ncol("$%.2f"),
                                "Yield": ncol("%.1f%%"), "Annualized": ncol("%.0f%%"),
                                "Breakeven": ncol("$%.2f"), "Cushion": ncol("%.1f%%"),
                                "IV": ncol("%.0f%%"),
                                "OI": st.column_config.NumberColumn(format="localized"),
                                "Cash/contract": st.column_config.NumberColumn(format="dollar"),
                            },
                        )
                        st.caption("Sorted by annualized premium yield · row shading = 52-week "
                                   "position (green = near the low) · **Premium** uses the bid when "
                                   "available (conservative) · **Cushion** = drop absorbed before "
                                   "breakeven · **⚠️ Earnings** = report lands before expiry.")
                        st.download_button("📥 Download put candidates as CSV",
                                           data=pdf_.to_csv(index=False).encode(),
                                           file_name=f"csp_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                           mime="text/csv")
            with st.expander("ℹ️ Cash-secured puts — mechanics & risks, plainly"):
                st.markdown(
                    """
Selling 1 put = agreeing to buy **100 shares at the strike** until expiry, for the premium up front,
with the full cash reserved (**Cash/contract**). **Breakeven** = strike − premium: your true entry if
assigned. **Annualized** flatters short expiries — it assumes you can repeat the trade at the same
premium, which you often can't. High premium always means high priced-in risk (see IV): there is no
free yield. The stock can gap far below your strike — you still buy at the strike; max loss is the
breakeven going to zero, same as owning from there. In a real selloff ALL your short puts get
assigned together — size for that day. Quotes are delayed; work from your broker's live bid/ask.
Selling puts caps your upside at the premium: if the stock rips, you miss the move. Not advice.
"""
                )

        # ------------------------------------------------------------------
        # 🏠 Covered calls — rent out the names sitting near their highs
        # ------------------------------------------------------------------
        elif strategy == OPT_CC:
            st.markdown("**Own 100 shares of a red-zone name?** Selling an OTM call collects premium "
                        "now and commits you to sell at the strike (higher than today) if called away "
                        "— a disciplined trim at a price you chose, on names stretched near their "
                        "52-week highs.")
            red_df = us_df[us_df["52w Pos %"].notna() & (us_df["52w Pos %"] >= 60)] \
                .sort_values("52w Pos %", ascending=False)
            cc1, cc2 = st.columns([3, 1.6])
            with cc1:
                picked_cc = st.multiselect(
                    "Stocks to screen (pre-filled with red-zone names, nearest 52-week high first — "
                    "only meaningful for stocks you own 100+ shares of)",
                    options=sorted(us_df["Ticker"]), default=list(red_df["Ticker"].head(8)),
                    max_selections=PUTS_MAX_TICKERS,
                    help=f"Each ticker ≈3 free Yahoo calls (cached {OPTIONS_TTL//3600}h).")
            with cc2:
                min_ann = st.slider(
                    "Min annualized yield %", 0, 30, 5,
                    help="Annualized premium yield on the share position. Higher yield = closer "
                         "strike = higher chance of being called away.")
            if not picked_cc:
                st.info("Pick tickers you actually hold — covered calls without the shares are naked "
                        "calls, a different (and dangerous) trade.")
            else:
                _live = fetch_live_spots(picked_cc)
                spots = {s: (_live.get(s) or (quotes.get(s) or {}).get("price")) for s in picked_cc}
                spots = {s: p for s, p in spots.items() if p}
                with st.spinner(f"Fetching call chains for {len(spots)} tickers (15-65 days out)…"):
                    by_sym = fetch_options_all(spots, "cc")
                    earn_c = fetch_earnings_dates(list(spots))
                cc_rows = []
                for s_, rows_ in by_sym.items():
                    m_ = us_df[us_df["Ticker"] == s_]
                    pos_ = m_["52w Pos %"].iloc[0] if len(m_) else None
                    for r_ in (rows_ or []):
                        ed = earn_c.get(s_)
                        cc_rows.append({
                            "Ticker": s_, "52w Pos %": pos_,
                            "Spot": r_["spot"], "Strike": r_["strike"],
                            "Headroom %": r_["otm_pct"] * 100,
                            "Expiry": r_["expiry"], "DTE": r_["dte"],
                            "Premium": r_["premium"], "Src": r_["premium_src"],
                            "Yield %": r_["yield"] * 100, "Annualized %": r_["annualized"] * 100,
                            "If called %": r_["called_return"] * 100,
                            "IV %": r_["iv"] * 100 if r_["iv"] is not None else None,
                            "OI": r_["oi"],
                            "Earnings pre-expiry": (f"⚠️ {ed}" if ed and today_iso <= ed <= r_["expiry"]
                                                    else "—"),
                        })
                cdf_ = pd.DataFrame(cc_rows)
                if cdf_.empty:
                    st.info("No call candidates found — check the names have listed options, or try "
                            "during US market hours.")
                else:
                    cdf_ = cdf_[cdf_["Annualized %"] >= float(min_ann)] \
                        .sort_values("Annualized %", ascending=False).head(40)
                    if cdf_.empty:
                        st.info(f"Nothing yields {min_ann}%+ annualized — premiums are thin (typical "
                                f"when volatility is becalmed). Lower the bar or wait for livelier markets.")
                    else:
                        cc_display = pd.DataFrame({
                            "Ticker": cdf_["Ticker"], "52w": cdf_["52w Pos %"],
                            "Spot": cdf_["Spot"], "Strike": cdf_["Strike"],
                            "Headroom": cdf_["Headroom %"],
                            "Expiry": cdf_["Expiry"], "DTE": cdf_["DTE"],
                            "Premium": cdf_["Premium"], "Src": cdf_["Src"],
                            "Yield": cdf_["Yield %"], "Annualized": cdf_["Annualized %"],
                            "If called": cdf_["If called %"],
                            "IV": cdf_["IV %"], "OI": cdf_["OI"],
                            "Earnings": cdf_["Earnings pre-expiry"],
                        })
                        cstyler = (cc_display.style
                                   .map(color_signed, subset=["Yield", "If called"])
                                   .apply(row_bg_styler(cdf_["52w Pos %"]), axis=1))
                        st.dataframe(
                            cstyler, width="stretch", hide_index=True,
                            height=(len(cc_display) + 1) * 36 + 3,
                            column_config={
                                "52w": ncol("%.0f%%"), "Spot": ncol("$%.2f"),
                                "Strike": ncol("$%.2f"),
                                "Headroom": ncol("%.0f%% up", "Room to run before the cap kicks in."),
                                "DTE": ncol("%dd"), "Premium": ncol("$%.2f"),
                                "Yield": ncol("%.1f%%"), "Annualized": ncol("%.0f%%"),
                                "If called": ncol("%+.1f%%",
                                                  "Total return if assigned: strike gain + premium."),
                                "IV": ncol("%.0f%%"),
                                "OI": st.column_config.NumberColumn(format="localized"),
                            },
                        )
                        st.caption("Sorted by annualized premium yield · row shading = 52-week "
                                   "position (red = near the high — the zone you're renting out) · "
                                   "**Headroom** = upside kept before the strike caps you · "
                                   "**If called** = total return should the shares get assigned.")
                        st.download_button("📥 Download covered-call candidates as CSV",
                                           data=cdf_.to_csv(index=False).encode(),
                                           file_name=f"cc_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                           mime="text/csv")
            with st.expander("ℹ️ Covered calls — mechanics & risks, plainly"):
                st.markdown(
                    """
1 covered call = you own **100 shares** and sell someone the right to buy them at the strike until
expiry. You keep the premium no matter what. If the stock finishes above the strike, your shares are
**called away** at the strike — total return = headroom + premium (**If called**), and you miss
everything above it. That's the real cost: covered calls systematically sell your best months. The
premium does NOT protect you on the way down — you're still long the shares with only the premium as
buffer. Writing calls over an earnings date (⚠️) doubles the odds of being capped through a gap.
Skip this on 10x-radar names you believe in — capping a potential multi-bagger for a few percent of
premium is the worst trade in this app. Not advice.
"""
                )

        # ------------------------------------------------------------------
        # 🚀 LEAPS calls — defined-risk expression of a 10x thesis
        # ------------------------------------------------------------------
        elif strategy == OPT_LEAPS:
            st.markdown("**Long-dated calls (9-26 months) on 10x-radar names**: instead of buying the "
                        "shares, buy the right to. Max loss = the premium, full stop; upside "
                        "participates in the thesis. The price: the stock must RISE past breakeven "
                        "by expiry, or the position expires worthless.")
            _, tenx_last = history_snapshot("tenx", 0)
            tenx_last = tenx_last or {}
            ranked_10x = [s for s in sorted(tenx_last, key=tenx_last.get, reverse=True)
                          if "." not in s]
            lp1, lp2 = st.columns([3, 1.6])
            with lp1:
                picked_lp = st.multiselect(
                    "Stocks to screen (pre-filled with the current 10x Radar leaders)",
                    options=(ranked_10x[:60] or sorted(s for s in TENX_SYMBOLS if "." not in s)),
                    default=ranked_10x[:6],
                    max_selections=PUTS_MAX_TICKERS,
                    help="Ranked by 10x score. Visit the 10x Radar tab once if this list is empty.")
            with lp2:
                max_be = st.slider(
                    "Max breakeven move %", 5, 40, 25,
                    help="Only show contracts where the stock needs to rise at most this much "
                         "by expiry to break even.")
            if not picked_lp:
                st.info("Pick at least one ticker — the 10x Radar tab populates the ranked list.")
            else:
                picked_scan = [s for s in picked_lp if s not in set(SYMBOLS)]
                extra_q = fetch_scan_quotes(api_key, picked_scan) if picked_scan else {}
                _live = fetch_live_spots(picked_lp)
                spots, pos_map = {}, {}
                for s_ in picked_lp:
                    q_ = quotes.get(s_) or extra_q.get(s_) or {}
                    p_ = _live.get(s_) or q_.get("price")
                    lo_, hi_ = q_.get("yearLow"), q_.get("yearHigh")
                    if p_:
                        spots[s_] = p_
                        pos_map[s_] = ((p_ - lo_) / (hi_ - lo_) * 100
                                       if (lo_ is not None and hi_ and hi_ > lo_) else None)
                with st.spinner(f"Fetching LEAPS chains for {len(spots)} tickers (9-26 months out)…"):
                    by_sym = fetch_options_all(spots, "leaps")
                lp_rows = []
                for s_, rows_ in by_sym.items():
                    for r_ in (rows_ or []):
                        lp_rows.append({
                            "Ticker": s_, "10x": tenx_last.get(s_), "52w Pos %": pos_map.get(s_),
                            "Spot": r_["spot"], "Strike": r_["strike"],
                            "Moneyness %": r_["moneyness"] * 100,
                            "Expiry": r_["expiry"], "Months": r_["dte"] / 30.44,
                            "Premium": r_["premium"], "Src": r_["premium_src"],
                            "Breakeven": r_["breakeven"], "BE move %": r_["be_move"] * 100,
                            "Cost/contract": r_["cost"], "Leverage x": r_["leverage"],
                            "IV %": r_["iv"] * 100 if r_["iv"] is not None else None,
                            "OI": r_["oi"],
                        })
                ldf_ = pd.DataFrame(lp_rows)
                if ldf_.empty:
                    st.info("No LEAPS found — smaller names often list nothing past ~9 months, and "
                            "quotes can be empty outside US market hours.")
                else:
                    ldf_ = ldf_[ldf_["BE move %"] <= float(max_be)] \
                        .sort_values(["BE move %"]).head(40)
                    if ldf_.empty:
                        st.info(f"Every contract needs more than a {max_be}% rise to break even — "
                                f"IV on these names is expensive. Raise the slider, or take that as "
                                f"the market's honest warning.")
                    else:
                        lp_display = pd.DataFrame({
                            "Ticker": ldf_["Ticker"], "10x": ldf_["10x"], "52w": ldf_["52w Pos %"],
                            "Spot": ldf_["Spot"], "Strike": ldf_["Strike"],
                            "Moneyness": ldf_["Moneyness %"], "Expiry": ldf_["Expiry"],
                            "Months": ldf_["Months"], "Premium": ldf_["Premium"], "Src": ldf_["Src"],
                            "Breakeven": ldf_["Breakeven"], "BE move": ldf_["BE move %"],
                            "Cost/contract": ldf_["Cost/contract"], "Leverage": ldf_["Leverage x"],
                            "IV": ldf_["IV %"], "OI": ldf_["OI"],
                        })
                        lstyler = (lp_display.style
                                   .map(color_score, subset=["10x"])
                                   .map(color_signed, subset=["Moneyness"])
                                   .apply(row_bg_styler(ldf_["52w Pos %"]), axis=1))
                        st.dataframe(
                            lstyler, width="stretch", hide_index=True,
                            height=(len(lp_display) + 1) * 36 + 3,
                            column_config={
                                "10x": ncol("%.0f", "10x Radar score (latest snapshot)."),
                                "52w": ncol("%.0f%%"), "Spot": ncol("$%.2f"),
                                "Strike": ncol("$%.2f"),
                                "Moneyness": ncol("%+.0f%%", "Strike vs spot: + = OTM, − = ITM."),
                                "Months": ncol("%.0fmo"), "Premium": ncol("$%.2f"),
                                "Breakeven": ncol("$%.2f"),
                                "BE move": ncol("%+.0f%%",
                                                "Rise needed by expiry just to break even."),
                                "Cost/contract": st.column_config.NumberColumn(format="dollar"),
                                "Leverage": ncol("%.1fx",
                                                 "Share exposure per premium dollar (not delta-adjusted)."),
                                "IV": ncol("%.0f%%"),
                                "OI": st.column_config.NumberColumn(format="localized"),
                            },
                        )
                        st.caption("Sorted by smallest breakeven move · **Premium** uses the ask "
                                   "(you're the buyer) · **Cost/contract** is also your max loss · "
                                   "row shading = 52-week position.")
                        st.download_button("📥 Download LEAPS candidates as CSV",
                                           data=ldf_.to_csv(index=False).encode(),
                                           file_name=f"leaps_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                           mime="text/csv")
            with st.expander("ℹ️ LEAPS calls — mechanics & risks, plainly"):
                st.markdown(
                    """
A LEAPS call is the right (not obligation) to buy 100 shares at the strike until an expiry 1-2 years
out. **Max loss = the premium — and losing it all is a normal outcome**, not a tail case: the stock
can rise slower than breakeven, chop sideways, or dip at the wrong moment, and the option still goes
to zero while the shareholder is fine. **BE move** is the honest hurdle: it's how much of your thesis
is already consumed by the option's price. IV on exactly the exciting radar names is expensive —
you're buying the thesis at a premium after the market noticed. LEAPS pay no dividends, and thin OI
means wide spreads (check **OI** before trusting the price). Position-size rule of thumb: money you'd
be genuinely fine watching go to zero. Not advice.
"""
                )

        # ------------------------------------------------------------------
        # 🎡 Wheel screen — the spreadsheet criteria, automated
        # ------------------------------------------------------------------
        elif strategy == OPT_WHEEL:
            st.markdown(
                "**Your wheel-sheet criteria, automated.** ✅ requires all three: **uptrend** (price "
                "above its 200-day average AND positive 12-1-month momentum), **sane valuation** "
                "(0 < P/E < 100; negative or missing P/E passes only with net cash), and **premium** "
                "≥ your bar at **~30 delta / ~30 DTE** (delta computed from each contract's IV). "
                "🟨 yellow rows missed EPS estimates last quarter — your sheet's highlight rule."
            )
            # Stock-level criteria from data already in memory.
            crit = {}
            for s_ in us_df["Ticker"]:
                q_ = quotes.get(s_) or {}
                f_ = fundamentals.get(s_) or {}
                p_, ma_, pe_ = q_.get("price"), q_.get("ma200"), q_.get("pe")
                mom_ = mom_12_1.get(s_)
                trend_ok = bool(p_ and ma_ and p_ > ma_ and (mom_ is None or mom_ > 0))
                net_cash = ((f_.get("net_debt_ebitda") is not None and f_["net_debt_ebitda"] < 0)
                            or (f_.get("debt_equity") is not None and f_["debt_equity"] < 0.10))
                if pe_ is None or pe_ <= 0:
                    pe_ok = net_cash
                else:
                    pe_ok = pe_ < 100 or net_cash
                crit[s_] = {"trend": trend_ok, "pe_ok": pe_ok, "pe": pe_, "net_cash": net_cash}

            passing = [s for s, c in crit.items() if c["trend"] and c["pe_ok"]]
            passing.sort(key=lambda s: (mom_12_1.get(s) if mom_12_1.get(s) is not None else -9),
                         reverse=True)
            wc1, wc2 = st.columns([3, 1.6])
            with wc1:
                picked_w = st.multiselect(
                    "Stocks to screen (pre-filled with names passing the trend + valuation tests, "
                    "strongest momentum first)",
                    options=sorted(us_df["Ticker"]), default=passing[:10],
                    max_selections=PUTS_MAX_TICKERS,
                    help="Add names that fail the stock tests if you want — the ✅/❌ columns "
                         "will show exactly which criterion they miss.")
            with wc2:
                min_cycle = st.slider(
                    "Min premium per cycle %", 0.5, 3.0, 1.5, 0.25,
                    help="Your sheet's bar: premium ÷ strike for one ~30-day cycle. "
                         "1.5% ≈ 18%+ annualized if repeatable.")

            if not picked_w:
                st.info("Pick at least one ticker.")
            else:
                _live = fetch_live_spots(picked_w)
                spots = {s: (_live.get(s) or (quotes.get(s) or {}).get("price")) for s in picked_w}
                spots = {s: p for s, p in spots.items() if p}
                with st.spinner(f"Testing ~30Δ/30DTE puts on {len(spots)} tickers…"):
                    by_sym = fetch_options_all(spots, "wheel")
                    surprises = fetch_surprises(list(spots))
                w_rows, no_chain = [], []
                for s_ in picked_w:
                    rows_ = by_sym.get(s_) or []
                    c_ = crit.get(s_, {"trend": False, "pe_ok": False, "pe": None, "net_cash": False})
                    if not rows_:
                        no_chain.append(s_)
                        continue
                    best = rows_[0]                      # highest cycle return in the Δ band
                    surp = surprises.get(s_)
                    meets = best["cycle_return"] * 100 >= min_cycle
                    w_rows.append({
                        "_ok": c_["trend"] and c_["pe_ok"] and meets,
                        "_missed": surp is not None and surp < 0,
                        "Ticker": s_,
                        "Wheel": "✅" if (c_["trend"] and c_["pe_ok"] and meets) else "❌",
                        "Trend": "✅" if c_["trend"] else "❌",
                        "P/E": c_["pe"],
                        "Val": "✅" if c_["pe_ok"] else "❌",
                        "Net cash": "💰" if c_["net_cash"] else "—",
                        "Last qtr %": surp,
                        "Spot": best["spot"], "Strike": best["strike"], "Δ": best["delta"],
                        "Expiry": best["expiry"], "DTE": best["dte"],
                        "Premium": best["premium"], "Src": best["premium_src"],
                        "Cycle %": best["cycle_return"] * 100,
                        "Annualized %": best["annualized"] * 100,
                        "Breakeven": best["breakeven"],
                        "IV %": best["iv"] * 100 if best["iv"] is not None else None,
                        "OI": best["oi"], "Cash needed": best["cash_needed"],
                    })
                if no_chain:
                    st.caption("No ~30Δ/30DTE put found (no listed options, empty quotes outside "
                               "market hours, or IV unusable): " + ", ".join(no_chain))
                wdf_ = pd.DataFrame(w_rows)
                if wdf_.empty:
                    st.info("No contracts to test — try during US market hours.")
                else:
                    wdf_ = wdf_.sort_values(["_ok", "Cycle %"], ascending=[False, False])
                    missed_map = wdf_["_missed"]

                    def _wheel_bg(row):
                        return (["background-color: rgba(250, 204, 21, 0.30);"] * len(row)
                                if missed_map.get(row.name) else [""] * len(row))

                    w_display = wdf_.drop(columns=["_ok", "_missed"])
                    wstyler = (w_display.style
                               .map(color_signed, subset=["Last qtr %", "Cycle %"])
                               .apply(_wheel_bg, axis=1))
                    st.dataframe(
                        wstyler, width="stretch", hide_index=True,
                        height=(len(w_display) + 1) * 36 + 3,
                        column_config={
                            "P/E": ncol("%.1f"),
                            "Last qtr %": ncol("%+.1f%%", "EPS surprise last reported quarter; "
                                                          "negative = missed (yellow row)."),
                            "Spot": ncol("$%.2f"), "Strike": ncol("$%.2f"),
                            "Δ": ncol("%.2f", "Black-Scholes put delta from the contract's IV."),
                            "DTE": ncol("%dd"), "Premium": ncol("$%.2f"),
                            "Cycle %": ncol("%.2f%%", "Premium ÷ strike for this cycle — "
                                                      "your sheet's 1.5% test."),
                            "Annualized %": ncol("%.0f%%"), "Breakeven": ncol("$%.2f"),
                            "IV %": ncol("%.0f%%"),
                            "OI": st.column_config.NumberColumn(format="localized"),
                            "Cash needed": st.column_config.NumberColumn(format="dollar"),
                        },
                    )
                    n_pass = int(wdf_["_ok"].sum())
                    st.caption(f"**{n_pass}/{len(wdf_)}** picked names pass all three wheel tests at "
                               f"a {min_cycle}% cycle bar · best ~30Δ contract shown per ticker · "
                               f"🟨 = missed estimates last quarter · sorted pass-first, richest "
                               f"premium first.")
                    st.download_button("📥 Download wheel screen as CSV",
                                       data=wdf_.drop(columns=["_ok", "_missed"])
                                       .to_csv(index=False).encode(),
                                       file_name=f"wheel_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                       mime="text/csv")
            with st.expander("ℹ️ How the sheet's criteria map here — and wheel risks"):
                st.markdown(
                    """
**Criterion → implementation.** *"Clear upward trend over 1.5 years"* → price above its 200-day
average AND positive 12-minus-1-month momentum (long uptrends keep both true; a fresh breakdown
fails fast). *"P/E under 100 or strong cash reserves, no negative P/E"* → 0 < P/E < 100 passes;
negative/missing P/E passes only with net cash (💰); P/E ≥ 100 passes only with net cash.
*"Min 1.5% return using 30 delta / 30 DTE"* → best put with Black-Scholes delta between −0.23 and
−0.37, 20-45 days out, premium ÷ strike vs your slider. *"Highlighted if missed a quarter"* →
🟨 row when the last reported EPS came in under estimates.

**Wheel risks.** The wheel earns steady premium until a real downtrend, when you're assigned into
falling names and the "income" becomes an unrealized loss you then write calls against below cost.
The trend test is the guard — respect it when it flips ❌ mid-position. Delta here is computed from
delayed IV, so treat the Δ column as approximate and confirm greeks in your broker. Not advice.
"""
                )

        # ------------------------------------------------------------------
        # 📐 Put credit spreads — defined max loss; XSP/SPX = no wash sales
        # ------------------------------------------------------------------
        else:
            st.markdown(
                "**The defined-risk version of selling puts**: sell a ~20-35Δ put, buy a cheaper put "
                "below it in the same expiry. Worst case is known up front (**Max loss** = width − "
                "credit) — no assignment surprise, far less buying power. On **XSP/SPX** the contracts "
                "are **§1256**: cash-settled, 60/40 tax treatment, and **wash-sale rules don't apply** "
                "— the structural fix for trading income without ticker-rotation bookkeeping."
            )
            idx_price = mkt.get("index_price")
            idx_spots = {"^XSP": idx_price / 10.0 if idx_price else None,
                         "^SPX": idx_price}
            green_df = us_df[us_df["52w Pos %"].notna() & (us_df["52w Pos %"] <= 40)] \
                .sort_values("52w Pos %")
            sc1, sc2 = st.columns([3, 1.6])
            with sc1:
                picked_s = st.multiselect(
                    "Underlyings (indexes first — §1256, no wash sales — then your green-zone names)",
                    options=list(INDEX_UNDERLYINGS) + sorted(us_df["Ticker"]),
                    default=["^XSP"] + list(green_df["Ticker"].head(4)),
                    max_selections=PUTS_MAX_TICKERS,
                    format_func=lambda s: INDEX_UNDERLYINGS.get(s, s),
                    help="^XSP is 1/10th the S&P 500 — spread widths ~$100-500 of risk, "
                         "right-sized for smaller accounts. ^SPX is the full-size version.")
            with sc2:
                min_ror = st.slider(
                    "Min return on risk %", 5, 50, 15,
                    help="Credit ÷ max loss. 15% ≈ risking $85 to make $15 per $1-wide spread — "
                         "higher demands better pay but closer strikes.")
            if not picked_s:
                st.info("Pick at least one underlying — ^XSP is the wash-sale-free default.")
            else:
                _live = fetch_live_spots(picked_s)
                spots = {}
                for s_ in picked_s:
                    if s_ in idx_spots:
                        p_ = _live.get(s_) or idx_spots[s_]
                    else:
                        p_ = _live.get(s_) or (quotes.get(s_) or {}).get("price")
                    if p_:
                        spots[s_] = p_
                if len(spots) < len(picked_s):
                    missing_ = [s for s in picked_s if s not in spots]
                    st.caption("No spot price yet for: " + ", ".join(missing_) +
                               (" — index levels come from the market-context quote; refresh prices "
                                "if it's blank." if any(s in idx_spots for s in missing_) else ""))
                with st.spinner(f"Building spreads for {len(spots)} underlyings (20-50 days out)…"):
                    by_sym = fetch_options_all(spots, "pcs")
                    earn_s = fetch_earnings_dates([s for s in spots if s not in idx_spots])
                sp_rows = []
                for s_, rows_ in by_sym.items():
                    is_idx = s_ in idx_spots
                    m_ = us_df[us_df["Ticker"] == s_]
                    pos_ = (mkt.get("index_52w_pos") if is_idx
                            else (m_["52w Pos %"].iloc[0] if len(m_) else None))
                    for r_ in (rows_ or []):
                        ed = earn_s.get(s_)
                        sp_rows.append({
                            "Ticker": s_, "52w Pos %": pos_,
                            "Spot": r_["spot"], "Short K": r_["short_strike"],
                            "Long K": r_["long_strike"], "Width": r_["width"],
                            "Δ short": r_["delta"], "Expiry": r_["expiry"], "DTE": r_["dte"],
                            "Credit": r_["credit"], "Max loss": r_["max_loss"],
                            "RoR %": r_["ror"] * 100, "Annualized %": r_["annualized"] * 100,
                            "POP ≈ %": r_["pop"] * 100,
                            "Breakeven": r_["breakeven"], "Cushion %": r_["cushion"] * 100,
                            "IV %": r_["iv"] * 100 if r_["iv"] is not None else None,
                            "OI (min)": r_["oi"], "BP / spread": r_["bp_needed"],
                            "Tax": "🛡️ §1256 — no wash sale" if is_idx else "equity",
                            "Earnings": (f"⚠️ {ed}" if (not is_idx and ed
                                                        and today_iso <= ed <= r_["expiry"]) else "—"),
                        })
                sdf_ = pd.DataFrame(sp_rows)
                if sdf_.empty:
                    st.info("No spreads found — index chains can come back empty outside US market "
                            "hours, and thin names may lack usable quotes. Try during the trading day.")
                else:
                    sdf_ = sdf_[sdf_["RoR %"] >= float(min_ror)] \
                        .sort_values("RoR %", ascending=False).head(40)
                    if sdf_.empty:
                        st.info(f"Nothing pays {min_ror}%+ on risk in the ~20-35Δ band — premiums are "
                                f"thin (typical when the entry meter reads greed/calm). Lower the bar "
                                f"or wait for volatility.")
                    else:
                        sp_display = pd.DataFrame({
                            "Ticker": sdf_["Ticker"], "52w": sdf_["52w Pos %"],
                            "Spot": sdf_["Spot"], "Short K": sdf_["Short K"],
                            "Long K": sdf_["Long K"], "Width": sdf_["Width"],
                            "Δ": sdf_["Δ short"], "Expiry": sdf_["Expiry"], "DTE": sdf_["DTE"],
                            "Credit": sdf_["Credit"], "Max loss": sdf_["Max loss"],
                            "RoR": sdf_["RoR %"], "Ann.": sdf_["Annualized %"],
                            "POP≈": sdf_["POP ≈ %"], "Breakeven": sdf_["Breakeven"],
                            "Cushion": sdf_["Cushion %"], "IV": sdf_["IV %"],
                            "OI": sdf_["OI (min)"], "BP/spread": sdf_["BP / spread"],
                            "Tax": sdf_["Tax"], "Earnings": sdf_["Earnings"],
                        })
                        sstyler = (sp_display.style
                                   .map(color_signed, subset=["RoR", "Cushion"])
                                   .apply(row_bg_styler(sdf_["52w Pos %"]), axis=1))
                        st.dataframe(
                            sstyler, width="stretch", hide_index=True,
                            height=(len(sp_display) + 1) * 36 + 3,
                            column_config={
                                "52w": ncol("%.0f%%"), "Spot": ncol("$%.2f"),
                                "Short K": ncol("$%.0f"), "Long K": ncol("$%.0f"),
                                "Width": ncol("$%.0f"),
                                "Δ": ncol("%.2f", "Short-leg Black-Scholes delta from chain IV."),
                                "DTE": ncol("%dd"), "Credit": ncol("$%.2f"),
                                "Max loss": ncol("$%.2f", "Per share; ×100 per spread. Known up front."),
                                "RoR": ncol("%.0f%%", "Credit ÷ max loss for the cycle."),
                                "Ann.": ncol("%.0f%%"),
                                "POP≈": ncol("%.0f%%", "≈ probability the short strike expires OTM "
                                                       "(1 − |Δ|). Approximation, not a promise."),
                                "Breakeven": ncol("$%.2f"), "Cushion": ncol("%.1f%%"),
                                "IV": ncol("%.0f%%"),
                                "OI": st.column_config.NumberColumn(format="localized"),
                                "BP/spread": st.column_config.NumberColumn(format="dollar"),
                            },
                        )
                        st.caption("Sorted by return on risk · **Credit** = short bid − long ask "
                                   "(conservative; live mid is usually better) · **POP≈** and **Δ** "
                                   "derive from delayed IV · 🛡️ index rows are §1256 contracts — "
                                   "cash-settled, no wash-sale rules, 60/40 tax (confirm with a tax "
                                   "professional).")
                        st.download_button("📥 Download spread candidates as CSV",
                                           data=sdf_.to_csv(index=False).encode(),
                                           file_name=f"pcs_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                           mime="text/csv")
            with st.expander("ℹ️ Put credit spreads — mechanics & risks, plainly"):
                st.markdown(
                    """
**The trade.** Sell a put around 20-35Δ, buy a put 1-10% further down, same expiry. You collect the
**Credit**; your worst case is **Max loss** (width − credit) no matter what the stock does — that's
the entire point versus a cash-secured put. Buying power needed is the max loss, not the full strike,
so returns *on capital at risk* look high (**RoR**); remember the flip side — you lose that capital
fully if the underlying finishes below the long strike.

**Why XSP/SPX here.** Index options are **§1256 contracts**: marked-to-market at year-end, gains
taxed 60% long-term / 40% short-term regardless of holding period, **and wash-sale rules do not
apply** — you can trade XSP spreads every week without the rotation bookkeeping single names need.
They're also cash-settled European options: no early assignment, no surprise shares, no single-stock
earnings gaps. XSP is 1/10th of SPX, so a $5-wide XSP spread risks ≈$400-480 — sized for regular
accounts. (Tax treatment is *not* advice — confirm §1256 handling with your tax professional.)

**The honest math for your 2-3%/month goal.** A 30Δ spread wins often but loses ~5-6× the credit
when it loses. At 15% RoR you need to win ~87% of the time just to break even before costs — right
around what 30Δ delivers. The realistic edge comes from selling when IV is elevated (fear) and
sizing so one full loss costs ≤2-3% of the account: that's how "income" survives the losing month.
Spreads near **⚠️ earnings** or after the entry meter flips to fear pay more for a reason.

**Execution notes.** Enter as a single spread order at the mid, never two separate legs. Thin OI on
either leg (see **OI**) means wide fills — indexes are the most liquid thing listed. Quotes here are
~15-min delayed. Not advice.
"""
                )
        st.divider()
