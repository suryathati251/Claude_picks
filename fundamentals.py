"""
fundamentals.py — reported-financials layer for the watchlist screener.

Free-tier-friendly redesign
---------------------------
The old version asked FMP for *pre-computed* ratio fields (e.g. ``roicTTM``,
``grossProfitMarginTTM``) whose names vary by plan and have changed over time, so
most of them mapped to nothing and the board filled with blanks. We now derive
the ratios ourselves from RAW statement numbers (``revenue``, ``grossProfit``,
``operatingIncome``, ``netIncome``) whose field names are stable, and only lean on
``key-metrics-ttm`` for the few capital-efficiency metrics that aren't trivially
derivable (ROIC, FCF yield).

Cost: **2 calls per ticker**
  1. ``key-metrics-ttm``  -> ROIC, FCF yield, (earnings yield)
  2. ``income-statement`` (annual, last 2) -> revenue growth + gross/operating/net margins

Prices come from a single ``batch-quote`` in app.py, so the whole ~89-ticker board
fits inside the 250-requests/day free cap. Combined with app.py's disk cache and
per-run budget, a refresh costs almost nothing once the cache is warm.

Rate-limit aware: a 429 / "Limit Reach" response raises ``FMPRateLimitError`` so the
caller can stop early and keep showing cached values instead of blanking the board.

Pure functions only — caching / threading / budgeting live in app.py.
"""
from __future__ import annotations

import logging
from typing import Optional

import requests

logger = logging.getLogger("fundamentals")

FMP_BASE = "https://financialmodelingprep.com/stable"
_TIMEOUT = 15


class FMPRateLimitError(RuntimeError):
    """Raised when FMP signals the request/bandwidth allowance is exhausted."""


# ---------------------------------------------------------------------------
# Factor config — edit weights / direction here. higher_better=True means a
# bigger raw value should rank better (earnings yield); False means smaller is
# better (e.g. EV/EBITDA if you add it).
# ---------------------------------------------------------------------------
FACTORS = {
    "earnings_yield": {"weight": 1.0, "higher_better": True},
    "roic":           {"weight": 1.0, "higher_better": True},
    "rev_growth":     {"weight": 1.0, "higher_better": True},
    "fcf_yield":      {"weight": 1.0, "higher_better": True},
}

# Growth / asymmetric-upside model. Surfaces the CHARACTERISTICS that hypergrowth
# winners tend to share — fast revenue growth, high & scalable gross margins, and
# a healthy Rule-of-40 (growth + profitability). It does NOT predict which names
# 10x; most won't. Treat a high growth score as "belongs in the speculative basket
# to research," sized small, never as a forecast.
GROWTH_FACTORS = {
    "rev_growth":   {"weight": 1.5, "higher_better": True},
    "gross_margin": {"weight": 1.0, "higher_better": True},
    "rule_of_40":   {"weight": 1.0, "higher_better": True},
    "fcf_yield":    {"weight": 0.5, "higher_better": True},  # rewards self-funding growth
}

# Every metric the layer can produce. app.py iterates these, so keep in sync.
METRIC_KEYS = (
    "earnings_yield", "roic", "fcf_yield", "rev_growth",
    "gross_margin", "operating_margin", "net_margin", "fcf_margin", "rule_of_40",
)

# Candidate FMP field names for the few metrics we still read pre-computed.
# First present & numeric wins; we fall back to derivation when none match.
_ROIC_KEYS = ["returnOnInvestedCapitalTTM", "roicTTM", "returnOnInvestedCapital",
              "returnOnCapitalEmployedTTM", "returnOnCapitalEmployed"]
_FCFY_KEYS = ["freeCashFlowYieldTTM", "freeCashFlowYield"]
_EY_KEYS   = ["earningsYieldTTM", "earningsYield"]
_FCFM_KEYS = ["freeCashFlowMarginTTM", "freeCashFlowMargin"]


# ---------------------------------------------------------------------------
# Low-level fetch helpers
# ---------------------------------------------------------------------------
def _looks_rate_limited(text: str) -> bool:
    t = (text or "").lower()
    return ("limit reach" in t or "upgrade your plan" in t
            or "too many requests" in t or "bandwidth" in t)


def _get(url: str, params: dict, symbol: str, what: str):
    """GET one FMP endpoint. Returns parsed JSON (list or dict) or None on a
    non-fatal miss. Raises FMPRateLimitError when the quota is exhausted so the
    caller can stop the whole batch instead of hammering a dead quota."""
    try:
        r = requests.get(url, params=params, timeout=_TIMEOUT)
    except Exception as e:  # noqa: BLE001 — network hiccup: log + treat as miss
        logger.warning("%s: %s request error: %s", symbol, what, str(e)[:160])
        return None

    if r.status_code == 429 or _looks_rate_limited(r.text):
        raise FMPRateLimitError(f"{symbol}/{what}: {r.text[:120]}")
    if r.status_code != 200:
        logger.warning("%s: %s HTTP %s: %s", symbol, what, r.status_code, r.text[:160])
        return None

    try:
        data = r.json()
    except ValueError:
        logger.warning("%s: %s non-JSON: %s", symbol, what, r.text[:120])
        return None

    if isinstance(data, dict) and ("Error Message" in data or "Error" in data):
        msg = str(data.get("Error Message") or data.get("Error"))
        if _looks_rate_limited(msg):
            raise FMPRateLimitError(f"{symbol}/{what}: {msg[:120]}")
        logger.warning("%s: %s FMP error: %s", symbol, what, msg[:160])
        return None
    return data


def _first_record(data) -> Optional[dict]:
    if isinstance(data, list):
        return data[0] if data else None
    if isinstance(data, dict) and data:
        return data
    return None


def _num(record: dict, *keys) -> Optional[float]:
    """First key present & numeric, else None."""
    for k in keys:
        if k in record and record[k] is not None:
            try:
                return float(record[k])
            except (TypeError, ValueError):
                continue
    return None


def _first_present(record: dict, candidates: list[str], metric: str, symbol: str,
                   warn: bool = True) -> Optional[float]:
    val = _num(record, *candidates)
    if val is None and warn:
        logger.info("%s: no field mapped for '%s' (tried %s)", symbol, metric, candidates)
    return val


# ---------------------------------------------------------------------------
# Public: fetch one ticker's fundamentals (2 network calls)
# ---------------------------------------------------------------------------
def fetch_fundamentals(symbol: str, api_key: str, market_cap: Optional[float] = None) -> dict:
    """Return a dict keyed by METRIC_KEYS (values may be None when unavailable).

    Network-only; let app.py wrap this with cache / budget / stale-fallback.
    Raises FMPRateLimitError if FMP signals the quota is spent.
    """
    out: dict = {k: None for k in METRIC_KEYS}

    # 1) key-metrics-ttm — ROIC, FCF yield, earnings yield (capital efficiency).
    km = _first_record(_get(f"{FMP_BASE}/key-metrics-ttm",
                            {"symbol": symbol, "apikey": api_key}, symbol, "key-metrics-ttm"))
    if km:
        out["roic"]           = _first_present(km, _ROIC_KEYS, "roic", symbol)
        out["fcf_yield"]      = _first_present(km, _FCFY_KEYS, "fcf_yield", symbol)
        out["earnings_yield"] = _first_present(km, _EY_KEYS, "earnings_yield", symbol, warn=False)
        out["fcf_margin"]     = _first_present(km, _FCFM_KEYS, "fcf_margin", symbol, warn=False)

    # 2) income-statement (annual, last 2) — revenue growth + margins from RAW
    #    numbers, so we don't depend on FMP's pre-computed ratio field names.
    inc = _get(f"{FMP_BASE}/income-statement",
               {"symbol": symbol, "period": "annual", "limit": 2, "apikey": api_key},
               symbol, "income-statement")
    rows = inc if isinstance(inc, list) else ([inc] if isinstance(inc, dict) and inc else [])
    if rows:
        cur = rows[0]
        rev = _num(cur, "revenue")
        gp  = _num(cur, "grossProfit")
        opi = _num(cur, "operatingIncome")
        ni  = _num(cur, "netIncome")
        if rev and rev != 0:
            if gp is not None:
                out["gross_margin"] = gp / rev
            if opi is not None:
                out["operating_margin"] = opi / rev
            if ni is not None:
                out["net_margin"] = ni / rev
        if len(rows) >= 2:
            prev_rev = _num(rows[1], "revenue")
            if rev is not None and prev_rev and prev_rev != 0:
                out["rev_growth"] = (rev - prev_rev) / abs(prev_rev)
        # Earnings-yield fallback when key-metrics didn't carry it: NI / market cap.
        if out["earnings_yield"] is None and ni is not None and market_cap:
            out["earnings_yield"] = ni / market_cap

    # 3) Rule of 40 = revenue growth % + profitability margin %.
    #    Prefer FCF margin (canonical); fall back to operating margin, which is
    #    always derivable from the income statement, so Rule40 stops being blank.
    if out["rev_growth"] is not None:
        margin = out["fcf_margin"]
        if margin is None:
            margin = out["operating_margin"]
        if margin is not None:
            out["rule_of_40"] = (out["rev_growth"] + margin) * 100.0

    return out


def has_any(fund: dict) -> bool:
    """True if at least one metric was populated (used to decide whether to cache)."""
    return any(fund.get(k) is not None for k in METRIC_KEYS)


# ---------------------------------------------------------------------------
# Scoring — rank-based composite, robust to outliers and missing data.
# ---------------------------------------------------------------------------
def _percentile_ranks(values: dict[str, Optional[float]], higher_better: bool) -> dict[str, float]:
    """Map ticker->raw value to ticker->percentile in [0,1] among tickers that
    HAVE a value. Missing values get no rank (excluded from this factor)."""
    present = {t: v for t, v in values.items() if v is not None}
    n = len(present)
    if n == 0:
        return {}
    if n == 1:
        return {t: 1.0 for t in present}
    ranks = {}
    for i, (t, _) in enumerate(sorted(present.items(), key=lambda kv: kv[1], reverse=higher_better)):
        ranks[t] = 1.0 - (i / (n - 1))  # best -> 1.0, worst -> 0.0
    return ranks


def compute_composite_scores(fund_by_ticker: dict[str, dict],
                             factors: dict = FACTORS) -> dict[str, dict]:
    """Given {ticker: {factor: raw_value}}, return
    {ticker: {"score": float|None, "coverage": int, "ranks": {factor: pct}}}.

    score = weighted mean of available factor percentiles, weights renormalized
    over only the factors present for that ticker. coverage = how many factors
    were available (use it to distrust thin rows)."""
    tickers = list(fund_by_ticker.keys())

    factor_ranks: dict[str, dict[str, float]] = {}
    for fname, cfg in factors.items():
        raw = {t: (fund_by_ticker.get(t) or {}).get(fname) for t in tickers}
        factor_ranks[fname] = _percentile_ranks(raw, cfg["higher_better"])

    results: dict[str, dict] = {}
    for t in tickers:
        num = denom = 0.0
        ranks_here = {}
        for fname, cfg in factors.items():
            pct = factor_ranks[fname].get(t)
            if pct is None:
                continue
            ranks_here[fname] = pct
            num += cfg["weight"] * pct
            denom += cfg["weight"]
        results[t] = {
            "score": (num / denom) if denom > 0 else None,
            "coverage": len(ranks_here),
            "ranks": ranks_here,
        }
    return results


# ---------------------------------------------------------------------------
# Target sanity guard — catches the split/stale mismatch (e.g. NFLX 358 vs ~83)
# ---------------------------------------------------------------------------
def target_is_sane(target: Optional[float], lo: Optional[float], hi: Optional[float],
                   hi_mult: float = 2.0, lo_mult: float = 0.3) -> bool:
    """False when a static target sits implausibly outside the live 52-week range,
    the fingerprint of a split-adjustment or stale-data mismatch. Returns True when
    we lack the inputs to judge (don't flag what we can't check)."""
    if target is None or lo is None or hi is None or hi <= 0:
        return True
    if target > hi * hi_mult:
        return False
    if target < lo * lo_mult:
        return False
    return True
