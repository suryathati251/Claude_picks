"""
fundamentals.py — reported-financials layer for the watchlist screener.

Replaces analyst-target ranking with a bottom-up composite score built from
live FMP fundamentals: earnings yield (EBIT/EV), ROIC, revenue growth, and
free-cash-flow yield. Pure functions only — caching/threading live in app.py,
mirroring the existing fetch_profile pattern.

Design notes
------------
* FMP "stable" field names vary by plan and have changed over time, so every
  metric is pulled via a candidate-key list and we LOG (never silently drop)
  anything we can't map. Run once and check the log to confirm field coverage
  for your plan, then prune the candidate lists.
* Scoring is rank-based (percentile within the universe), Greenblatt-style:
  robust to outliers and to missing data, since weights renormalize over only
  the factors a given ticker actually has.
"""
from __future__ import annotations

import logging
from typing import Optional

import requests

logger = logging.getLogger("fundamentals")

FMP_BASE = "https://financialmodelingprep.com/stable"

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

# Candidate FMP field names per metric (first present wins). Add to these if
# the startup log reports a metric as unmapped for your plan.
_FIELD_CANDIDATES = {
    "earnings_yield": ["earningsYieldTTM", "earningsYield"],
    "roic":           ["roicTTM", "returnOnInvestedCapitalTTM", "returnOnInvestedCapital"],
    "fcf_yield":      ["freeCashFlowYieldTTM", "freeCashFlowYield"],
    "gross_margin":   ["grossProfitMarginTTM", "grossProfitMargin", "grossMarginTTM"],
    "fcf_margin":     ["freeCashFlowMarginTTM", "freeCashFlowMargin"],
    # ev/ebitda kept here in case you want to add it as a (lower_better) factor
    "ev_ebitda":      ["enterpriseValueOverEBITDATTM", "evToEBITDATTM", "enterpriseValueMultipleTTM"],
}
_GROWTH_CANDIDATES = ["revenueGrowth", "growthRevenue", "revenueGrowthTTM"]


def _first_present(record: dict, candidates: list[str], metric_name: str, symbol: str) -> Optional[float]:
    """Return the first candidate key present & numeric; log if none match."""
    for key in candidates:
        if key in record and record[key] is not None:
            try:
                return float(record[key])
            except (TypeError, ValueError):
                continue
    logger.warning("%s: no field mapped for '%s' (tried %s)", symbol, metric_name, candidates)
    return None


def _get_json(url: str, params: dict, symbol: str, what: str):
    """GET + parse one FMP endpoint, returning the first record dict or None."""
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:160]}")
        data = r.json()
    except Exception as e:  # noqa: BLE001 — we want to swallow + log, not crash a thread
        logger.warning("%s: %s fetch failed: %s", symbol, what, str(e)[:160])
        return None
    if isinstance(data, dict) and "Error Message" in data:
        logger.warning("%s: %s FMP error: %s", symbol, what, str(data["Error Message"])[:160])
        return None
    if isinstance(data, list) and data:
        return data[0]
    if isinstance(data, dict) and data:
        return data
    return None


def fetch_fundamentals(symbol: str, api_key: str) -> dict:
    """
    Fetch reported fundamentals for one symbol. Returns a dict with keys from
    FACTORS (values may be None when unavailable). Network-only; let app.py
    wrap this with its cache/stale-fallback machinery.

    Uses 2 calls/ticker: key-metrics-ttm (yield/ROIC/FCF) + financial-growth
    (revenue growth). With profile that's 3 calls/ticker — see note in app.py
    about the FMP free-tier daily budget.
    """
    out = {k: None for k in set(FACTORS) | set(GROWTH_FACTORS) | {"gross_margin", "fcf_margin", "rule_of_40"}}

    km = _get_json(f"{FMP_BASE}/key-metrics-ttm", {"symbol": symbol, "apikey": api_key},
                   symbol, "key-metrics-ttm")
    if km:
        out["earnings_yield"] = _first_present(km, _FIELD_CANDIDATES["earnings_yield"], "earnings_yield", symbol)
        out["roic"] = _first_present(km, _FIELD_CANDIDATES["roic"], "roic", symbol)
        out["fcf_yield"] = _first_present(km, _FIELD_CANDIDATES["fcf_yield"], "fcf_yield", symbol)
        # Margins sometimes live in key-metrics-ttm; if not, they stay None and the
        # factor gracefully drops out (no extra API call spent here — see ratios note).
        out["gross_margin"] = _first_present(km, _FIELD_CANDIDATES["gross_margin"], "gross_margin", symbol)
        out["fcf_margin"] = _first_present(km, _FIELD_CANDIDATES["fcf_margin"], "fcf_margin", symbol)

    fg = _get_json(f"{FMP_BASE}/financial-growth",
                   {"symbol": symbol, "period": "annual", "limit": 1, "apikey": api_key},
                   symbol, "financial-growth")
    if fg:
        out["rev_growth"] = _first_present(fg, _GROWTH_CANDIDATES, "rev_growth", symbol)

    # Rule of 40 = revenue growth % + FCF margin % — the canonical "is this growth
    # healthy or just cash-burning" check. Derived from data already fetched.
    if out["rev_growth"] is not None and out["fcf_margin"] is not None:
        out["rule_of_40"] = (out["rev_growth"] + out["fcf_margin"]) * 100.0

    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _percentile_ranks(values: dict[str, Optional[float]], higher_better: bool) -> dict[str, float]:
    """
    Map ticker->raw value to ticker->percentile in [0,1] among the tickers that
    HAVE a value. Missing values get no rank (excluded from this factor).
    """
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
    """
    Given {ticker: {factor: raw_value}}, return
    {ticker: {"score": float|None, "coverage": int, "ranks": {factor: pct}}}.

    score = weighted mean of available factor percentiles, with weights
    renormalized over only the factors present for that ticker. coverage = how
    many of the factors were available (use it to distrust thin rows).
    """
    tickers = list(fund_by_ticker.keys())

    # Build per-factor percentile maps across the whole universe.
    factor_ranks: dict[str, dict[str, float]] = {}
    for fname, cfg in factors.items():
        raw = {t: fund_by_ticker[t].get(fname) for t in tickers}
        factor_ranks[fname] = _percentile_ranks(raw, cfg["higher_better"])

    results: dict[str, dict] = {}
    for t in tickers:
        num = 0.0
        denom = 0.0
        ranks_here = {}
        for fname, cfg in factors.items():
            pct = factor_ranks[fname].get(t)
            if pct is None:
                continue
            ranks_here[fname] = pct
            num += cfg["weight"] * pct
            denom += cfg["weight"]
        score = (num / denom) if denom > 0 else None
        results[t] = {"score": score, "coverage": len(ranks_here), "ranks": ranks_here}
    return results


# ---------------------------------------------------------------------------
# Target sanity guard — catches the split/stale mismatch (e.g. NFLX 358 vs ~83)
# ---------------------------------------------------------------------------
def target_is_sane(target: Optional[float], lo: Optional[float], hi: Optional[float],
                   hi_mult: float = 2.0, lo_mult: float = 0.3) -> bool:
    """
    False when a static target sits implausibly outside the live 52-week range,
    which is the fingerprint of a split-adjustment or stale-data mismatch.
    Returns True when we lack the inputs to judge (don't flag what we can't check).
    """
    if target is None or lo is None or hi is None or hi <= 0:
        return True
    if target > hi * hi_mult:
        return False
    if target < lo * lo_mult:
        return False
    return True
