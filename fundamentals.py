"""
fundamentals.py — reported-financials + scoring layer for the watchlist.

Data (2 calls/ticker, free-tier friendly)
-----------------------------------------
Metrics are derived from RAW statement numbers (stable field names) instead of
FMP's pre-computed ratio fields (which vary by plan and caused the old blanks):
  1. ``key-metrics-ttm``  -> ROIC, FCF yield, (earnings yield)
  2. ``income-statement`` (annual, last 2) -> revenue growth, gross/operating/net
     margins, AND a Piotroski-style SAFETY score (profitable, cash-generative,
     growing, margins improving) — all from data we already fetch.
Momentum (52-week position, price vs 200-day average) is derived in app.py from
the single batch-quote, so it costs no extra calls.

Scoring (sector-neutral, multi-factor)
--------------------------------------
Each metric is percentile-ranked WITHIN its sector (universe fallback for thin
sectors), so a bank's earnings yield competes with other banks, not with software.
Ranks are equal-weighted into five family sub-scores — Value, Quality, Growth,
Momentum, Safety — which are then blended per a selectable "lens". Equal weighting
is deliberate: tuned weights overfit and rarely survive out of sample.

Everything here is pure (no Streamlit / threading). A 429 / "Limit Reach"
response raises ``FMPRateLimitError`` so app.py can stop early and keep cache.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import requests

logger = logging.getLogger("fundamentals")

FMP_BASE = "https://financialmodelingprep.com/stable"
_TIMEOUT = 15


class FMPRateLimitError(RuntimeError):
    """Raised when FMP signals the request/bandwidth allowance is exhausted."""


# ---------------------------------------------------------------------------
# Metric universe
# ---------------------------------------------------------------------------
# Every metric the layer can carry. app.py fills mom_* from the quote.
METRIC_KEYS = (
    "earnings_yield", "roic", "fcf_yield", "rev_growth",
    "gross_margin", "operating_margin", "net_margin", "fcf_margin", "rule_of_40",
    "mom_52w", "mom_ma200", "safety",
)

# Candidate FMP field names for the few metrics we still read pre-computed.
_ROIC_KEYS = ["returnOnInvestedCapitalTTM", "roicTTM", "returnOnInvestedCapital",
              "returnOnCapitalEmployedTTM", "returnOnCapitalEmployed"]
_FCFY_KEYS = ["freeCashFlowYieldTTM", "freeCashFlowYield"]
_EY_KEYS   = ["earningsYieldTTM", "earningsYield"]
_FCFM_KEYS = ["freeCashFlowMarginTTM", "freeCashFlowMargin"]


# ---------------------------------------------------------------------------
# Factor families & lens presets  (higher_better per metric)
# ---------------------------------------------------------------------------
FAMILIES: dict[str, dict[str, bool]] = {
    "Value":    {"earnings_yield": True, "fcf_yield": True},
    "Quality":  {"roic": True, "gross_margin": True, "operating_margin": True},
    "Growth":   {"rev_growth": True, "rule_of_40": True},
    "Momentum": {"mom_52w": True, "mom_ma200": True},
    "Safety":   {"safety": True},
}

# Lens = weights over families. All sub-scores always display; the lens only sets
# the headline composite + sort. Default is a balanced Value+Quality+Momentum.
LENSES: dict[str, dict[str, float]] = {
    "Blended (V+Q+M)":     {"Value": 1.0, "Quality": 1.0, "Momentum": 1.0},
    "Value / Quality":     {"Value": 1.0, "Quality": 1.0},
    "Growth / Asymmetric": {"Growth": 1.5, "Quality": 0.5, "Momentum": 0.5},
    "Momentum":            {"Momentum": 1.0, "Quality": 0.5, "Value": 0.5},
    "Safety / Quality":    {"Safety": 1.0, "Quality": 1.0, "Value": 0.5},
}
DEFAULT_LENS = "Blended (V+Q+M)"

# Rank within sector only when there are enough peers; else fall back to universe.
MIN_SECTOR_PEERS = 5


# ---------------------------------------------------------------------------
# Low-level fetch helpers
# ---------------------------------------------------------------------------
def _looks_rate_limited(text: str) -> bool:
    t = (text or "").lower()
    return ("limit reach" in t or "upgrade your plan" in t
            or "too many requests" in t or "bandwidth" in t)


def _get(url: str, params: dict, symbol: str, what: str):
    """GET one FMP endpoint. Returns parsed JSON or None on a non-fatal miss.
    Raises FMPRateLimitError when the quota is exhausted."""
    try:
        r = requests.get(url, params=params, timeout=_TIMEOUT)
    except Exception as e:  # noqa: BLE001
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
    for k in keys:
        if k in record and record[k] is not None:
            try:
                return float(record[k])
            except (TypeError, ValueError):
                continue
    return None


def _first_present(record, candidates, metric, symbol, warn=True):
    val = _num(record, *candidates)
    if val is None and warn:
        logger.info("%s: no field mapped for '%s' (tried %s)", symbol, metric, candidates)
    return val


# ---------------------------------------------------------------------------
# Public: fetch one ticker's fundamentals (2 network calls)
# ---------------------------------------------------------------------------
def fetch_fundamentals(symbol: str, api_key: str, market_cap: Optional[float] = None) -> dict:
    """Return a dict keyed by METRIC_KEYS (values may be None). mom_* are filled
    later by app.py from the quote. Raises FMPRateLimitError if quota is spent."""
    out: dict = {k: None for k in METRIC_KEYS}

    # 1) key-metrics-ttm — ROIC, FCF yield, earnings yield.
    km = _first_record(_get(f"{FMP_BASE}/key-metrics-ttm",
                            {"symbol": symbol, "apikey": api_key}, symbol, "key-metrics-ttm"))
    if km:
        out["roic"]           = _first_present(km, _ROIC_KEYS, "roic", symbol)
        out["fcf_yield"]      = _first_present(km, _FCFY_KEYS, "fcf_yield", symbol)
        out["earnings_yield"] = _first_present(km, _EY_KEYS, "earnings_yield", symbol, warn=False)
        out["fcf_margin"]     = _first_present(km, _FCFM_KEYS, "fcf_margin", symbol, warn=False)

    # 2) income-statement (annual, last 2) — margins, growth, safety.
    inc = _get(f"{FMP_BASE}/income-statement",
               {"symbol": symbol, "period": "annual", "limit": 2, "apikey": api_key},
               symbol, "income-statement")
    rows = inc if isinstance(inc, list) else ([inc] if isinstance(inc, dict) and inc else [])
    if rows:
        cur = rows[0]
        rev = _num(cur, "revenue"); gp = _num(cur, "grossProfit")
        opi = _num(cur, "operatingIncome"); ni = _num(cur, "netIncome")
        if rev and rev != 0:
            if gp is not None:  out["gross_margin"] = gp / rev
            if opi is not None: out["operating_margin"] = opi / rev
            if ni is not None:  out["net_margin"] = ni / rev

        # prior year (for growth + margin-trend safety checks)
        gm_prev = om_prev = None
        prev_rev = None
        if len(rows) >= 2:
            prev = rows[1]
            prev_rev = _num(prev, "revenue")
            if prev_rev and prev_rev != 0:
                out["rev_growth"] = (rev - prev_rev) / abs(prev_rev) if rev is not None else None
                pgp = _num(prev, "grossProfit"); popi = _num(prev, "operatingIncome")
                if pgp is not None:  gm_prev = pgp / prev_rev
                if popi is not None: om_prev = popi / prev_rev

        # earnings-yield fallback: net income / market cap
        if out["earnings_yield"] is None and ni is not None and market_cap:
            out["earnings_yield"] = ni / market_cap

        # Safety = fraction of Piotroski-style health checks passed (free).
        checks = []
        if ni is not None:                       checks.append(ni > 0)
        if out["operating_margin"] is not None:  checks.append(out["operating_margin"] > 0)
        if out["fcf_yield"] is not None:         checks.append(out["fcf_yield"] > 0)
        if out["rev_growth"] is not None:        checks.append(out["rev_growth"] > 0)
        if out["gross_margin"] is not None and gm_prev is not None:
            checks.append(out["gross_margin"] >= gm_prev)
        if out["operating_margin"] is not None and om_prev is not None:
            checks.append(out["operating_margin"] >= om_prev)
        if checks:
            out["safety"] = sum(1 for c in checks if c) / len(checks)

    # Rule of 40 = revenue growth % + (FCF margin if present, else operating margin) %.
    if out["rev_growth"] is not None:
        margin = out["fcf_margin"] if out["fcf_margin"] is not None else out["operating_margin"]
        if margin is not None:
            out["rule_of_40"] = (out["rev_growth"] + margin) * 100.0

    return out


def has_any(fund: dict) -> bool:
    """True if at least one metric was populated (used to decide whether to cache)."""
    return any(fund.get(k) is not None for k in METRIC_KEYS)


# ---------------------------------------------------------------------------
# Scoring — sector-neutral percentile ranks -> family sub-scores -> lens blend
# ---------------------------------------------------------------------------
def _percentile_ranks(values: dict[str, float], higher_better: bool) -> dict[str, float]:
    """ticker->raw value (no None) mapped to ticker->percentile in [0,1]."""
    n = len(values)
    if n == 0:
        return {}
    if n == 1:
        return {t: 1.0 for t in values}
    ranks = {}
    for i, (t, _) in enumerate(sorted(values.items(), key=lambda kv: kv[1], reverse=higher_better)):
        ranks[t] = 1.0 - (i / (n - 1))
    return ranks


def _sector_percentiles(values: dict[str, Optional[float]], sectors: dict[str, str],
                        higher_better: bool, min_peers: int = MIN_SECTOR_PEERS) -> dict[str, float]:
    """Percentile-rank within each sector; fall back to universe rank for tickers
    whose sector has fewer than ``min_peers`` rated names (too few to rank fairly)."""
    present = {t: v for t, v in values.items() if v is not None}
    if not present:
        return {}
    universe = _percentile_ranks(present, higher_better)

    by_sector: dict[str, dict[str, float]] = defaultdict(dict)
    for t, v in present.items():
        by_sector[sectors.get(t, "?")][t] = v

    out: dict[str, float] = {}
    for sec, vals in by_sector.items():
        if len(vals) >= min_peers:
            out.update(_percentile_ranks(vals, higher_better))
        else:
            for t in vals:
                out[t] = universe[t]
    return out


def compute_family_scores(metrics_by_ticker: dict[str, dict],
                          sectors: dict[str, str]) -> dict[str, dict]:
    """Return {ticker: {family: {"score": 0..1|None, "coverage": int, "n": int}}}.

    Each metric is sector-neutral percentile-ranked across the universe, then
    averaged (equal weight) into its family sub-score, renormalized over only the
    metrics that ticker actually has.
    """
    tickers = list(metrics_by_ticker.keys())

    metric_pct: dict[tuple, dict[str, float]] = {}
    for fam, metrics in FAMILIES.items():
        for m, higher_better in metrics.items():
            raw = {t: (metrics_by_ticker.get(t) or {}).get(m) for t in tickers}
            metric_pct[(fam, m)] = _sector_percentiles(raw, sectors, higher_better)

    results: dict[str, dict] = {}
    for t in tickers:
        fam_scores = {}
        for fam, metrics in FAMILIES.items():
            ps = [metric_pct[(fam, m)].get(t) for m in metrics]
            ps = [p for p in ps if p is not None]
            fam_scores[fam] = {
                "score": (sum(ps) / len(ps)) if ps else None,
                "coverage": len(ps),
                "n": len(metrics),
            }
        results[t] = fam_scores
    return results


def composite_for_lens(fam_scores: dict[str, dict], weights: dict[str, float]):
    """Blend family sub-scores into one 0..1 composite for a lens.
    Returns (score|None, families_present, families_in_lens)."""
    num = den = 0.0
    present = 0
    for fam, w in weights.items():
        s = (fam_scores.get(fam) or {}).get("score")
        if s is not None:
            num += w * s
            den += w
            present += 1
    return ((num / den) if den > 0 else None, present, len(weights))


# ---------------------------------------------------------------------------
# Target sanity guard — catches the split/stale mismatch (e.g. NFLX 358 vs ~83)
# ---------------------------------------------------------------------------
def target_is_sane(target: Optional[float], lo: Optional[float], hi: Optional[float],
                   hi_mult: float = 2.0, lo_mult: float = 0.3) -> bool:
    if target is None or lo is None or hi is None or hi <= 0:
        return True
    if target > hi * hi_mult:
        return False
    if target < lo * lo_mult:
        return False
    return True
