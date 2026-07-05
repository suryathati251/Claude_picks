"""
fundamentals.py — reported-financials + scoring layer for the watchlist.

Data (3 calls/ticker, free-tier friendly)
-----------------------------------------
Metrics are derived from RAW statement numbers (stable field names) instead of
FMP's pre-computed ratio fields (which vary by plan and caused the old blanks):
  1. ``key-metrics-ttm``  -> ROIC, FCF yield, (earnings yield)
  2. ``income-statement`` (annual, last 2) -> revenue growth, gross/operating/net
     margins, P/S, PEG, EPS growth, interest coverage, AND a Piotroski-style
     SAFETY score — all from data we already fetch.
  3. ``balance-sheet-statement`` (annual, latest) -> debt/equity, net-debt/EBITDA.
When FMP is rate-limited, yahoo_fallback.py fills the same metric keys for free.
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
import statistics
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
    "mom_52w", "mom_ma200", "mom_12_1", "safety",
    # valuation / leverage additions (QARP)
    "ps_ratio", "peg", "eps_growth", "ev_ebit",
    "debt_equity", "net_debt_ebitda", "interest_coverage",
    # quality: Novy-Marx gross profitability (gross profit / total assets)
    "gross_profitability",
    # moat: durability signals over ~5 years (no extra API calls)
    "gross_margin_avg", "margin_stability", "growth_consistency",
)

# Candidate FMP field names for the few metrics we still read pre-computed.
_ROIC_KEYS = ["returnOnInvestedCapitalTTM", "roicTTM", "returnOnInvestedCapital",
              "returnOnCapitalEmployedTTM", "returnOnCapitalEmployed"]
_FCFY_KEYS = ["freeCashFlowYieldTTM", "freeCashFlowYield"]
_EY_KEYS   = ["earningsYieldTTM", "earningsYield"]
_FCFM_KEYS = ["freeCashFlowMarginTTM", "freeCashFlowMargin"]


# ---------------------------------------------------------------------------
# Factor families -> CONCEPT groups -> metrics   (higher_better per metric)
# ---------------------------------------------------------------------------
# Why two levels: averaging correlated metrics flat double-counts them. e.g.
# gross + operating margin move together, so flat-averaging Quality would give
# "margins" 2/3 weight and dilute ROIC. Grouping correlated metrics into one
# CONCEPT and averaging concepts equally fixes that — adding a third margin
# metric no longer drowns the independent signals. higher_better=False means a
# LOW raw value should score high (P/S, PEG, leverage).
FAMILIES: dict[str, dict[str, dict[str, bool]]] = {
    "Value": {
        "earnings":   {"earnings_yield": True, "ev_ebit": False},  # earnings cheapness: equity yield + EV/EBIT
        "cashflow":   {"fcf_yield": True},                          # cash-flow cheapness
        "sales":      {"ps_ratio": False},                          # sales cheapness
        "growth_adj": {"peg": False},                               # growth-adjusted cheapness
    },
    "Quality": {
        "returns": {"roic": True, "gross_profitability": True},      # returns on capital/assets
        "margins": {"gross_margin": True, "operating_margin": True}, # correlated -> one concept
    },
    "Growth": {
        "topline":    {"rev_growth": True},
        "bottomline": {"eps_growth": True},
        "rule40":     {"rule_of_40": True},
    },
    "Momentum": {
        "trend_12_1": {"mom_12_1": True},                  # canonical 12-1m return (primary)
        "position":   {"mom_52w": True, "mom_ma200": True},# crude price proxies (fallback)
    },
    "Safety": {
        "health":   {"safety": True},                      # Piotroski-style checks
        "leverage": {"debt_equity": False, "net_debt_ebitda": False},  # both leverage -> one concept
        "coverage": {"interest_coverage": True},
    },
    # Economic MOAT = the quantitative fingerprint of a durable competitive edge:
    # sustained high returns on capital + durable pricing power + low volatility.
    # (roic is shared with Quality on purpose — it IS the core moat signal — but the
    # multi-year LEVEL + STABILITY metrics are what make Moat distinct from Quality.)
    "Moat": {
        "returns_level": {"roic": True},                            # sustained high returns on capital
        "pricing_power": {"gross_margin_avg": True},                # durable 5-yr gross margin
        "durability":    {"margin_stability": True,                 # low margin volatility
                          "growth_consistency": True},              # consistent revenue growth
    },
}

# Lens = weights over families. All sub-scores always display; the lens only sets
# the headline composite + sort.
#
# Default = QARP ("Quality At a Reasonable Price") — the best simple, evidence-
# backed recipe for undervalued high-quality stocks. It is essentially
# Greenblatt's Magic Formula (cheap on earnings/FCF yield × high ROIC) widened
# with a balance-sheet Safety leg and a small Growth leg, then red-flag-adjusted
# (see compute_flags): falling revenue, heavy debt, negative FCF subtract points;
# PEG < 1, net cash, ROIC > 20% add points. Cheapness alone finds value traps;
# quality alone overpays — the intersection is where the edge historically lives.
# Moat is folded into every lens (durability of returns is a positive everywhere):
# weight 0.5 in the quality/value/safety lenses, a lighter 0.25 in growth/momentum
# (young growth/momentum names often haven't established a moat yet), and 1.0 in the
# dedicated Wide-Moat lens.
LENSES: dict[str, dict[str, float]] = {
    "QARP (Underv. Quality)": {"Value": 1.0, "Quality": 1.0, "Safety": 0.75, "Growth": 0.25, "Moat": 0.5},
    "Blended (V+Q+M)":     {"Value": 1.0, "Quality": 1.0, "Momentum": 1.0, "Moat": 0.5},
    "Value / Quality":     {"Value": 1.0, "Quality": 1.0, "Moat": 0.5},
    "Growth / Asymmetric": {"Growth": 1.5, "Quality": 0.5, "Momentum": 0.5, "Moat": 0.25},
    "Momentum":            {"Momentum": 1.0, "Quality": 0.5, "Value": 0.5, "Moat": 0.25},
    "Safety / Quality":    {"Safety": 1.0, "Quality": 1.0, "Value": 0.5, "Moat": 0.5},
    "Wide-Moat Compounders": {"Moat": 1.0, "Quality": 0.5, "Value": 0.5},
}
DEFAULT_LENS = "QARP (Underv. Quality)"

# Rank within sector only when there are enough peers; else fall back to universe.
MIN_SECTOR_PEERS = 5

# A family sub-score is shrunk toward neutral (0.5) in proportion to how many of
# its CONCEPTS are missing, so a stock with 1-of-3 concepts can't fluke a 100.
# Safety gate: a gentle multiplier on the composite from the ABSOLUTE Piotroski
# health fraction — fragile names can't top any lens, even momentum/growth.
SAFETY_GATE_FLOOR = 0.85   # worst-health name keeps 85% of its score


def safety_gate(safety_raw: Optional[float]) -> float:
    """Map an absolute health fraction in [0,1] to a composite multiplier in
    [SAFETY_GATE_FLOOR, 1.0]. None (unknown) -> 1.0 (no penalty for missing data)."""
    if safety_raw is None:
        return 1.0
    s = max(0.0, min(1.0, safety_raw))
    return SAFETY_GATE_FLOOR + (1.0 - SAFETY_GATE_FLOOR) * s


def edge_net_debt_ebitda(total_debt: Optional[float], cash: Optional[float],
                         ebitda: Optional[float]) -> Optional[float]:
    """Net-debt / EBITDA with edge handling so risky names don't escape the
    Safety screen by having undefined EBITDA. EBITDA<=0 + net debt -> worst (99);
    net cash -> good (-1)."""
    if total_debt is None:
        return None
    net_debt = total_debt - (cash or 0.0)
    if ebitda and ebitda > 0:
        return net_debt / ebitda
    return -1.0 if net_debt <= 0 else 99.0


def edge_interest_coverage(opi: Optional[float], int_exp: Optional[float],
                           total_debt: Optional[float]) -> Optional[float]:
    """Interest coverage; a debt-free firm gets a high (good) sentinel instead of
    None, so it earns credit for having no interest burden."""
    if int_exp and int_exp > 0:
        return (opi / int_exp) if opi is not None else None
    if (total_debt or 0.0) <= 0.0:
        return 50.0   # effectively unleveraged -> excellent coverage
    return None       # has debt but interest figure missing -> leave unranked


def edge_ev_ebit(market_cap: Optional[float], total_debt: Optional[float],
                 cash: Optional[float], ebit: Optional[float]) -> Optional[float]:
    """EV/EBIT (lower = cheaper). EV = market cap + debt − cash. Negative/zero EBIT
    (unprofitable) -> a high sentinel so it ranks as 'expensive', not blank."""
    if market_cap is None or ebit is None:
        return None
    ev = market_cap + (total_debt or 0.0) - (cash or 0.0)
    if ebit <= 0:
        return 999.0          # unprofitable -> worst (expensive)
    if ev <= 0:
        return None           # net-cash quirk (EV<=0) -> can't rank meaningfully
    return ev / ebit


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


def moat_metrics(gross_margins: list, operating_margins: list, revenues: list):
    """Durability signals for the Moat family, from several years of statements.

    Returns (gross_margin_avg, margin_stability, growth_consistency):
    * gross_margin_avg  — mean gross margin (pricing-power LEVEL).
    * margin_stability  — 0..1, 1 = rock-steady margins. = 1 − mean coefficient-of-
                          variation of gross & operating margins (≥3 yrs needed).
    * growth_consistency— 0..1, fraction of year-over-year revenue changes that were
                          positive (durable demand; ≥3 yrs needed).
    None when there isn't enough history to judge.
    """
    gm = [x for x in gross_margins if x is not None]
    om = [x for x in operating_margins if x is not None]
    revs = [x for x in revenues if x is not None]

    gross_margin_avg = (sum(gm) / len(gm)) if gm else None

    covs = []
    for series in (gm, om):
        if len(series) >= 3:
            m = sum(series) / len(series)
            if m != 0:
                covs.append(statistics.pstdev(series) / abs(m))
    margin_stability = max(0.0, 1.0 - sum(covs) / len(covs)) if covs else None

    growth_consistency = None
    if len(revs) >= 3:                       # revs are most-recent-first
        deltas = [revs[i] - revs[i + 1] for i in range(len(revs) - 1)]
        growth_consistency = sum(1 for d in deltas if d > 0) / len(deltas)

    return gross_margin_avg, margin_stability, growth_consistency


# Number of FMP calls fetch_fundamentals makes per ticker (used for budgeting).
CALLS_PER_TICKER = 3


# ---------------------------------------------------------------------------
# Public: fetch one ticker's fundamentals (3 network calls)
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

    # 2) income-statement (annual, last 5) — margins, growth, safety + moat durability.
    #    5 years (vs 2) costs NO extra call and powers the Moat stability metrics.
    inc = _get(f"{FMP_BASE}/income-statement",
               {"symbol": symbol, "period": "annual", "limit": 5, "apikey": api_key},
               symbol, "income-statement")
    rows = inc if isinstance(inc, list) else ([inc] if isinstance(inc, dict) and inc else [])
    ebitda = None
    gp = None
    opi = None
    if rows:
        cur = rows[0]
        rev = _num(cur, "revenue"); gp = _num(cur, "grossProfit")
        opi = _num(cur, "operatingIncome"); ni = _num(cur, "netIncome")
        ebitda = _num(cur, "ebitda")
        int_exp = _num(cur, "interestExpense")
        if rev and rev != 0:
            if gp is not None:  out["gross_margin"] = gp / rev
            if opi is not None: out["operating_margin"] = opi / rev
            if ni is not None:  out["net_margin"] = ni / rev

        # P/S — market cap over trailing revenue (no extra call needed).
        if market_cap and rev and rev > 0:
            out["ps_ratio"] = market_cap / rev

        # Interest coverage — how many times operating income covers interest.
        if opi is not None and int_exp and int_exp > 0:
            out["interest_coverage"] = opi / int_exp

        # prior year (for growth + margin-trend safety checks)
        gm_prev = om_prev = None
        prev_rev = None
        if len(rows) >= 2:
            prev = rows[1]
            prev_rev = _num(prev, "revenue")
            prev_ni = _num(prev, "netIncome")
            if prev_rev and prev_rev != 0:
                out["rev_growth"] = (rev - prev_rev) / abs(prev_rev) if rev is not None else None
                pgp = _num(prev, "grossProfit"); popi = _num(prev, "operatingIncome")
                if pgp is not None:  gm_prev = pgp / prev_rev
                if popi is not None: om_prev = popi / prev_rev
            # EPS-growth proxy: net-income growth (only meaningful off a positive base)
            if ni is not None and prev_ni and prev_ni > 0:
                out["eps_growth"] = (ni - prev_ni) / prev_ni

        # earnings-yield fallback: net income / market cap
        if out["earnings_yield"] is None and ni is not None and market_cap:
            out["earnings_yield"] = ni / market_cap

        # PEG — P/E over EPS-growth%. Only meaningful when BOTH are positive.
        if (market_cap and ni and ni > 0
                and out["eps_growth"] is not None and out["eps_growth"] > 0):
            pe = market_cap / ni
            out["peg"] = pe / (out["eps_growth"] * 100.0)

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

        # Moat durability — multi-year margins + revenue from the SAME 5-yr call.
        yr_gm, yr_om, yr_rev = [], [], []
        for r in rows:
            rv = _num(r, "revenue")
            yr_rev.append(rv)
            if rv and rv != 0:
                g_ = _num(r, "grossProfit"); o_ = _num(r, "operatingIncome")
                yr_gm.append(g_ / rv if g_ is not None else None)
                yr_om.append(o_ / rv if o_ is not None else None)
        out["gross_margin_avg"], out["margin_stability"], out["growth_consistency"] = \
            moat_metrics(yr_gm, yr_om, yr_rev)

    # 3) balance-sheet (annual, latest) — leverage for the Safety family & flags.
    bs = _first_record(_get(f"{FMP_BASE}/balance-sheet-statement",
                            {"symbol": symbol, "period": "annual", "limit": 1,
                             "apikey": api_key}, symbol, "balance-sheet"))
    if bs:
        total_debt = _num(bs, "totalDebt")
        if total_debt is None:
            std = _num(bs, "shortTermDebt"); ltd = _num(bs, "longTermDebt")
            if std is not None or ltd is not None:
                total_debt = (std or 0.0) + (ltd or 0.0)
        cash = _num(bs, "cashAndCashEquivalents", "cashAndShortTermInvestments") or 0.0
        equity = _num(bs, "totalStockholdersEquity", "totalEquity")
        total_assets = _num(bs, "totalAssets")
        if total_debt is not None and equity and equity > 0:
            out["debt_equity"] = total_debt / equity
        # Net-debt/EBITDA with edge handling (net cash -> good; EBITDA<=0 -> worst).
        out["net_debt_ebitda"] = edge_net_debt_ebitda(total_debt, cash, ebitda)
        # Debt-free firms get credit for having no interest burden.
        if out["interest_coverage"] is None and total_debt is not None and total_debt <= 0:
            out["interest_coverage"] = 50.0
        # Gross profitability (Novy-Marx): gross profit / total assets.
        if gp is not None and total_assets and total_assets > 0:
            out["gross_profitability"] = gp / total_assets
        # EV/EBIT — capital-structure-aware cheapness (EV = mktcap + debt − cash).
        out["ev_ebit"] = edge_ev_ebit(market_cap, total_debt, cash, opi)

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
    """Return {ticker: {family: {"score", "raw", "coverage", "n", "metrics"}}}.

    Pipeline per family:
      1. Each metric is sector-neutral percentile-ranked across the universe.
      2. Metrics are averaged WITHIN their concept (so correlated metrics count
         once), then concepts are averaged equally -> raw family score.
      3. The raw score is shrunk toward 0.5 in proportion to missing concepts, so
         a stock with thin coverage can't fluke an extreme score.
    ``coverage``/``n`` count CONCEPTS present/total; ``metrics`` counts raw metrics.
    """
    tickers = list(metrics_by_ticker.keys())

    # Rank every metric once (sector-neutral).
    metric_pct: dict[tuple, dict[str, float]] = {}
    for fam, concepts in FAMILIES.items():
        for concept, metrics in concepts.items():
            for m, higher_better in metrics.items():
                raw = {t: (metrics_by_ticker.get(t) or {}).get(m) for t in tickers}
                metric_pct[(fam, concept, m)] = _sector_percentiles(raw, sectors, higher_better)

    results: dict[str, dict] = {}
    for t in tickers:
        fam_scores = {}
        for fam, concepts in FAMILIES.items():
            concept_scores, n_metrics = [], 0
            for concept, metrics in concepts.items():
                ps = [metric_pct[(fam, concept, m)].get(t) for m in metrics]
                ps = [p for p in ps if p is not None]
                if ps:
                    concept_scores.append(sum(ps) / len(ps))
                    n_metrics += len(ps)
            n_concepts = len(concepts)
            if concept_scores:
                raw = sum(concept_scores) / len(concept_scores)
                cov_frac = len(concept_scores) / n_concepts
                score = 0.5 + (raw - 0.5) * cov_frac          # shrink toward neutral
            else:
                raw = score = None
            fam_scores[fam] = {
                "score": score, "raw": raw,
                "coverage": len(concept_scores), "n": n_concepts, "metrics": n_metrics,
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
# Red / green flags — absolute checks layered ON TOP of the relative ranks.
# ---------------------------------------------------------------------------
# Percentile ranks only say "cheaper/safer than peers"; flags catch things that
# are bad (or great) in absolute terms regardless of peers. The returned delta
# is added to the 0–100 composite score; flags are shown in the table.
FLAG_PENALTY_CAP = -25.0
FLAG_BONUS_CAP = 15.0


def compute_flags(fund: dict) -> tuple[float, list[str]]:
    """Return (score_delta, flag_strings) for one ticker's metrics dict.
    None metrics are skipped — no penalty for missing data."""
    if not fund:
        return 0.0, []
    delta = 0.0
    flags: list[str] = []

    def m(key):
        return fund.get(key)

    # Consistent traffic-light scheme: 🔴 = risk (subtracts points), 🟢 = positive
    # (adds points). Each label is plain-English with the metric in parentheses.

    # ---- 🔴 risk flags ----
    rg = m("rev_growth")
    if rg is not None and rg < 0:
        if rg < -0.10:
            delta -= 12; flags.append("🔴 revenue falling >10%")
        else:
            delta -= 6; flags.append("🔴 revenue declining")

    de = m("debt_equity")
    nde = m("net_debt_ebitda")
    if de is not None and de > 2.0:
        delta -= 8; flags.append("🔴 high debt (D/E > 2)")
    elif de is not None and de > 1.0:
        delta -= 4; flags.append("🔴 elevated debt (D/E > 1)")
    if nde is not None and nde > 4.0:
        delta -= 8; flags.append("🔴 heavy leverage (net debt > 4× EBITDA)")
    elif nde is not None and nde > 3.0:
        delta -= 4; flags.append("🔴 leveraged (net debt > 3× EBITDA)")

    ic = m("interest_coverage")
    if ic is not None and ic < 2.0:
        delta -= 6; flags.append("🔴 thin interest coverage (< 2×)")

    fcfy = m("fcf_yield")
    if fcfy is not None and fcfy < 0:
        delta -= 5; flags.append("🔴 burning cash (negative FCF)")
    nm = m("net_margin")
    if nm is not None and nm < 0:
        delta -= 4; flags.append("🔴 unprofitable (negative margin)")

    # extra danger when leverage meets shrinking revenue (the classic value trap)
    if (rg is not None and rg < 0) and ((de or 0) > 1.0 or (nde or 0) > 3.0):
        delta -= 5; flags.append("🔴 value-trap risk (high debt + falling revenue)")

    # ---- 🟢 positive flags ----
    peg = m("peg")
    if peg is not None and 0 < peg < 1.0:
        delta += 6; flags.append("🟢 cheap vs growth (PEG < 1)")
    ps = m("ps_ratio")
    if ps is not None and ps < 2.0 and (rg or 0) > 0.10:
        delta += 4; flags.append("🟢 cheap on sales, still growing (P/S < 2)")
    roic = m("roic")
    if roic is not None and roic > 0.20:
        delta += 5; flags.append("🟢 high returns on capital (ROIC > 20%)")
    if ((nde is not None and nde < 0) or (de is not None and de < 0.10)):
        delta += 4; flags.append("🟢 strong balance sheet (net cash)")

    return max(FLAG_PENALTY_CAP, min(FLAG_BONUS_CAP, delta)), flags


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
