"""
tenx_radar.py — "10x Radar": find names whose REVENUES ARE EXPLODING.

What it looks for (the classic profile of past 10-baggers — NVDA '23, SMCI '23,
MU/SNDK in memory upcycles): quarterly revenue growth that is BIG and GETTING
BIGGER, margins inflecting up at the same time (operating leverage), a market
cap small enough that 10x is arithmetically plausible, and price momentum
confirming the market has started to notice.

Data: QUARTERLY income statements from Yahoo Finance (keyless, free — zero FMP
quota used), fetched per ticker and disk-cached by app.py. Annual data is far
too slow to catch these inflections; a memory-cycle turn shows up in quarterly
prints 6–12 months before the annual numbers move.

Pure module: no Streamlit, no threading, no caching here.

NOT A PREDICTION. Most hypergrowth names do not 10x; many round-trip. The score
measures how closely TODAY'S REPORTED numbers resemble the historical profile —
it knows nothing about the future.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("tenx_radar")

try:
    import yfinance as yf
    HAVE_YF = True
except ImportError:  # pragma: no cover
    yf = None
    HAVE_YF = False

# Weights for the five 10x components (renormalized over what's available,
# then shrunk toward neutral for missing coverage — same idea as fundamentals.py).
TENX_WEIGHTS = {
    "explosion": 0.35,   # latest-quarter YoY revenue growth (the core signal)
    "accel":     0.25,   # is that growth rate itself rising?
    "leverage":  0.15,   # gross/operating margin inflecting up (operating leverage)
    "headroom":  0.15,   # market-cap room to 10x (mega-caps mathematically can't)
    "momentum":  0.10,   # price confirming (12-1m return, else 52w position)
}

# Metric keys this module produces per ticker.
TENX_KEYS = (
    "q_rev_yoy", "q_rev_yoy_prev", "rev_accel", "q_rev_qoq", "seq_accel",
    "gm_delta", "om_delta", "ttm_rev", "n_quarters", "latest_q",
)

SMALL_BASE_TTM = 200e6        # under ~$200M TTM revenue -> flag tiny base
MIN_BASE_QREV = 5e6           # ignore YoY off a near-zero base (garbage %)


def _f(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # NaN -> None


# ---------------------------------------------------------------------------
# Quarterly statement fetch (Yahoo — keyless, no FMP quota)
# ---------------------------------------------------------------------------
def _rows_from_df(df, *names) -> list:
    """All quarterly values for the first matching row label, most-recent-first.
    Case/space-insensitive matching, same convention as yahoo_fallback."""
    if df is None or getattr(df, "empty", True):
        return []
    norm = {str(idx).replace(" ", "").lower(): idx for idx in df.index}
    for name in names:
        idx = norm.get(name.replace(" ", "").lower())
        if idx is None:
            continue
        series = df.loc[idx]
        # Columns are period-end Timestamps; ensure most-recent-first.
        try:
            series = series[sorted(series.index, reverse=True)]
        except Exception:  # noqa: BLE001
            pass
        return [_f(v) for v in series.tolist()]
    return []


def fetch_quarterly_yahoo(symbol: str) -> Optional[dict]:
    """Return {"rev": [...], "gp": [...], "opi": [...], "dates": [...]} with
    quarterly values most-recent-first, or None on failure. Yahoo returns
    ~5-6 quarters — enough for a YoY comparison and one acceleration read."""
    if not HAVE_YF:
        return None
    try:
        inc = yf.Ticker(symbol).quarterly_income_stmt
    except Exception as e:  # noqa: BLE001
        logger.warning("%s: yahoo quarterly stmt failed: %s", symbol, str(e)[:120])
        return None
    if inc is None or getattr(inc, "empty", True):
        return None
    try:
        dates = [str(c)[:10] for c in sorted(inc.columns, reverse=True)]
    except Exception:  # noqa: BLE001
        dates = []
    rev = _rows_from_df(inc, "Total Revenue", "TotalRevenue", "Operating Revenue")
    if not rev or all(v is None for v in rev):
        return None
    return {
        "rev": rev,
        "gp": _rows_from_df(inc, "Gross Profit"),
        "opi": _rows_from_df(inc, "Operating Income", "EBIT"),
        "dates": dates,
    }


def fetch_next_earnings(symbol: str) -> Optional[str]:
    """Next scheduled earnings date as 'YYYY-MM-DD' (Yahoo calendar, keyless),
    or the most recent past date if nothing upcoming, or None. Never raises."""
    if not HAVE_YF:
        return None
    try:
        cal = yf.Ticker(symbol).calendar
    except Exception:  # noqa: BLE001
        return None
    dates = []
    try:
        if isinstance(cal, dict):
            dates = list(cal.get("Earnings Date") or [])
        elif cal is not None and hasattr(cal, "loc"):   # older yfinance: DataFrame
            try:
                dates = list(cal.loc["Earnings Date"])
            except Exception:  # noqa: BLE001
                dates = []
    except Exception:  # noqa: BLE001
        return None
    iso = sorted(str(d)[:10] for d in dates if d is not None)
    if not iso:
        return None
    from datetime import date
    today = date.today().isoformat()
    future = [d for d in iso if d >= today]
    return future[0] if future else iso[-1]


def fetch_last_earnings_surprise(symbol: str) -> Optional[float]:
    """EPS surprise (%) of the most recent REPORTED quarter, from Yahoo's
    earnings history. Negative = the company missed estimates. None when
    unavailable. Never raises."""
    if not HAVE_YF:
        return None
    try:
        df = yf.Ticker(symbol).get_earnings_dates(limit=8)
    except Exception:  # noqa: BLE001
        return None
    if df is None or getattr(df, "empty", True):
        return None
    try:
        for idx in df.index:                      # newest first
            rep = df.loc[idx].get("Reported EPS")
            if rep is not None and rep == rep:    # first row actually reported
                surp = df.loc[idx].get("Surprise(%)")
                return float(surp) if (surp is not None and surp == surp) else None
    except Exception:  # noqa: BLE001
        return None
    return None


# ---------------------------------------------------------------------------
# Metrics from the quarterly series
# ---------------------------------------------------------------------------
def _yoy(series: list, i: int) -> Optional[float]:
    """YoY growth of quarter i vs quarter i+4 (series most-recent-first)."""
    if len(series) < i + 5:
        return None
    now, base = series[i], series[i + 4]
    if now is None or base is None or base < MIN_BASE_QREV:
        return None
    return (now - base) / abs(base)


def _margin(rev: list, num: list, i: int) -> Optional[float]:
    if len(rev) <= i or len(num) <= i:
        return None
    r, n = rev[i], num[i]
    if r is None or n is None or r <= 0:
        return None
    return n / r


def compute_tenx_metrics(qdata: Optional[dict]) -> dict:
    """Turn the raw quarterly series into the radar's metric dict."""
    out: dict = {k: None for k in TENX_KEYS}
    if not qdata:
        return out
    rev = qdata.get("rev") or []
    gp = qdata.get("gp") or []
    opi = qdata.get("opi") or []
    dates = qdata.get("dates") or []

    out["n_quarters"] = len([v for v in rev if v is not None])
    out["latest_q"] = dates[0] if dates else None

    out["q_rev_yoy"] = _yoy(rev, 0)          # latest quarter vs same quarter last year
    out["q_rev_yoy_prev"] = _yoy(rev, 1)     # the quarter before, vs ITS year-ago
    if out["q_rev_yoy"] is not None and out["q_rev_yoy_prev"] is not None:
        out["rev_accel"] = out["q_rev_yoy"] - out["q_rev_yoy_prev"]

    # Sequential growth + acceleration — fallback when only ~5 quarters exist.
    if len(rev) >= 2 and rev[0] is not None and rev[1] and rev[1] > MIN_BASE_QREV:
        out["q_rev_qoq"] = rev[0] / rev[1] - 1.0
    if (len(rev) >= 3 and out["q_rev_qoq"] is not None
            and rev[1] is not None and rev[2] and rev[2] > MIN_BASE_QREV):
        out["seq_accel"] = out["q_rev_qoq"] - (rev[1] / rev[2] - 1.0)

    # Margin inflection: latest quarter vs the same quarter a year ago
    # (YoY comparison sidesteps seasonality).
    gm_now, gm_yr = _margin(rev, gp, 0), _margin(rev, gp, 4)
    if gm_now is not None and gm_yr is not None:
        out["gm_delta"] = gm_now - gm_yr
    om_now, om_yr = _margin(rev, opi, 0), _margin(rev, opi, 4)
    if om_now is not None and om_yr is not None:
        out["om_delta"] = om_now - om_yr

    ttm = [v for v in rev[:4] if v is not None]
    out["ttm_rev"] = sum(ttm) if len(ttm) == 4 else (sum(ttm) / len(ttm) * 4 if ttm else None)
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _headroom_from_mcap(mcap: Optional[float]) -> Optional[float]:
    """How plausible is a 10x from this size? Mega-caps get dinged: a $4T name
    10x-ing means a $40T company. Small caps have the arithmetic on their side
    (and the blow-up risk — that's what the other components + flags are for)."""
    if mcap is None or mcap <= 0:
        return None
    if mcap < 2e9:    return 1.00
    if mcap < 10e9:   return 0.90
    if mcap < 50e9:   return 0.72
    if mcap < 200e9:  return 0.50
    if mcap < 1e12:   return 0.28
    return 0.10


def tenx_score(m: dict, market_cap: Optional[float],
               mom_12_1: Optional[float] = None,
               mom_52w: Optional[float] = None) -> tuple[Optional[float], dict, list[str]]:
    """(score 0-100 | None, component subscores 0..1, tag strings).

    Components are ABSOLUTE (not peer-percentile): "exploding" means exploding,
    not "growing faster than a utility". Returns None when the core signal
    (latest-quarter YoY revenue growth) is unavailable.
    """
    yoy = m.get("q_rev_yoy")
    if yoy is None:
        return None, {}, []

    sub: dict[str, Optional[float]] = {}

    # 1) Explosion — 0% YoY -> 0, 100%+ YoY -> 1.0 (linear between).
    sub["explosion"] = _clip01(yoy / 1.00)

    # 2) Acceleration — prefer true YoY-accel (needs 6 quarters), else sequential.
    accel = m.get("rev_accel")
    if accel is not None:
        sub["accel"] = _clip01((accel + 0.05) / 0.35)   # -5ppt -> 0 · +30ppt -> 1
    elif m.get("seq_accel") is not None:
        sub["accel"] = _clip01((m["seq_accel"] + 0.05) / 0.25)
    else:
        sub["accel"] = None

    # 3) Operating leverage — margins inflecting up vs the year-ago quarter.
    lev = []
    if m.get("gm_delta") is not None:
        lev.append(_clip01((m["gm_delta"] + 0.02) / 0.10))   # +8ppt GM -> max
    if m.get("om_delta") is not None:
        lev.append(_clip01((m["om_delta"] + 0.03) / 0.15))   # +12ppt OM -> max
    sub["leverage"] = sum(lev) / len(lev) if lev else None

    # 4) Headroom — market-cap tiers.
    sub["headroom"] = _headroom_from_mcap(market_cap)

    # 5) Momentum confirmation — canonical 12-1m return, else 52w position.
    if mom_12_1 is not None:
        sub["momentum"] = _clip01((mom_12_1 + 0.25) / 1.25)  # -25% -> 0 · +100% -> 1
    elif mom_52w is not None:
        sub["momentum"] = _clip01(mom_52w)
    else:
        sub["momentum"] = None

    num = den = 0.0
    present = 0
    for k, w in TENX_WEIGHTS.items():
        v = sub.get(k)
        if v is not None:
            num += w * v
            den += w
            present += 1
    raw = num / den if den > 0 else None
    if raw is None:
        return None, sub, []
    # Shrink toward neutral for missing components (can't fluke a 100 on 2 of 5).
    cov = present / len(TENX_WEIGHTS)
    score = (0.5 + (raw - 0.5) * cov) * 100.0

    # ---- tags (shown in the table's "Signals" column) ----
    tags: list[str] = []
    if yoy >= 1.00:
        tags.append("🚀 revenue exploding (>100% YoY)")
    elif yoy >= 0.40:
        tags.append("🚀 revenue surging (>40% YoY)")
    elif yoy >= 0.20:
        tags.append("📶 fast grower (>20% YoY)")
    accel_eff = accel if accel is not None else m.get("seq_accel")
    if accel_eff is not None:
        if accel_eff >= 0.10:
            tags.append("⚡ accelerating")
        elif accel_eff <= -0.10:
            tags.append("🐌 decelerating")
    if (m.get("gm_delta") or 0) >= 0.03 or (m.get("om_delta") or 0) >= 0.05:
        tags.append("📈 margin inflection")
    if market_cap is not None and market_cap < 20e9:
        tags.append("🎯 10x headroom (<$20B)")
    if (mom_12_1 or 0) >= 0.30 or (mom_52w or 0) >= 0.80:
        tags.append("💪 momentum confirmed")
    if m.get("ttm_rev") is not None and m["ttm_rev"] < SMALL_BASE_TTM:
        tags.append("⚠️ tiny revenue base")
    if (m.get("om_delta") or 0) <= -0.05:
        tags.append("⚠️ margins compressing")

    return score, sub, tags
