"""
entry_meter.py — "Is this a good time to put money into the S&P 500?"

A fear/greed ENTRY meter for index investing: the user's rule is "invest during
fear/lows, not during greed/highs". This module turns that into a 0–100 gauge
(CNN convention: 0 = extreme fear, 100 = extreme greed) built from seven
observable, free signals — then maps the reading to a *deployment stance*
(how aggressively to add), never a prediction.

Signals (all normalized to the greed scale, 100 = greedy/expensive):
  1. Drawdown from 52-week high   — at the high = 100 · −20% or worse = 0
  2. VIX level                    — ≤12 (complacent) = 100 · ≥40 (panic) = 0
  3. VIX vs its own 1-year range  — volatility percentile, inverted
  4. Trend stretch vs 200-day MA  — +10% above = 100 · −10% below = 0
  5. 52-week range position       — at the low = 0 · at the high = 100
  6. RSI(14) of the index         — ≤30 oversold = 0 · ≥70 overbought = 100
  7. Breadth                      — % of tracked tickers above their 200-day MA

Data: items 1/2/4/5 come free from the FMP quotes app.py already fetches
(market_risk.py); 3/6 need one keyless Yahoo history download (cached);
7 is computed from the watchlist quotes already in memory. Zero extra FMP quota.

HONEST FRAMING (also shown in the UI): market timing does not reliably work,
and historically a lump sum invested immediately beats waiting for a dip about
two-thirds of the time — because markets usually grind higher. What fear/greed
readings CAN do is discipline the *price* you pay when you were going to invest
anyway: deploy faster into fear, slower into greed, and never let the meter
talk you out of a fixed DCA schedule. Extreme fear can always get more extreme.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("entry_meter")

try:
    import yfinance as yf
    HAVE_YF = True
except ImportError:  # pragma: no cover
    yf = None
    HAVE_YF = False


def _f(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ---------------------------------------------------------------------------
# History-based inputs (one keyless Yahoo call, cached by app.py)
# ---------------------------------------------------------------------------
def rsi14(closes: list) -> Optional[float]:
    """Wilder's RSI(14) from a list of closes (oldest-first)."""
    closes = [c for c in closes if c is not None]
    if len(closes) < 15:
        return None
    deltas = [closes[i + 1] - closes[i] for i in range(len(closes) - 1)]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]
    avg_g = sum(gains[:14]) / 14.0
    avg_l = sum(losses[:14]) / 14.0
    for g, l in zip(gains[14:], losses[14:]):
        avg_g = (avg_g * 13 + g) / 14.0
        avg_l = (avg_l * 13 + l) / 14.0
    if avg_l == 0:
        return 100.0
    rs = avg_g / avg_l
    return 100.0 - 100.0 / (1.0 + rs)


def fetch_entry_history_yahoo() -> dict:
    """One bulk download: ^GSPC + ^VIX daily closes (1y). Returns
    {"spx_rsi14": float|None, "vix_pctile_1y": float|None} — best-effort,
    never raises. vix_pctile_1y is 0..1 (1 = VIX at its 1-year high = max fear)."""
    out = {"spx_rsi14": None, "vix_pctile_1y": None}
    if not HAVE_YF:
        return out
    try:
        hist = yf.download(tickers="^GSPC ^VIX", period="1y", interval="1d",
                           group_by="ticker", auto_adjust=True, progress=False, threads=False)
    except Exception as e:  # noqa: BLE001
        logger.warning("entry-meter history download failed: %s", str(e)[:160])
        return out
    try:
        spx = hist["^GSPC"]["Close"].dropna()
        if len(spx) >= 15:
            out["spx_rsi14"] = rsi14([_f(v) for v in spx.tolist()])
    except Exception:  # noqa: BLE001
        pass
    try:
        vix = hist["^VIX"]["Close"].dropna()
        if len(vix) >= 60:
            cur = _f(vix.iloc[-1])
            vals = [_f(v) for v in vix.tolist() if _f(v) is not None]
            if cur is not None and vals:
                below = sum(1 for v in vals if v <= cur)
                out["vix_pctile_1y"] = below / len(vals)
    except Exception:  # noqa: BLE001
        pass
    return out


# ---------------------------------------------------------------------------
# The meter
# ---------------------------------------------------------------------------
ZONES = [
    (25, "🟢 Extreme fear",
     "Historically the best accumulation zones — big drawdowns and panic volatility. "
     "If you have cash earmarked for the index, this is when a pre-planned, tranched "
     "deploy (e.g. thirds over a few weeks) has paid best. Expect it to FEEL terrible — "
     "that's the point. It can still go lower; never deploy money you may need soon."),
    (45, "🌿 Fear",
     "Below-average optimism: a meaningful pullback or elevated volatility. Reasonable "
     "zone to accelerate a DCA schedule or add an extra tranche — while keeping some dry "
     "powder in case fear deepens."),
    (55, "⚪ Neutral",
     "No edge from sentiment either way. The evidence-backed default: stick to your "
     "regular DCA schedule; don't force a view."),
    (75, "🟠 Greed",
     "Markets near highs, volatility becalmed, trend stretched. New lump sums have a "
     "worse historical entry-price distribution here — keep DCA going (never stop it), "
     "but this is the zone for patience with EXTRA cash, not urgency."),
    (101, "🔴 Extreme greed",
     "Everything stretched at once — the statistically worst entry zone of the cycle for "
     "new lump sums. Keep the scheduled DCA (timing skips cost more than they save), "
     "hold extra cash for the next fear episode, and rebalance if equity weight has "
     "drifted above plan."),
]


def compute_entry_meter(ctx: dict, breadth_pct: Optional[float],
                        spx_rsi14: Optional[float],
                        vix_pctile_1y: Optional[float]) -> dict:
    """Blend the signals into one 0–100 fear/greed reading.

    ctx is market_risk.get_market_context()'s dict. breadth_pct is the share
    (0–100) of tracked tickers trading above their own 200-day average.
    Returns {"score", "zone", "stance", "components": [(label, greed_0_100, detail)]}.
    """
    comps: list[tuple[str, float, str]] = []

    price, hi = ctx.get("index_price"), ctx.get("index_52w_high")
    if price and hi and hi > 0:
        dd = price / hi - 1.0                      # 0 at the high, negative below
        g = _clip01(1.0 + dd / 0.20) * 100         # −20% or worse -> 0
        comps.append(("Drawdown from 52w high", g, f"{dd*100:+.1f}% vs high"))

    vix = ctx.get("vix")
    if vix is not None:
        g = _clip01((40.0 - vix) / 28.0) * 100     # 12 -> 100 · 40 -> 0
        comps.append(("VIX level", g, f"{vix:.1f}"))

    if vix_pctile_1y is not None:
        comps.append(("VIX vs 1y range", (1.0 - vix_pctile_1y) * 100,
                      f"{vix_pctile_1y*100:.0f}th percentile of the past year"))

    stretch = ctx.get("pct_from_ma200")            # in percent
    if stretch is not None:
        g = _clip01((stretch + 10.0) / 20.0) * 100  # −10% -> 0 · +10% -> 100
        comps.append(("Stretch vs 200-day avg", g, f"{stretch:+.1f}%"))

    pos = ctx.get("index_52w_pos")                 # 0-100
    if pos is not None:
        comps.append(("52-week range position", _clip01(pos / 100.0) * 100,
                      f"{pos:.0f}% of range"))

    if spx_rsi14 is not None:
        g = _clip01((spx_rsi14 - 30.0) / 40.0) * 100  # 30 -> 0 · 70 -> 100
        comps.append(("Index RSI(14)", g, f"{spx_rsi14:.0f}"))

    if breadth_pct is not None:
        comps.append(("Breadth (tickers > 200-day)", _clip01(breadth_pct / 100.0) * 100,
                      f"{breadth_pct:.0f}% of tracked names"))

    if not comps:
        return {"score": None, "zone": "unknown", "stance": "No market data available.",
                "components": []}

    score = sum(c[1] for c in comps) / len(comps)
    for cutoff, zone, stance in ZONES:
        if score < cutoff:
            return {"score": score, "zone": zone, "stance": stance, "components": comps}
    # unreachable, but keep a safe default
    return {"score": score, "zone": ZONES[-1][1], "stance": ZONES[-1][2], "components": comps}
