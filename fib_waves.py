"""
fib_waves.py — Elliott Wave stage detection, scored with Fibonacci ratios.

For each stock this answers: "Which leg of a 5-wave move (1-2-3-4-5) or of the
A-B-C correction after it is the price in right now, and how well does the
count fit the classic Fibonacci proportions?"

How it works
------------
1. **Swings.** A ZigZag marks swing highs/lows. The swing size is a multiple of
   the stock's own volatility (median ATR as % of price), so a sleepy utility
   and a hypergrowth name are both measured in their own units. "auto" mode
   tries several sizes (small / medium / large swings) and keeps the best count.
2. **Hard rules** (a count that breaks any of these is thrown out):
   wave 2 never retraces all of wave 1 · wave 3 goes beyond wave 1 and is never
   the shortest of 1/3/5 · wave 4 never overlaps wave 1 · wave 5 goes beyond 3 ·
   a regular B never exceeds the wave-5 extreme.
3. **Fibonacci score.** Surviving counts are scored on how close each finished
   leg is to its textbook ratio (wave 2 ≈ 50–61.8% of 1, wave 3 ≈ 1.618× of 1,
   wave 4 ≈ 23.6–38.2% of 3, wave 5 ≈ 0.618–1× of 1, A ≈ 38–62% of 0→5,
   B ≈ 50–79% of A), each measured against what a random walk scores, plus
   credit when the count starts at a real extreme and a penalty when an
   impulse count fights the 200-day average.
4. **Confidence** cut-offs are calibrated on random-walk prices (see
   ``calibrate()``): on pure noise only ~1 count in 10 reaches High.

Elliott Wave is subjective by nature — two analysts often count the same chart
differently. Treat the output as context (where are we in the swing, where is
the next Fibonacci level, where is the count wrong), never as a trade signal.

Public API
----------
``analyze_waves(prices, symbol=None, sensitivity="auto") -> WaveResult``
``wave_chart(prices, result) -> plotly Figure``
``normalize_prices(data) -> DataFrame[date, open, high, low, close]``
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------
# Textbook Fibonacci proportions for each leg.
W2_OF_W1 = (0.5, 0.618)          # wave 2 retraces 50–61.8% of wave 1
W3_OF_W1 = (1.618, 2.618, 1.0)   # wave 3 extends 1.618× wave 1 (sometimes 2.618× or 1×)
W4_OF_W3 = (0.236, 0.382)        # wave 4 retraces 23.6–38.2% of wave 3
W5_OF_W1 = (0.618, 1.0)          # wave 5 ≈ 0.618× or 1× wave 1
A_OF_IMPULSE = (0.382, 0.5, 0.618)  # A retraces 38.2–61.8% of the whole 0→5 move
B_OF_A = (0.5, 0.618, 0.786)     # B retraces 50–78.6% of A
FIT_TOL = 0.12                   # ±12% (log scale) counts as a close fit

# Average fit each ratio gets on random-walk prices — only a fit ABOVE this is
# evidence for the count (measured by calibrate(); re-run it if you change rules).
BASELINE = {"W2/W1": 0.39, "W3/W1": 0.52, "W4/W3": 0.45, "W5/W1": 0.30,
            "A/(0-5)": 0.55, "B/A": 0.65}
FIB_WEIGHT = 1.0
START_WEIGHT = 0.6       # credit when wave 0 is the most extreme price of the prior ~6 months
START_HORIZON = 126      # bars (~6 months)
TREND_WEIGHT = 0.3       # impulse counts agreeing / fighting the 200-day average

# Confidence cut-offs on the final score, from random-walk calibration.
HIGH_SCORE = 1.0
MEDIUM_SCORE = 0.8

AUTO_ATR_MULTIPLES = (3.0, 4.5, 6.0)            # swing sizes tried in "auto" mode
SENSITIVITY = {"high": 3.0, "medium": 4.5, "low": 6.0}
LABELS = ["0", "1", "2", "3", "4", "5", "A", "B", "C"]
STAGE_OF = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: "A", 7: "B", 8: "C"}
MIN_BARS = 150


# --------------------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------------------
@dataclass
class WaveResult:
    symbol: Optional[str] = None
    stage: Any = None                 # 1..5, "A", "B", "C" or None
    trend: str = ""                   # "up" / "down" (direction of the 5-wave move)
    label: str = "No clear count"
    confidence: str = ""              # "High" / "Medium" / "Low"
    score: float = 0.0
    note: str = ""
    next_level: Optional[tuple] = None           # (name, price) nearest Fibonacci level ahead
    targets: list = field(default_factory=list)  # [(name, price)]
    invalidation: Optional[float] = None
    invalidation_reason: str = ""
    ratios: dict = field(default_factory=dict)   # measured ratios, e.g. {"W2/W1": 0.59}
    points: list = field(default_factory=list)   # [(date, price, "0".."5","A","B")]
    leg_end: Optional[tuple] = None              # (date, price, "3?") extreme of the leg in progress
    zigzag: list = field(default_factory=list)   # [(date, price)] every swing point
    swing_pct: float = 0.0
    maybe_ending: bool = False
    signal: str = ""                  # "Strong Buy" / "Buy" / "Hold" / "Sell"
    signal_reason: str = ""
    alternate: str = ""
    last_close: float = float("nan")
    last_date: Any = None

    @property
    def wave(self) -> str:
        return "" if self.stage is None else str(self.stage)

    def as_row(self) -> dict:
        """Flat dict for a screener table."""
        nxt = f"{self.next_level[1]:,.2f} ({self.next_level[0]})" if self.next_level else ""
        return {
            "Wave": self.wave,
            "Trend": self.trend.title(),
            "Wave confidence": self.confidence,
            "Next Fib level": nxt,
            "Invalid beyond": self.invalidation,
        }

    def summary(self) -> dict:
        """JSON-safe dict (for disk caches) — everything except the chart geometry."""
        return {
            "stage": self.wave or None, "trend": self.trend, "label": self.label,
            "confidence": self.confidence, "score": round(float(self.score), 3),
            "note": self.note,
            "next_level": list(self.next_level) if self.next_level else None,
            "invalidation": self.invalidation, "invalidation_reason": self.invalidation_reason,
            "ratios": self.ratios, "maybe_ending": self.maybe_ending,
            "alternate": self.alternate, "last_close": _safe(self.last_close),
            "signal": self.signal, "signal_reason": self.signal_reason,
        }


def _safe(v):
    try:
        v = float(v)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def normalize_prices(data) -> pd.DataFrame:
    """Return date/open/high/low/close sorted oldest->newest.

    Accepts a yfinance DataFrame (DatetimeIndex, Open/High/Low/Close), FMP /stable
    responses (list of dicts), FMP v3 ({"historical": [...]}), or any DataFrame
    with those columns in any case.
    """
    if isinstance(data, dict) and "historical" in data:
        data = data["historical"]
    df = pd.DataFrame(data).copy() if not isinstance(data, pd.DataFrame) else data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[-1] if c[-1] else c[0] for c in df.columns]
    df.columns = [str(c).split(".")[-1].strip().lower() for c in df.columns]
    if "date" not in df.columns:
        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        if "date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "date"})
    if "close" not in df.columns and "adjclose" in df.columns:
        df["close"] = df["adjclose"]
    for c in ("open", "high", "low"):
        if c not in df.columns:
            df[c] = df["close"]
    df = df[["date", "open", "high", "low", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date")
    df["high"] = df[["high", "close"]].max(axis=1)
    df["low"] = df[["low", "close"]].min(axis=1)
    return df.reset_index(drop=True)


def _atr_pct(df: pd.DataFrame, n: int = 14) -> float:
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    prev = np.r_[c[0], c[:-1]]
    tr = np.maximum.reduce([h - l, np.abs(h - prev), np.abs(l - prev)])
    atr = pd.Series(tr).rolling(n, min_periods=n).mean().values
    pct = atr / c
    pct = pct[np.isfinite(pct)][-250:]
    return float(np.median(pct)) if len(pct) else 0.02


# --------------------------------------------------------------------------------------
# Swings
# --------------------------------------------------------------------------------------
def _zigzag(high: np.ndarray, low: np.ndarray, thr: float):
    """Swing points as [(bar, price, +1 high / -1 low)], plus the tentative extreme
    of the leg still in progress (bar, price, kind)."""
    n = len(high)
    pivots = []
    hi_i = lo_i = 0
    trend = 0
    for i in range(1, n):
        if trend == 0:
            if high[i] > high[hi_i]:
                hi_i = i
            if low[i] < low[lo_i]:
                lo_i = i
            if lo_i < hi_i and high[hi_i] >= low[lo_i] * (1 + thr):
                pivots.append((lo_i, low[lo_i], -1)); trend = 1
            elif hi_i < lo_i and low[lo_i] <= high[hi_i] * (1 - thr):
                pivots.append((hi_i, high[hi_i], 1)); trend = -1
        elif trend == 1:
            if high[i] >= high[hi_i]:
                hi_i = i
            elif low[i] <= high[hi_i] * (1 - thr):
                pivots.append((hi_i, high[hi_i], 1)); trend = -1; lo_i = i
        else:
            if low[i] <= low[lo_i]:
                lo_i = i
            elif high[i] >= low[lo_i] * (1 + thr):
                pivots.append((lo_i, low[lo_i], -1)); trend = 1; hi_i = i
    if trend == 1:
        tent = (hi_i, high[hi_i], 1)
    elif trend == -1:
        tent = (lo_i, low[lo_i], -1)
    else:
        tent = None
    return pivots, tent


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------
def _fit(ratio: float, targets) -> float:
    """1.0 at a textbook ratio, fading to ~0 about 25% away (log scale)."""
    if not ratio or ratio <= 0 or not math.isfinite(ratio):
        return 0.0
    return max(math.exp(-0.5 * (math.log(ratio / t) / FIT_TOL) ** 2) for t in targets)


RATIO_TARGETS = {"W2/W1": W2_OF_W1, "W3/W1": W3_OF_W1, "W4/W3": W4_OF_W3,
                 "W5/W1": W5_OF_W1, "A/(0-5)": A_OF_IMPULSE, "B/A": B_OF_A}


def _start_significance(nl: np.ndarray, i0: int, p0: float) -> float:
    """1.0 when wave 0 is the most extreme price of the prior ~6 months;
    proportionally less when a more extreme price is more recent."""
    beyond = np.flatnonzero(nl[:i0] < p0 - 1e-12)
    bars_back = i0 - (beyond[-1] if len(beyond) else -START_HORIZON)
    return min(1.0, bars_back / START_HORIZON)


@dataclass
class _Count:
    d: int                 # +1 up-trend impulse, -1 down-trend impulse
    bars: list             # bar index of each completed point (0..k)
    q: list                # normalized price (d * price) of each completed point
    e: float               # normalized extreme of the leg in progress
    e_bar: int
    k: int                 # completed points after wave 0 → current wave = k + 1
    ratios: dict
    score: float
    thr: float


def _valid(p: list, k: int) -> bool:
    """Hard Elliott rules on normalized points p[0..k+1] (p[k+1] = leg in progress).
    Magnitude rules only bind completed points; overshoot rules bind all."""
    m = len(p)
    w = [abs(p[i] - p[i - 1]) for i in range(1, m)]
    if m > 2 and not p[2] > p[0]:                      # W2 can't retrace all of W1
        return False
    if k >= 3 and not p[3] > p[1]:                     # W3 must exceed the W1 end
        return False
    if m > 4 and not p[4] > p[1]:                      # W4 can't overlap W1
        return False
    if m > 5 and w[2] < w[0] and w[2] < w[4]:          # W3 can't be the shortest
        return False
    if k >= 5 and not p[5] > p[3]:                     # no truncated fifths
        return False
    if m > 6 and not p[6] > p[0]:                      # A can't erase the whole impulse
        return False
    if m > 7 and not p[7] < p[5]:                      # regular B stays below the 5 extreme
        return False
    if m > 8 and not p[8] > p[0]:
        return False
    return True


def _ratios(p: list, k: int) -> dict:
    """Fibonacci ratios of COMPLETED legs only."""
    w = [abs(p[i] - p[i - 1]) for i in range(1, len(p))]
    r = {}
    if k >= 2: r["W2/W1"] = w[1] / w[0]
    if k >= 3: r["W3/W1"] = w[2] / w[0]
    if k >= 4: r["W4/W3"] = w[3] / w[2]
    if k >= 5: r["W5/W1"] = w[4] / w[0]
    if k >= 6: r["A/(0-5)"] = (p[5] - p[6]) / (p[5] - p[0])
    if k >= 7: r["B/A"] = (p[7] - p[6]) / (p[5] - p[6])
    return r


def _candidates(df: pd.DataFrame, thr: float):
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    pivots, tent = _zigzag(high, low, thr)
    if tent is None or len(pivots) < 1:
        return []
    ma200 = pd.Series(close).rolling(200, min_periods=150).mean().values[-1]
    out = []
    n_p = len(pivots)
    for s in range(max(0, n_p - 8), n_p):
        start = pivots[s]
        d = -start[2]                        # a low starts an up-move, a high a down-move
        seq = pivots[s:]
        k = len(seq) - 1
        q = [d * pp[1] for pp in seq]
        e = d * tent[1]
        pts = q + [e]
        if not _valid(pts, k):
            continue
        ratios = _ratios(pts, k)
        fib = sum(_fit(v, RATIO_TARGETS[name]) - BASELINE[name] for name, v in ratios.items())
        nl = low if d == 1 else -high
        sig = _start_significance(nl, start[0], d * start[1])
        trend_term = 0.0
        if k + 1 <= 5 and np.isfinite(ma200):
            trend_term = TREND_WEIGHT if d * (close[-1] - ma200) > 0 else -TREND_WEIGHT
        score = FIB_WEIGHT * fib + START_WEIGHT * (sig - 0.5) + trend_term
        out.append(_Count(d=d, bars=[pp[0] for pp in seq], q=q, e=e, e_bar=tent[0],
                          k=k, ratios=ratios, score=score, thr=thr))
    return out


# --------------------------------------------------------------------------------------
# Targets / invalidation / plain-English note
# --------------------------------------------------------------------------------------
def _levels(c: _Count):
    """[(name, normalized price)] Fibonacci levels for the wave in progress, plus
    (normalized invalidation, reason)."""
    q, stage = c.q, c.k + 1
    w = [abs(q[i] - q[i - 1]) for i in range(1, len(q))]
    if stage == 1:
        return [], (q[0], "wave 1 can't fall back through its own start")
    if stage == 2:
        return ([(f"{r:.1%} ret. W1", q[1] - r * w[0]) for r in (0.382, 0.5, 0.618)],
                (q[0], "wave 2 can't retrace all of wave 1"))
    if stage == 3:
        return ([(f"{r:g}×W1", q[2] + r * w[0]) for r in (1.0, 1.618, 2.618)],
                (q[2], "wave 3 can't break the wave-2 extreme"))
    if stage == 4:
        return ([(f"{r:.1%} ret. W3", q[3] - r * w[2]) for r in (0.236, 0.382, 0.5)],
                (q[1], "wave 4 can't overlap wave 1"))
    if stage == 5:
        lv = [(f"{r:g}×W1", q[4] + r * w[0]) for r in (0.618, 1.0)]
        lv.append(("0.618×(0→3)", q[4] + 0.618 * (q[3] - q[0])))
        return lv, (q[4], "wave 5 can't break the wave-4 extreme")
    if stage == 6:   # A
        full = q[5] - q[0]
        return ([(f"{r:.1%} ret. 0→5", q[5] - r * full) for r in (0.382, 0.5, 0.618)],
                (q[5], "beyond the wave-5 extreme the impulse is still extending"))
    if stage == 7:   # B
        a = q[5] - q[6]
        return ([(f"{r:.1%} ret. A", q[6] + r * a) for r in (0.5, 0.618, 0.786)],
                (q[5], "a regular B can't exceed the wave-5 extreme"))
    # stage 8: C
    a = q[5] - q[6]
    return ([(f"{r:g}×A", q[7] - r * a) for r in (1.0, 1.618)],
            (q[7], "C reversing past the B extreme ends the correction"))


_STAGE_TEXT = {
    1: ("wave 1", "an early, still-unconfirmed new move"),
    2: ("wave 2", "the first pullback after wave 1"),
    3: ("wave 3", "usually the longest, strongest leg"),
    4: ("wave 4", "a pullback before the final push"),
    5: ("wave 5", "the final push"),
    6: ("wave A", "the first leg of the correction after a finished 5-wave move"),
    7: ("wave B", "a counter-move inside the correction"),
    8: ("wave C", "the last leg of the correction"),
}


# --------------------------------------------------------------------------------------
# Rule-based signal from the wave count alone
# --------------------------------------------------------------------------------------
BUY_ZONE = {2: 0.5, 4: 0.236, 8: 0.618}   # pullback depth that puts waves 2 / 4 / C in the buy zone


def _signal(c: _Count, res: WaveResult, now: float) -> tuple:
    """Map the count to Strong Buy / Buy / Hold / Sell.

    Buy the END of pullbacks that precede an up-leg (late wave 2, 4 or C of an
    up-move), avoid chasing mature moves (late wave 5, A, B), and treat
    down-trend impulse legs as Sell. Medium+ confidence is required for any
    Buy/Sell except wave 2, which can never score above Low (no finished leg
    to measure) and is therefore capped at Buy. ``now`` is the normalized close.
    """
    q, stage, conf = c.q, c.k + 1, res.confidence
    up = c.d == 1
    solid = conf in ("High", "Medium")

    def depth():
        if stage == 2:
            return (q[1] - now) / (q[1] - q[0])
        if stage == 4:
            return (q[3] - now) / (q[3] - q[2])
        return (q[7] - now) / (q[5] - q[6])        # stage 8 (C) vs the length of A

    if up:
        if stage in (2, 4, 8):
            name = {2: "wave 2", 4: "wave 4", 8: "wave C"}[stage]
            dp, zone = depth(), BUY_ZONE[stage]
            if dp < zone:
                return "Hold", f"{name} pullback only {dp:.0%} deep — buy zone starts at {zone:.1%}"
            if stage == 2:
                return "Buy", (f"late wave 2 ({dp:.0%} retracement) ahead of a potential wave 3 — "
                               "unconfirmed count, keep the stop at the invalidation price")
            if conf == "High":
                return "Strong Buy", f"late {name} in the buy zone ({dp:.0%}) with a high-confidence count"
            if solid:
                return "Buy", f"late {name} in the buy zone ({dp:.0%})"
            return "Hold", f"{name} in the buy zone but the count is low-confidence"
        if stage == 3:
            if solid and (now - q[2]) < (q[1] - q[0]):
                return "Buy", "early wave 3 — still below the 1×W1 target"
            return "Hold", "wave 3 already well underway — avoid chasing"
        if stage == 5:
            if res.maybe_ending and solid:
                return "Sell", "wave 5 has reached its minimum target — the move is mature"
            return "Hold", "wave 5 in progress — hold, don't add"
        if stage in (6, 7):
            if solid:
                return "Sell", ("wave A — a correction has started" if stage == 6
                                else "wave B rally inside a correction — a classic trap")
            return "Hold", "correction under way, low-confidence count"
        return "Hold", "wave 1 — too early to confirm a new up-move"

    # Down-trend impulse (↓)
    if stage == 5 and res.maybe_ending:
        return "Hold", "down-move's wave 5 near its target — watch for a bottom, don't sell into it"
    if stage in (1, 2, 3, 4, 5) and solid:
        return "Sell", f"wave {stage} of a down-move"
    if stage == 8 and solid:
        return "Sell", "wave C rally inside a down-trend is ending"
    return "Hold", "down-trend count is low-confidence or a counter-rally is under way"


def _fill(res: WaveResult, c: _Count, alt: Optional[_Count], df: pd.DataFrame) -> WaveResult:
    d = c.d
    real = lambda v: float(d * v)          # normalized -> actual price
    dates = df["date"].tolist()
    close = df["close"].values
    stage_n = c.k + 1
    res.stage = STAGE_OF[stage_n]
    res.trend = "up" if d == 1 else "down"
    res.score = float(c.score)
    res.swing_pct = float(c.thr)
    res.ratios = {name: round(float(v), 3) for name, v in c.ratios.items()}
    res.points = [(dates[b], real(v), LABELS[i]) for i, (b, v) in enumerate(zip(c.bars, c.q))]
    res.leg_end = (dates[c.e_bar], real(c.e), LABELS[stage_n] + "?")

    n_rat = len(c.ratios)
    if c.score >= HIGH_SCORE and n_rat >= 2:
        res.confidence = "High"
    elif c.score >= MEDIUM_SCORE and n_rat >= 1:
        res.confidence = "Medium"
    else:
        res.confidence = "Low"

    arrow = "↑" if d == 1 else "↓"
    res.label = f"Wave {res.stage} {arrow}"

    levels, (inv_n, why) = _levels(c)
    res.targets = [(name, real(v)) for name, v in levels]
    res.invalidation = real(inv_n)
    res.invalidation_reason = why
    now = float(d * close[-1])
    # Direction the leg in progress travels (normalized): impulse legs 1/3/5 and B go with d.
    leg_dir = 1 if stage_n in (1, 3, 5, 7) else -1
    ahead = [(name, v) for name, v in levels if leg_dir * (v - now) > 0]
    if ahead:
        name, v = min(ahead, key=lambda t: abs(t[1] - now))
        res.next_level = (name, real(v))

    if stage_n == 5:
        w1 = c.q[1] - c.q[0]
        res.maybe_ending = bool((c.e - c.q[4]) >= 0.618 * w1)

    fmt = lambda v: f"{v:,.2f}"
    move = "an up-move" if d == 1 else "a down-move"
    wname, wdesc = _STAGE_TEXT[stage_n]
    note = f"Likely in {wname} of {move} ({wdesc})."
    if res.next_level:
        note += f" Next Fibonacci level: {fmt(res.next_level[1])} ({res.next_level[0]})."
    if res.maybe_ending:
        note += " Wave 5 has already reached its minimum target — the move may be near completion."
    against = "below" if res.invalidation < close[-1] else "above"
    note += f" The count fails {against} {fmt(res.invalidation)} ({why})."
    if res.confidence == "Low":
        note += " Low confidence: the legs don't match Fibonacci proportions better than chance."
    res.note = note
    res.signal, res.signal_reason = _signal(c, res, now)
    if alt is not None:
        res.alternate = f"Wave {STAGE_OF[alt.k + 1]} {'↑' if alt.d == 1 else '↓'} (score {alt.score:.2f})"
    return res


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------
def analyze_waves(prices, symbol: Optional[str] = None, sensitivity: str = "auto") -> WaveResult:
    """Best Elliott Wave count for the price history (≥ ~150 daily bars; 2 years is ideal)."""
    res = WaveResult(symbol=symbol)
    try:
        df = normalize_prices(prices)
    except Exception as e:  # noqa: BLE001
        res.note = f"Couldn't read prices: {e}"
        return res
    if len(df) < MIN_BARS:
        res.note = f"Need at least {MIN_BARS} daily bars (have {len(df)})."
        return res
    res.last_close = float(df["close"].iloc[-1])
    res.last_date = df["date"].iloc[-1]

    atr = _atr_pct(df)
    mults = AUTO_ATR_MULTIPLES if sensitivity == "auto" else (SENSITIVITY.get(sensitivity, 4.5),)
    cands, zz_by_thr = [], {}
    for m in mults:
        thr = max(0.03, m * atr)
        cs = _candidates(df, thr)
        cands += cs
        zz_by_thr[thr] = thr
    if not cands:
        res.note = "No valid Elliott count — price is choppy or the swings break the wave rules."
        return res

    cands.sort(key=lambda c: (c.score, c.k), reverse=True)
    best = cands[0]
    alt = next((c for c in cands[1:] if (c.k, c.d) != (best.k, best.d)), None)
    _fill(res, best, alt, df)

    pivots, tent = _zigzag(df["high"].values, df["low"].values, best.thr)
    dates = df["date"].tolist()
    res.zigzag = [(dates[b], float(p)) for b, p, _ in pivots] + (
        [(dates[tent[0]], float(tent[1]))] if tent else [])
    return res


def analyze_many(price_frames: dict, sensitivity: str = "auto") -> dict:
    """{symbol: prices} -> {symbol: WaveResult}. Never raises."""
    out = {}
    for sym, px in (price_frames or {}).items():
        try:
            out[sym] = analyze_waves(px, symbol=sym, sensitivity=sensitivity)
        except Exception as e:  # noqa: BLE001
            out[sym] = WaveResult(symbol=sym, note=f"analysis failed: {e}")
    return out


# --------------------------------------------------------------------------------------
# Chart
# --------------------------------------------------------------------------------------
def wave_chart(prices, res: WaveResult, min_bars: int = 150, height: int = 520):
    """Plotly figure: candles, swing points, the labelled count, Fibonacci levels
    for the wave in progress and the invalidation line."""
    import plotly.graph_objects as go

    df = normalize_prices(prices)
    if res.points:
        first = res.points[0][0]
        i0 = int(df.index[df["date"] >= first][0])
        pad = max(20, (len(df) - i0) // 6)
        start = max(0, min(i0 - pad, len(df) - min_bars))
    else:
        start = max(0, len(df) - 250)
    view = df.iloc[start:]
    x0 = view["date"].iloc[0]

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=view["date"], open=view["open"], high=view["high"],
                                 low=view["low"], close=view["close"], name="Price",
                                 increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
                                 showlegend=False))
    zz = [(t, p) for t, p in res.zigzag if t >= x0]
    if zz:
        fig.add_trace(go.Scatter(x=[t for t, _ in zz], y=[p for _, p in zz], mode="lines",
                                 line=dict(color="rgba(128,128,128,0.55)", width=1),
                                 name="Swings", hoverinfo="skip"))
    if res.points:
        pts = list(res.points) + ([res.leg_end] if res.leg_end else [])
        fig.add_trace(go.Scatter(
            x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines+markers+text",
            text=[p[2] for p in pts], textposition="top center",
            textfont=dict(size=15, color="#f5a623"),
            line=dict(color="#f5a623", width=2.5), marker=dict(size=8),
            name="Wave count"))
    for name, price in res.targets:
        fig.add_hline(y=price, line=dict(color="#4a90e2", width=1, dash="dot"),
                      annotation_text=f"{name}  {price:,.2f}", annotation_position="right")
    if res.invalidation is not None:
        fig.add_hline(y=res.invalidation, line=dict(color="#d0021b", width=1.5, dash="dash"),
                      annotation_text=f"invalidation  {res.invalidation:,.2f}",
                      annotation_position="right")
    title = f"{res.symbol or ''}  ·  {res.label}"
    if res.confidence:
        title += f"  ·  {res.confidence} confidence"
    fig.update_layout(title=title, height=height, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=150, t=50, b=10),
                      legend=dict(orientation="h", y=1.02, x=0))
    return fig


# --------------------------------------------------------------------------------------
# Calibration against random walks (run: python fib_waves.py)
# --------------------------------------------------------------------------------------
def _random_walk(n: int = 504, seed: int = 0, vol: float = 0.02) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(0, vol, n)
    close = 100 * np.exp(np.cumsum(r))
    spread = np.abs(rng.normal(0, vol * 0.6, n))
    high = close * (1 + spread)
    low = close * (1 - np.abs(rng.normal(0, vol * 0.6, n)))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n), "open": open_,
                         "high": np.maximum(high, np.maximum(open_, close)),
                         "low": np.minimum(low, np.minimum(open_, close)), "close": close})


def calibrate(n: int = 1500) -> dict:
    """Measure per-ratio random-walk fits and the null score distribution."""
    fits = {k: [] for k in RATIO_TARGETS}
    scores = []
    for i in range(n):
        r = analyze_waves(_random_walk(seed=i, vol=0.012 + 0.02 * (i % 5) / 4))
        if r.stage is None:
            continue
        scores.append(r.score)
        for name, v in r.ratios.items():
            fits[name].append(_fit(v, RATIO_TARGETS[name]))
    s = np.array(scores)
    return {"fit_mean": {k: round(float(np.mean(v)), 3) for k, v in fits.items() if v},
            "score_p67": round(float(np.quantile(s, 0.67)), 3),
            "score_p90": round(float(np.quantile(s, 0.90)), 3),
            "counted": len(s) / n}


if __name__ == "__main__":
    print(calibrate())
