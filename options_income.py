"""
options_income.py — cash-secured put candidates from Yahoo options chains.

The strategy this feeds (an income twist on "buy fear"): for a stock you
already want to own at a lower price — ideally one sitting near its 52-week
low — sell an out-of-the-money PUT at the strike you'd happily pay, with cash
set aside to buy 100 shares. Outcomes: the stock stays above the strike and
you keep the premium (annualized yield shown), or it falls through and you buy
at the strike minus premium — the entry you wanted, at a discount.

Data: yfinance options chains (keyless, ~15-min delayed — fine for research,
not execution). One call lists expirations + one call per fetched expiry.
app.py caches results and bounds how many tickers are screened per load.

Pure module: no Streamlit, no caching here. Everything is best-effort and
returns [] rather than raising.

NOT ADVICE. Selling puts obligates you to buy 100 shares per contract at the
strike. Premium yields look best exactly when the market prices real risk.
"""
from __future__ import annotations

import logging
import math
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger("options_income")

try:
    import yfinance as yf
    HAVE_YF = True
except ImportError:  # pragma: no cover
    yf = None
    HAVE_YF = False

DTE_MIN, DTE_MAX = 15, 65        # the classic CSP/CC sweet spot (~1-2 months)
MAX_EXPIRIES = 2                 # nearest two monthly-ish expiries in range
OTM_MAX = 0.25                   # puts: ignore strikes >25% below spot (dead premium)
CC_OTM_MAX = 0.15                # covered calls: strikes up to 15% above spot
LEAPS_DTE_MIN, LEAPS_DTE_MAX = 270, 800   # ~9 months to ~2.2 years out
LEAPS_MONEY_LO, LEAPS_MONEY_HI = 0.95, 1.20   # strikes 5% below to 20% above spot
MIN_PREMIUM = 0.05               # skip quotes with no real bid/last
PER_TICKER_ROWS = 6              # best rows kept per ticker


def _f(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _expiries_between(tk, dte_min: int, dte_max: int, limit: int,
                      furthest_first: bool = False) -> list[tuple[str, int]]:
    """[(expiry, dte)] for listed expirations inside a DTE window."""
    try:
        exps = list(tk.options or [])
    except Exception as e:  # noqa: BLE001
        logger.warning("options list failed: %s", str(e)[:100])
        return []
    today = date.today()
    out = []
    for e in exps:
        try:
            dte = (datetime.strptime(e, "%Y-%m-%d").date() - today).days
        except ValueError:
            continue
        if dte_min <= dte <= dte_max:
            out.append((e, dte))
    if furthest_first:
        out.sort(key=lambda t: -t[1])
    return out[:limit]


def live_spot(symbol: str) -> Optional[float]:
    """Near-live price via yfinance fast_info (keyless, ~15-min delayed). Used
    by the options screens so contracts are never priced off a stale cache.
    None on any failure — callers fall back to their cached quote."""
    if not HAVE_YF:
        return None
    try:
        v = _f(getattr(yf.Ticker(symbol).fast_info, "last_price", None))
        return v if v and v > 0 else None
    except Exception:  # noqa: BLE001
        return None


def fetch_put_candidates(symbol: str, spot: Optional[float]) -> list[dict]:
    """Best cash-secured-put candidates for one ticker (strikes at/below spot,
    15-65 DTE), ranked by annualized premium yield. [] on any failure or when
    the name has no listed options (non-US tickers, tiny caps)."""
    if not HAVE_YF or not spot or spot <= 0:
        return []
    try:
        tk = yf.Ticker(symbol)
    except Exception:  # noqa: BLE001
        return []
    in_range = _expiries_between(tk, DTE_MIN, DTE_MAX, MAX_EXPIRIES)

    rows: list[dict] = []
    for expiry, dte in in_range:
        try:
            puts = tk.option_chain(expiry).puts
        except Exception as e:  # noqa: BLE001
            logger.warning("%s %s: chain failed: %s", symbol, expiry, str(e)[:100])
            continue
        if puts is None or getattr(puts, "empty", True):
            continue
        for _, r in puts.iterrows():
            strike = _f(r.get("strike"))
            if strike is None or strike > spot or strike < spot * (1 - OTM_MAX):
                continue
            bid, last = _f(r.get("bid")) or 0.0, _f(r.get("lastPrice")) or 0.0
            premium = bid if bid > 0 else last          # bid = conservative
            if premium < MIN_PREMIUM:
                continue
            oi = int(_f(r.get("openInterest")) or 0)
            iv = _f(r.get("impliedVolatility"))
            yld = premium / strike
            rows.append({
                "symbol": symbol,
                "spot": spot,
                "strike": strike,
                "otm_pct": (spot - strike) / spot,           # how far below spot
                "expiry": expiry,
                "dte": dte,
                "premium": premium,
                "premium_src": "bid" if bid > 0 else "last",
                "yield": yld,                                 # for the period
                "annualized": yld * 365.0 / max(dte, 1),
                "breakeven": strike - premium,                # effective entry if assigned
                "cushion": (spot - (strike - premium)) / spot,  # drop absorbed before losing
                "iv": iv,
                "oi": oi,
                "cash_needed": strike * 100.0,
            })
    rows.sort(key=lambda r: r["annualized"], reverse=True)
    return rows[:PER_TICKER_ROWS]


def fetch_call_candidates(symbol: str, spot: Optional[float]) -> list[dict]:
    """COVERED-CALL candidates: OTM calls up to 15% above spot, 15-65 DTE, for
    shares you already own — premium yield on the position plus the total
    return if the stock is called away at the strike. Ranked by annualized
    premium yield."""
    if not HAVE_YF or not spot or spot <= 0:
        return []
    try:
        tk = yf.Ticker(symbol)
    except Exception:  # noqa: BLE001
        return []
    rows: list[dict] = []
    for expiry, dte in _expiries_between(tk, DTE_MIN, DTE_MAX, MAX_EXPIRIES):
        try:
            calls = tk.option_chain(expiry).calls
        except Exception as e:  # noqa: BLE001
            logger.warning("%s %s: calls chain failed: %s", symbol, expiry, str(e)[:100])
            continue
        if calls is None or getattr(calls, "empty", True):
            continue
        for _, r in calls.iterrows():
            strike = _f(r.get("strike"))
            if strike is None or strike < spot or strike > spot * (1 + CC_OTM_MAX):
                continue
            bid, last = _f(r.get("bid")) or 0.0, _f(r.get("lastPrice")) or 0.0
            premium = bid if bid > 0 else last          # you sell -> bid is honest
            if premium < MIN_PREMIUM:
                continue
            yld = premium / spot                        # yield on the shares held
            called = (strike - spot + premium) / spot   # total return if assigned
            rows.append({
                "symbol": symbol, "spot": spot, "strike": strike,
                "otm_pct": (strike - spot) / spot,      # headroom to the cap
                "expiry": expiry, "dte": dte,
                "premium": premium, "premium_src": "bid" if bid > 0 else "last",
                "yield": yld,
                "annualized": yld * 365.0 / max(dte, 1),
                "called_return": called,
                "called_annualized": called * 365.0 / max(dte, 1),
                "iv": _f(r.get("impliedVolatility")),
                "oi": int(_f(r.get("openInterest")) or 0),
            })
    rows.sort(key=lambda r: r["annualized"], reverse=True)
    return rows[:PER_TICKER_ROWS]


def fetch_leaps_candidates(symbol: str, spot: Optional[float]) -> list[dict]:
    """LONG-DATED CALL (LEAPS) candidates: 9-26 months out, strikes 5% below to
    20% above spot — a defined-risk way to express a 10x-radar thesis. You BUY
    these, so premium uses the ask (conservative). Ranked by the smallest move
    needed to break even at expiry."""
    if not HAVE_YF or not spot or spot <= 0:
        return []
    try:
        tk = yf.Ticker(symbol)
    except Exception:  # noqa: BLE001
        return []
    rows: list[dict] = []
    for expiry, dte in _expiries_between(tk, LEAPS_DTE_MIN, LEAPS_DTE_MAX, 2,
                                         furthest_first=True):
        try:
            calls = tk.option_chain(expiry).calls
        except Exception as e:  # noqa: BLE001
            logger.warning("%s %s: leaps chain failed: %s", symbol, expiry, str(e)[:100])
            continue
        if calls is None or getattr(calls, "empty", True):
            continue
        for _, r in calls.iterrows():
            strike = _f(r.get("strike"))
            if strike is None or not (spot * LEAPS_MONEY_LO <= strike <= spot * LEAPS_MONEY_HI):
                continue
            ask, last = _f(r.get("ask")) or 0.0, _f(r.get("lastPrice")) or 0.0
            premium = ask if ask > 0 else last          # you buy -> ask is honest
            if premium < MIN_PREMIUM:
                continue
            breakeven = strike + premium
            rows.append({
                "symbol": symbol, "spot": spot, "strike": strike,
                "moneyness": (strike - spot) / spot,    # + = OTM, - = ITM
                "expiry": expiry, "dte": dte,
                "premium": premium, "premium_src": "ask" if ask > 0 else "last",
                "breakeven": breakeven,
                "be_move": (breakeven - spot) / spot,   # rise needed by expiry
                "cost": premium * 100.0,                # per contract; also max loss
                "leverage": spot / premium if premium > 0 else None,
                "iv": _f(r.get("impliedVolatility")),
                "oi": int(_f(r.get("openInterest")) or 0),
            })
    rows.sort(key=lambda r: r["be_move"])
    return rows[:PER_TICKER_ROWS]


# ---------------------------------------------------------------------------
# Wheel strategy — ~30-delta / ~30-DTE puts (premium test: >=1.5% per cycle)
# ---------------------------------------------------------------------------
RISK_FREE = 0.04          # r for the delta approximation; precision hardly matters
WHEEL_DTE_MIN, WHEEL_DTE_MAX = 20, 45
WHEEL_DELTA_LO, WHEEL_DELTA_HI = -0.37, -0.23   # "around 30 delta"


def put_delta(spot: float, strike: float, iv: Optional[float],
              dte: int) -> Optional[float]:
    """Black-Scholes put delta from the chain's implied volatility (Yahoo ships
    IV but no greeks). Negative number in [-1, 0]; None when IV is unusable."""
    if not spot or not strike or iv is None or iv <= 0 or dte <= 0:
        return None
    t = dte / 365.0
    try:
        d1 = (math.log(spot / strike) + (RISK_FREE + iv * iv / 2.0) * t) / (iv * math.sqrt(t))
    except (ValueError, ZeroDivisionError):
        return None
    return 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0))) - 1.0


def fetch_wheel_candidates(symbol: str, spot: Optional[float]) -> list[dict]:
    """WHEEL puts: ~30-delta, ~30-DTE cash-secured puts — the entry leg of the
    wheel. Returns the best contracts by premium return per cycle
    (premium / strike), the number the 1.5% criterion tests."""
    if not HAVE_YF or not spot or spot <= 0:
        return []
    try:
        tk = yf.Ticker(symbol)
    except Exception:  # noqa: BLE001
        return []
    expiries = _expiries_between(tk, WHEEL_DTE_MIN, WHEEL_DTE_MAX, 3)
    expiries.sort(key=lambda t: abs(t[1] - 30))          # closest to 30 DTE first
    rows: list[dict] = []
    for expiry, dte in expiries[:2]:
        try:
            puts = tk.option_chain(expiry).puts
        except Exception as e:  # noqa: BLE001
            logger.warning("%s %s: wheel chain failed: %s", symbol, expiry, str(e)[:100])
            continue
        if puts is None or getattr(puts, "empty", True):
            continue
        for _, r in puts.iterrows():
            strike = _f(r.get("strike"))
            iv = _f(r.get("impliedVolatility"))
            if strike is None or strike > spot:
                continue
            delta = put_delta(spot, strike, iv, dte)
            if delta is None or not (WHEEL_DELTA_LO <= delta <= WHEEL_DELTA_HI):
                continue
            bid, last = _f(r.get("bid")) or 0.0, _f(r.get("lastPrice")) or 0.0
            premium = bid if bid > 0 else last
            if premium < MIN_PREMIUM:
                continue
            ret = premium / strike                        # the wheel's cycle return
            rows.append({
                "symbol": symbol, "spot": spot, "strike": strike, "delta": delta,
                "expiry": expiry, "dte": dte,
                "premium": premium, "premium_src": "bid" if bid > 0 else "last",
                "cycle_return": ret,
                "annualized": ret * 365.0 / max(dte, 1),
                "breakeven": strike - premium,
                "iv": iv, "oi": int(_f(r.get("openInterest")) or 0),
                "cash_needed": strike * 100.0,
            })
    rows.sort(key=lambda r: r["cycle_return"], reverse=True)
    return rows[:4]


# ---------------------------------------------------------------------------
# Put credit spreads — defined-risk premium selling (works on XSP/SPX too)
# ---------------------------------------------------------------------------
SPREAD_DTE_MIN, SPREAD_DTE_MAX = 20, 50
SPREAD_DELTA_LO, SPREAD_DELTA_HI = -0.37, -0.20   # short leg around 30-20 delta
SPREAD_MAX_WIDTH_PCT = 0.10                       # long leg within 10% of spot below
MIN_CREDIT = 0.05


def fetch_put_spread_candidates(symbol: str, spot: Optional[float]) -> list[dict]:
    """PUT CREDIT SPREADS: sell a ~20-35Δ put, buy a further-OTM put in the same
    expiry. Max loss = width − credit, known up front — the defined-risk version
    of the cash-secured put. Credit is conservative: short leg at the BID, long
    leg at the ASK. Ranked by return on risk (credit ÷ max loss)."""
    if not HAVE_YF or not spot or spot <= 0:
        return []
    try:
        tk = yf.Ticker(symbol)
    except Exception:  # noqa: BLE001
        return []
    rows: list[dict] = []
    for expiry, dte in _expiries_between(tk, SPREAD_DTE_MIN, SPREAD_DTE_MAX, 3)[:2]:
        try:
            puts = tk.option_chain(expiry).puts
        except Exception as e:  # noqa: BLE001
            logger.warning("%s %s: spread chain failed: %s", symbol, expiry, str(e)[:100])
            continue
        if puts is None or getattr(puts, "empty", True):
            continue
        legs = {}
        for _, r in puts.iterrows():
            k = _f(r.get("strike"))
            if k is None:
                continue
            legs[k] = {"bid": _f(r.get("bid")) or 0.0, "ask": _f(r.get("ask")) or 0.0,
                       "last": _f(r.get("lastPrice")) or 0.0,
                       "iv": _f(r.get("impliedVolatility")),
                       "oi": int(_f(r.get("openInterest")) or 0)}
        strikes = sorted(legs)
        for k_short in strikes:
            if k_short > spot:
                continue
            leg_s = legs[k_short]
            delta = put_delta(spot, k_short, leg_s["iv"], dte)
            if delta is None or not (SPREAD_DELTA_LO <= delta <= SPREAD_DELTA_HI):
                continue
            short_prem = leg_s["bid"] if leg_s["bid"] > 0 else leg_s["last"]
            if short_prem < MIN_CREDIT:
                continue
            lowers = [k for k in strikes
                      if k < k_short and (k_short - k) <= SPREAD_MAX_WIDTH_PCT * spot]
            for k_long in sorted(lowers, reverse=True)[:3]:   # nearest 3 widths
                leg_l = legs[k_long]
                long_prem = leg_l["ask"] if leg_l["ask"] > 0 else leg_l["last"]
                credit = short_prem - long_prem
                width = k_short - k_long
                max_loss = width - credit
                if credit < MIN_CREDIT or max_loss <= 0:
                    continue
                rows.append({
                    "symbol": symbol, "spot": spot,
                    "short_strike": k_short, "long_strike": k_long, "width": width,
                    "delta": delta, "expiry": expiry, "dte": dte,
                    "credit": credit,
                    "max_loss": max_loss,
                    "ror": credit / max_loss,                      # return on risk
                    "annualized": (credit / max_loss) * 365.0 / max(dte, 1),
                    "breakeven": k_short - credit,
                    "cushion": (spot - (k_short - credit)) / spot,
                    "pop": 1.0 + delta,                            # ≈ P(short leg expires OTM)
                    "iv": leg_s["iv"],
                    "oi": min(leg_s["oi"], leg_l["oi"]),
                    "bp_needed": max_loss * 100.0,                 # buying power per spread
                })
    rows.sort(key=lambda r: r["ror"], reverse=True)
    return rows[:PER_TICKER_ROWS]
