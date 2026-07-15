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
