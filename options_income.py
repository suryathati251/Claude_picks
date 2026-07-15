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

DTE_MIN, DTE_MAX = 15, 65        # the classic CSP sweet spot (~1-2 months)
MAX_EXPIRIES = 2                 # nearest two monthly-ish expiries in range
OTM_MAX = 0.25                   # ignore strikes >25% below spot (dead premium)
MIN_PREMIUM = 0.05               # skip quotes with no real bid/last
PER_TICKER_ROWS = 6              # best rows kept per ticker


def _f(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def fetch_put_candidates(symbol: str, spot: Optional[float]) -> list[dict]:
    """Best cash-secured-put candidates for one ticker (strikes at/below spot,
    15-65 DTE), ranked by annualized premium yield. [] on any failure or when
    the name has no listed options (non-US tickers, tiny caps)."""
    if not HAVE_YF or not spot or spot <= 0:
        return []
    try:
        tk = yf.Ticker(symbol)
        expiries = list(tk.options or [])
    except Exception as e:  # noqa: BLE001
        logger.warning("%s: options list failed: %s", symbol, str(e)[:100])
        return []

    today = date.today()
    in_range = []
    for e in expiries:
        try:
            dte = (datetime.strptime(e, "%Y-%m-%d").date() - today).days
        except ValueError:
            continue
        if DTE_MIN <= dte <= DTE_MAX:
            in_range.append((e, dte))
    in_range = in_range[:MAX_EXPIRIES]

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
