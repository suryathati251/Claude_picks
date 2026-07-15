"""
market_risk.py — market-CONDITIONS context for the watchlist.

IMPORTANT FRAMING: this is NOT a buy/sell signal generator. Reliably timing
market tops and bottoms does not work. What this module does is describe the
*current environment* (trend regime, volatility regime, how stretched the index
is) so you can calibrate the things you actually control: position size, how
slowly you scale in/out, and when to rebalance.

Pulls the S&P 500 (^GSPC) and VIX (^VIX) from FMP — 2 calls total per refresh,
not per-ticker. app.py caches the result so refreshes are nearly free, and a
rate-limit response degrades to "—" instead of burning quota or crashing.
"""
from __future__ import annotations

import logging
from typing import Optional

import requests

logger = logging.getLogger("market_risk")

FMP_BASE = "https://financialmodelingprep.com/stable"
_TIMEOUT = 12


def _looks_rate_limited(text: str) -> bool:
    """Genuine quota exhaustion only — plan-restriction messages that merely
    say 'upgrade your plan' are endpoint availability, not quota."""
    t = (text or "").lower()
    return "limit reach" in t or "too many requests" in t or "bandwidth" in t


def _quote(symbol: str, api_key: str) -> tuple[Optional[dict], bool]:
    """Return (record, rate_limited). Never raises."""
    try:
        r = requests.get(f"{FMP_BASE}/quote",
                         params={"symbol": symbol, "apikey": api_key}, timeout=_TIMEOUT)
    except Exception as e:  # noqa: BLE001
        logger.warning("%s quote failed: %s", symbol, str(e)[:160])
        return None, False
    if r.status_code == 429 or _looks_rate_limited(r.text):
        logger.warning("%s quote rate-limited", symbol)
        return None, True
    if r.status_code != 200:
        logger.warning("%s quote HTTP %s", symbol, r.status_code)
        return None, False
    try:
        data = r.json()
    except ValueError:
        return None, False
    if isinstance(data, list) and data:
        return data[0], False
    if isinstance(data, dict) and data and "Error Message" not in data:
        return data, False
    return None, False


def _num(record: dict, *keys) -> Optional[float]:
    for k in keys:
        if k in record and record[k] is not None:
            try:
                return float(record[k])
            except (TypeError, ValueError):
                continue
    return None


def _vix_regime(vix: Optional[float]) -> tuple[str, str]:
    """Return (label, context). Volatility regimes are descriptive, not predictive."""
    if vix is None:
        return "unknown", "VIX unavailable."
    if vix < 15:
        return "calm", "Low volatility. Markets are complacent; this is when risk feels easy to take on — which is itself worth noting."
    if vix < 20:
        return "normal", "Volatility is around its long-run average."
    if vix < 28:
        return "elevated", "Volatility is elevated — bigger daily swings; position sizes feel larger than they did."
    if vix < 40:
        return "high", "High volatility / stress. Drawdowns can be fast and deep here."
    return "extreme", "Extreme stress. Historically these readings cluster near both bottoms AND further declines — direction is NOT knowable from the level."


def get_market_context(api_key: str) -> dict:
    """Returns a dict describing the current environment. All fields optional/None
    on fetch failure. NO field is a buy/sell instruction. Sets ``rate_limited``
    when FMP signalled the quota is spent."""
    ctx: dict = {
        "index_price": None, "index_ma200": None, "index_ma50": None,
        "trend": "unknown", "pct_from_ma200": None,
        "index_52w_low": None, "index_52w_high": None, "index_52w_pos": None,
        "vix": None, "vix_label": "unknown", "vix_context": "",
        "notes": [], "rate_limited": False,
    }

    spx, rl1 = _quote("^GSPC", api_key)
    if spx:
        price = _num(spx, "price")
        ma200 = _num(spx, "priceAvg200", "priceAverage200")
        ma50 = _num(spx, "priceAvg50", "priceAverage50")
        lo = _num(spx, "yearLow")
        hi = _num(spx, "yearHigh")
        ctx["index_price"], ctx["index_ma200"], ctx["index_ma50"] = price, ma200, ma50
        ctx["index_52w_low"], ctx["index_52w_high"] = lo, hi
        if price and ma200:
            ctx["pct_from_ma200"] = (price - ma200) / ma200 * 100
            ctx["trend"] = ("above 200-day (uptrend regime)" if price >= ma200
                            else "below 200-day (downtrend regime)")
        if price and lo and hi and hi > lo:
            ctx["index_52w_pos"] = (price - lo) / (hi - lo) * 100

    vix, rl2 = _quote("^VIX", api_key)
    if vix:
        v = _num(vix, "price")
        ctx["vix"] = v
        ctx["vix_label"], ctx["vix_context"] = _vix_regime(v)

    ctx["rate_limited"] = rl1 or rl2

    if ctx["pct_from_ma200"] is not None:
        if ctx["pct_from_ma200"] > 12:
            ctx["notes"].append(
                "Index is well above its 200-day average — extended. Not a sell signal, "
                "but a reminder that average entry prices matter; scaling in over tranches "
                "beats committing everything at a stretched level."
            )
        elif ctx["pct_from_ma200"] < -12:
            ctx["notes"].append(
                "Index is well below its 200-day average — fear regime. Historically a worse "
                "time to PANIC-SELL than to buy, but 'below' can always go lower; this is where "
                "pre-committed rules (rebalance bands, fixed DCA) protect you from emotion."
            )
    return ctx
