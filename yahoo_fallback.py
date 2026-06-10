"""
yahoo_fallback.py — free, keyless fallback data source (yfinance).

Why this exists
---------------
FMP's free tier caps at ~250 calls/day. Once that's spent, the app used to show
empty cells for anything not already cached — the cache can't serve data that was
never successfully fetched. Yahoo Finance has no daily API-key quota, so anything
FMP couldn't deliver gets filled from here, then written into the SAME disk cache
app.py uses, so the next load is free either way.

Design
------
* Quotes:        one bulk ``yf.download`` (1y daily history) for price, day %,
                 52-week range and 200-day average, plus ``fast_info`` for
                 market cap. No ``.info`` call needed -> fast and rarely limited.
* Fundamentals:  ``Ticker.info`` (one HTTP call per ticker) mapped onto the same
                 metric keys fundamentals.py produces, including the new
                 valuation/leverage metrics (P/S, PEG, debt/equity, etc.).
* Everything is best-effort: any per-ticker failure returns None for that ticker
  and never raises out of this module.

yfinance is an optional dependency: if it isn't installed, the public functions
return empty dicts and app.py simply behaves as before.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("yahoo_fallback")

try:
    import yfinance as yf
    HAVE_YF = True
except ImportError:  # pragma: no cover
    yf = None
    HAVE_YF = False


def _f(v) -> Optional[float]:
    """Coerce to float, mapping NaN/None/garbage to None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # NaN check


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------
def fetch_quotes_yahoo(symbols: list[str]) -> dict[str, dict]:
    """Return {symbol: quote-dict} in the same shape app.py's _norm_quote emits.
    Missing tickers are simply absent. Never raises."""
    if not HAVE_YF or not symbols:
        return {}
    out: dict[str, dict] = {}
    try:
        hist = yf.download(
            tickers=" ".join(symbols), period="1y", interval="1d",
            group_by="ticker", auto_adjust=True, progress=False, threads=True,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("yahoo bulk download failed: %s", str(e)[:160])
        return {}

    for sym in symbols:
        try:
            try:
                df = hist[sym]            # MultiIndex (normal with group_by="ticker")
            except (KeyError, TypeError):
                df = hist                 # single-ticker flat frame
            close = df["Close"].dropna()
            if close.empty:
                continue
            price = _f(close.iloc[-1])
            prev = _f(close.iloc[-2]) if len(close) >= 2 else None
            lo, hi = _f(close.min()), _f(close.max())
            ma200 = _f(close.tail(200).mean()) if len(close) >= 60 else None
            out[sym] = {
                "symbol": sym,
                "price": price,
                "changePercentage": ((price - prev) / prev * 100.0)
                                     if (price is not None and prev) else None,
                "marketCap": None,   # filled below from fast_info
                "yearLow": lo, "yearHigh": hi, "ma200": ma200,
                "pe": None, "eps": None,
                "source": "yahoo",
            }
        except Exception:  # noqa: BLE001
            continue

    # market cap + trailing P/E via fast_info (cheap, no .info scrape)
    for sym in list(out.keys()):
        try:
            fi = yf.Ticker(sym).fast_info
            out[sym]["marketCap"] = _f(getattr(fi, "market_cap", None))
        except Exception:  # noqa: BLE001
            pass
    return out


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------
def fetch_fundamentals_yahoo(symbol: str) -> Optional[dict]:
    """Map Yahoo's ``.info`` onto fundamentals.py's metric keys.
    Returns None on total failure; otherwise a dict (values may be None)."""
    if not HAVE_YF:
        return None
    try:
        info = yf.Ticker(symbol).info or {}
    except Exception as e:  # noqa: BLE001
        logger.warning("%s: yahoo info failed: %s", symbol, str(e)[:120])
        return None
    if not info or len(info) < 5:
        # Yahoo sometimes returns a near-empty shell; treat as a miss.
        return None

    mcap = _f(info.get("marketCap"))
    rev = _f(info.get("totalRevenue"))
    fcf = _f(info.get("freeCashflow"))
    ebitda = _f(info.get("ebitda"))
    total_debt = _f(info.get("totalDebt"))
    total_cash = _f(info.get("totalCash"))
    pe = _f(info.get("trailingPE"))
    eps_g = _f(info.get("earningsGrowth"))     # fraction, e.g. 0.12
    rev_g = _f(info.get("revenueGrowth"))      # fraction
    op_m = _f(info.get("operatingMargins"))
    gross_m = _f(info.get("grossMargins"))
    net_m = _f(info.get("profitMargins"))

    out: dict = {
        "earnings_yield": (1.0 / pe) if pe and pe > 0 else None,
        "roic": None,  # Yahoo has no ROIC; approximate with ROA-lifted ROE blend below
        "fcf_yield": (fcf / mcap) if (fcf is not None and mcap) else None,
        "rev_growth": rev_g,
        "gross_margin": gross_m,
        "operating_margin": op_m,
        "net_margin": net_m,
        "fcf_margin": (fcf / rev) if (fcf is not None and rev) else None,
        "rule_of_40": None,
        "mom_52w": None, "mom_ma200": None,   # app.py fills from the quote
        "safety": None,                        # composed in app.py merge if possible
        # new valuation / leverage metrics
        "ps_ratio": (mcap / rev) if (mcap and rev) else _f(info.get("priceToSalesTrailing12Months")),
        "peg": None,
        "eps_growth": eps_g,
        "debt_equity": None,
        "net_debt_ebitda": None,
        "interest_coverage": None,
        "source": "yahoo",
    }

    # ROIC proxy: ROE damped toward ROA (Yahoo exposes both).
    roe = _f(info.get("returnOnEquity"))
    roa = _f(info.get("returnOnAssets"))
    if roe is not None and roa is not None:
        out["roic"] = (roe + roa) / 2.0
    elif roa is not None:
        out["roic"] = roa

    # PEG: only meaningful with positive P/E and positive growth.
    if pe and pe > 0 and eps_g and eps_g > 0:
        out["peg"] = pe / (eps_g * 100.0)
    else:
        peg_info = _f(info.get("trailingPegRatio") or info.get("pegRatio"))
        out["peg"] = peg_info if (peg_info and peg_info > 0) else None

    # Leverage. debtToEquity from Yahoo is in PERCENT (e.g. 41.5) — normalize.
    de = _f(info.get("debtToEquity"))
    if de is not None:
        out["debt_equity"] = de / 100.0 if de > 10 else de
    if total_debt is not None and ebitda and ebitda > 0:
        out["net_debt_ebitda"] = (total_debt - (total_cash or 0.0)) / ebitda

    if out["rev_growth"] is not None:
        margin = out["fcf_margin"] if out["fcf_margin"] is not None else op_m
        if margin is not None:
            out["rule_of_40"] = (out["rev_growth"] + margin) * 100.0

    return out
