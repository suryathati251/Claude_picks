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

from fundamentals import edge_net_debt_ebitda, edge_interest_coverage, edge_ev_ebit, moat_metrics

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
def fetch_quotes_yahoo(symbols: list[str], with_mcap: bool = True) -> dict[str, dict]:
    """Return {symbol: quote-dict} in the same shape app.py's _norm_quote emits.
    Missing tickers are simply absent. Never raises.

    with_mcap=False skips the per-ticker fast_info market-cap calls — one bulk
    price download only. Use it when refreshing STALE quotes for many symbols
    (price freshness is the point; the caller keeps the old market cap)."""
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
    if with_mcap:
        for sym in list(out.keys()):
            try:
                fi = yf.Ticker(sym).fast_info
                out[sym]["marketCap"] = _f(getattr(fi, "market_cap", None))
            except Exception:  # noqa: BLE001
                pass
    return out


# ---------------------------------------------------------------------------
# Momentum — canonical 12-minus-1-month total return (free, from price history)
# ---------------------------------------------------------------------------
def fetch_momentum_yahoo(symbols: list[str]) -> dict[str, dict]:
    """Return {symbol: {"mom_12_1": float|None}}: the return from ~12 months ago
    to ~1 month ago (skipping the latest month to avoid short-term reversal — the
    canonical momentum factor). One bulk download; never raises."""
    if not HAVE_YF or not symbols:
        return {}
    try:
        hist = yf.download(
            tickers=" ".join(symbols), period="1y", interval="1d",
            group_by="ticker", auto_adjust=True, progress=False, threads=True,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("yahoo momentum download failed: %s", str(e)[:160])
        return {}

    out: dict[str, dict] = {}
    for sym in symbols:
        try:
            try:
                df = hist[sym]
            except (KeyError, TypeError):
                df = hist
            close = df["Close"].dropna()
            if len(close) < 200:                       # need ~10mo+ to be meaningful
                continue
            p_1m = _f(close.iloc[-21])                 # ~1 month ago
            p_12m = _f(close.iloc[-252]) if len(close) >= 252 else _f(close.iloc[0])
            if p_1m and p_12m and p_12m > 0:
                out[sym] = {"mom_12_1": p_1m / p_12m - 1.0}
        except Exception:  # noqa: BLE001
            continue
    return out


# ---------------------------------------------------------------------------
# Fundamentals — statement-based (NOT .info)
# ---------------------------------------------------------------------------
# Yahoo's quoteSummary endpoint (behind ``Ticker.info``) requires cookie/crumb
# auth and 401s / rate-limits cloud-provider IPs (e.g. Streamlit Cloud). The
# fundamentals-timeseries endpoint behind ``income_stmt`` / ``balance_sheet`` /
# ``cashflow`` does not, so we derive everything from raw statements — the same
# approach fundamentals.py uses with FMP.

def _row(df, *names) -> Optional[float]:
    """Get the most-recent value for the first matching row label.
    Matches case/space-insensitively so 'Total Revenue'/'TotalRevenue' both work."""
    if df is None or getattr(df, "empty", True):
        return None
    norm = {str(idx).replace(" ", "").lower(): idx for idx in df.index}
    for name in names:
        idx = norm.get(name.replace(" ", "").lower())
        if idx is None:
            continue
        series = df.loc[idx].dropna()
        if len(series):
            return _f(series.iloc[0])
    return None


def _row_prev(df, *names) -> Optional[float]:
    """Like _row but the SECOND most-recent value (prior fiscal year)."""
    if df is None or getattr(df, "empty", True):
        return None
    norm = {str(idx).replace(" ", "").lower(): idx for idx in df.index}
    for name in names:
        idx = norm.get(name.replace(" ", "").lower())
        if idx is None:
            continue
        series = df.loc[idx].dropna()
        if len(series) >= 2:
            return _f(series.iloc[1])
    return None


def _row_all(df, *names) -> list:
    """All yearly values for the first matching row label, most-recent-first."""
    if df is None or getattr(df, "empty", True):
        return []
    norm = {str(idx).replace(" ", "").lower(): idx for idx in df.index}
    for name in names:
        idx = norm.get(name.replace(" ", "").lower())
        if idx is not None:
            return [_f(v) for v in df.loc[idx].tolist()]
    return []


def fetch_fundamentals_yahoo(symbol: str, market_cap: Optional[float] = None) -> Optional[dict]:
    """Build fundamentals.py's metric dict from Yahoo annual statements.
    Returns None on total failure; otherwise a dict (values may be None)."""
    if not HAVE_YF:
        return None
    try:
        ti = yf.Ticker(symbol)
        inc = ti.income_stmt        # annual income statement
        bs = ti.balance_sheet       # annual balance sheet
        cf = ti.cashflow            # annual cash-flow statement
    except Exception as e:  # noqa: BLE001
        logger.warning("%s: yahoo statements failed: %s", symbol, str(e)[:120])
        return None
    if (inc is None or getattr(inc, "empty", True)) and (bs is None or getattr(bs, "empty", True)):
        return None

    if market_cap is None:
        try:
            market_cap = _f(getattr(ti.fast_info, "market_cap", None))
        except Exception:  # noqa: BLE001
            market_cap = None

    rev = _row(inc, "Total Revenue", "TotalRevenue", "Operating Revenue")
    gp = _row(inc, "Gross Profit")
    opi = _row(inc, "Operating Income", "EBIT")
    ni = _row(inc, "Net Income", "Net Income Common Stockholders")
    ebitda = _row(inc, "EBITDA", "Normalized EBITDA")
    int_exp = _row(inc, "Interest Expense")
    prev_rev = _row_prev(inc, "Total Revenue", "TotalRevenue", "Operating Revenue")
    prev_ni = _row_prev(inc, "Net Income", "Net Income Common Stockholders")
    prev_gp = _row_prev(inc, "Gross Profit")
    prev_opi = _row_prev(inc, "Operating Income", "EBIT")

    total_debt = _row(bs, "Total Debt")
    if total_debt is None:
        ltd = _row(bs, "Long Term Debt") or 0.0
        std = _row(bs, "Current Debt", "Short Term Debt") or 0.0
        total_debt = (ltd + std) if (ltd or std) else None
    cash = _row(bs, "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments") or 0.0
    equity = _row(bs, "Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity")
    invested = _row(bs, "Invested Capital")
    total_assets = _row(bs, "Total Assets")

    fcf = _row(cf, "Free Cash Flow")
    if fcf is None:
        ocf = _row(cf, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
        capex = _row(cf, "Capital Expenditure")
        if ocf is not None and capex is not None:
            fcf = ocf + capex if capex < 0 else ocf - capex

    out: dict = {
        "earnings_yield": (ni / market_cap) if (ni is not None and market_cap) else None,
        "roic": None,
        "fcf_yield": (fcf / market_cap) if (fcf is not None and market_cap) else None,
        "rev_growth": None,
        "gross_margin": (gp / rev) if (gp is not None and rev) else None,
        "operating_margin": (opi / rev) if (opi is not None and rev) else None,
        "net_margin": (ni / rev) if (ni is not None and rev) else None,
        "fcf_margin": (fcf / rev) if (fcf is not None and rev) else None,
        "rule_of_40": None,
        "mom_52w": None, "mom_ma200": None, "mom_12_1": None,  # app.py fills these
        "safety": None,
        "ps_ratio": (market_cap / rev) if (market_cap and rev and rev > 0) else None,
        "peg": None,
        "eps_growth": None,
        "debt_equity": None,
        "net_debt_ebitda": None,
        "interest_coverage": None,
        "ev_ebit": None,
        "gross_profitability": (gp / total_assets) if (gp is not None and total_assets and total_assets > 0) else None,
        "gross_margin_avg": None, "margin_stability": None, "growth_consistency": None,
        "source": "yahoo",
    }

    # Moat durability — from the multi-year income statement (most-recent-first).
    revs_all = _row_all(inc, "Total Revenue", "TotalRevenue", "Operating Revenue")
    gps_all = _row_all(inc, "Gross Profit")
    ops_all = _row_all(inc, "Operating Income", "EBIT")
    yr_gm = [(g / r if (g is not None and r) else None) for g, r in zip(gps_all, revs_all)]
    yr_om = [(o / r if (o is not None and r) else None) for o, r in zip(ops_all, revs_all)]
    out["gross_margin_avg"], out["margin_stability"], out["growth_consistency"] = \
        moat_metrics(yr_gm, yr_om, revs_all)

    if rev is not None and prev_rev:
        out["rev_growth"] = (rev - prev_rev) / abs(prev_rev)
    if ni is not None and prev_ni and prev_ni > 0:
        out["eps_growth"] = (ni - prev_ni) / prev_ni

    # ROIC ≈ NOPAT / invested capital (21% tax assumed)
    if opi is not None:
        base = invested if invested else (((total_debt or 0.0) + equity - cash) if equity else None)
        if base and base > 0:
            out["roic"] = (opi * 0.79) / base

    # PEG — P/E over EPS-growth%, both must be positive.
    if (market_cap and ni and ni > 0
            and out["eps_growth"] is not None and out["eps_growth"] > 0):
        out["peg"] = (market_cap / ni) / (out["eps_growth"] * 100.0)

    if total_debt is not None and equity and equity > 0:
        out["debt_equity"] = total_debt / equity
    # Edge-handled leverage/coverage (net cash -> good; EBITDA<=0 -> worst; debt-free -> strong coverage)
    out["net_debt_ebitda"] = edge_net_debt_ebitda(total_debt, cash, ebitda)
    out["interest_coverage"] = edge_interest_coverage(opi, int_exp, total_debt)
    out["ev_ebit"] = edge_ev_ebit(market_cap, total_debt, cash, opi)

    if out["rev_growth"] is not None:
        margin = out["fcf_margin"] if out["fcf_margin"] is not None else out["operating_margin"]
        if margin is not None:
            out["rule_of_40"] = (out["rev_growth"] + margin) * 100.0

    # Piotroski-style safety, same checks as the FMP path
    checks = []
    if ni is not None:                       checks.append(ni > 0)
    if out["operating_margin"] is not None:  checks.append(out["operating_margin"] > 0)
    if out["fcf_yield"] is not None:         checks.append(out["fcf_yield"] > 0)
    if out["rev_growth"] is not None:        checks.append(out["rev_growth"] > 0)
    if gp is not None and prev_gp is not None and rev and prev_rev:
        checks.append((gp / rev) >= (prev_gp / prev_rev))
    if opi is not None and prev_opi is not None and rev and prev_rev:
        checks.append((opi / rev) >= (prev_opi / prev_rev))
    if checks:
        out["safety"] = sum(1 for c in checks if c) / len(checks)

    return out
