"""
insider.py — insider-transaction signal for the watchlist.

Why: officers/directors buying their own stock with their own money — especially
several of them, near a 52-week low — is one of the few public "smart money"
tells. It pairs naturally with a buy-fear discipline. (Sells are far less
informative: insiders sell for taxes, diversification, houses.)

Data: FMP insider-trading endpoints. These cost 1 call per ticker and may not be
included in every plan, so app.py fetches them with a small separate budget,
caches results 14 days, and this module detects an unsupported plan (402/403 or
a "premium" error message) so the UI can say so instead of showing blanks.

Pure module: no Streamlit, no caching here. 429 raises FMPRateLimitError so the
caller can stop early, same convention as fundamentals.py.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import requests

from fundamentals import FMPRateLimitError

logger = logging.getLogger("insider")

FMP_STABLE = "https://financialmodelingprep.com/stable"
FMP_V4 = "https://financialmodelingprep.com/api/v4"
_TIMEOUT = 12
WINDOW_DAYS = 90


def _looks_rate_limited(text: str) -> bool:
    t = (text or "").lower()
    return ("limit reach" in t or "too many requests" in t or "bandwidth" in t)


def _looks_unsupported(status: int, text: str) -> bool:
    t = (text or "").lower()
    return status in (402, 403) or "premium" in t or "not available under your" in t \
        or "upgrade your plan" in t or "exclusive endpoint" in t


def _get_rows(url: str, params: dict, symbol: str):
    """Returns (rows|None, unsupported: bool). Raises FMPRateLimitError on 429."""
    try:
        r = requests.get(url, params=params, timeout=_TIMEOUT)
    except Exception as e:  # noqa: BLE001
        logger.warning("%s insider request error: %s", symbol, str(e)[:120])
        return None, False
    if r.status_code == 429 or _looks_rate_limited(r.text):
        raise FMPRateLimitError(f"{symbol}/insider: {r.text[:100]}")
    if _looks_unsupported(r.status_code, r.text):
        return None, True
    if r.status_code != 200:
        return None, False
    try:
        data = r.json()
    except ValueError:
        return None, False
    if isinstance(data, dict):
        msg = str(data.get("Error Message") or data.get("Error") or "")
        if msg:
            if _looks_rate_limited(msg):
                raise FMPRateLimitError(f"{symbol}/insider: {msg[:100]}")
            return None, _looks_unsupported(200, msg)
        data = [data] if data else []
    return (data if isinstance(data, list) else []), False


def fetch_insider(symbol: str, api_key: str) -> dict:
    """Summarize the last ~90 days of insider transactions for one ticker.

    Returns {"buys", "sells", "buy_val", "sell_val", "unsupported"} — counts of
    open-market purchases (P) vs sales (S); awards/gifts/options ignored.
    All-zero with unsupported=False just means no recent insider activity.
    """
    out = {"buys": 0, "sells": 0, "buy_val": 0.0, "sell_val": 0.0, "unsupported": False}
    rows, unsupported = _get_rows(f"{FMP_STABLE}/insider-trading/search",
                                  {"symbol": symbol, "page": 0, "limit": 100,
                                   "apikey": api_key}, symbol)
    if rows is None and not unsupported:   # stable endpoint missing -> legacy v4
        rows, unsupported = _get_rows(f"{FMP_V4}/insider-trading",
                                      {"symbol": symbol, "page": 0, "apikey": api_key}, symbol)
    if unsupported:
        out["unsupported"] = True
        return out
    if not rows:
        return out

    cutoff = (date.today() - timedelta(days=WINDOW_DAYS)).isoformat()
    for row in rows:
        d = str(row.get("transactionDate") or "")[:10]
        if d and d < cutoff:
            continue
        ttype = str(row.get("transactionType") or "").upper()
        acq = str(row.get("acquisitionOrDisposition") or "").upper()
        try:
            val = float(row.get("securitiesTransacted") or 0) * float(row.get("price") or 0)
        except (TypeError, ValueError):
            val = 0.0
        # Open-market purchase / sale codes: "P-Purchase" / "S-Sale".
        if ttype.startswith("P"):
            out["buys"] += 1
            out["buy_val"] += val
        elif ttype.startswith("S") and acq != "A":
            out["sells"] += 1
            out["sell_val"] += val
    return out


def insider_display(d: Optional[dict], pos_52w: Optional[float]) -> str:
    """Compact table cell, e.g. '🟢 3B/0S' · '2B/5S' · '—'.
    🟢 = net buying; 🟣 = cluster buying (3+ buys, no sells) near the 52w low —
    the classic contrarian tell."""
    if not d or d.get("unsupported"):
        return "—"
    b, s = d.get("buys", 0), d.get("sells", 0)
    if b == 0 and s == 0:
        return "0B/0S"
    cell = f"{b}B/{s}S"
    if b >= 3 and s == 0 and pos_52w is not None and pos_52w < 25:
        return f"🟣 {cell}"
    if b > s and b >= 2:
        return f"🟢 {cell}"
    return cell
