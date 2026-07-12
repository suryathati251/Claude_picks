"""
digest.py — morning digest for GitHub Actions (no Streamlit).

Runs standalone on a schedule (.github/workflows/daily-digest.yml), builds a
short report and delivers it:
  • S&P 500 fear/greed entry meter reading + stance (buy fear, not greed)
  • 10x Radar top 10 — exploding/accelerating quarterly revenues

Delivery: Telegram when TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID secrets are set;
otherwise the digest just prints to the Actions log (still useful — the run
history becomes a daily archive).

Env:
  FMP_API_KEY        optional — enables market context + market caps
  TELEGRAM_BOT_TOKEN optional — @BotFather token
  TELEGRAM_CHAT_ID   optional — your chat id (message @userinfobot)
  DIGEST_UNIVERSE    "watchlist" (default, ~120 tickers, ~2 min) or "full"
                     (adds the S&P 500 / Nasdaq-100 scan, ~10 min)

Not investment advice; screens reported data only.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests

import market_risk
from entry_meter import fetch_entry_history_yahoo, compute_entry_meter
from tenx_radar import fetch_quarterly_yahoo, compute_tenx_metrics, tenx_score
from watchlist_data import WATCHLIST
from watchlist_growth import GROWTH_WATCHLIST

FMP_BASE = "https://financialmodelingprep.com/stable"


def build_universe() -> tuple[list[str], dict[str, str]]:
    seen = {it["ticker"] for it in WATCHLIST}
    items = WATCHLIST + [g for g in GROWTH_WATCHLIST if g["ticker"] not in seen]
    syms = [it["ticker"] for it in items]
    names = {it["ticker"]: it["name"] for it in items}
    if os.getenv("DIGEST_UNIVERSE", "watchlist").lower() == "full":
        from tenx_universe import SCAN_SYMBOLS, SCAN_NAMES
        for s in SCAN_SYMBOLS:
            if s not in names:
                syms.append(s)
                names[s] = SCAN_NAMES.get(s, s)
    return syms, names


def fetch_mcaps(symbols: list[str], api_key: str | None) -> dict[str, float]:
    """Market caps via FMP batch quotes (cheap); {} without a key."""
    if not api_key:
        return {}
    out: dict[str, float] = {}
    for i in range(0, len(symbols), 50):
        chunk = symbols[i:i + 50]
        try:
            r = requests.get(f"{FMP_BASE}/batch-quote",
                             params={"symbols": ",".join(chunk), "apikey": api_key}, timeout=25)
            if r.status_code != 200:
                continue
            for row in (r.json() if isinstance(r.json(), list) else []):
                mc = row.get("marketCap") or row.get("mktCap")
                if row.get("symbol") and mc:
                    out[row["symbol"]] = float(mc)
        except Exception:  # noqa: BLE001
            continue
    return out


def radar_top(symbols: list[str], names: dict[str, str],
              mcaps: dict[str, float], n: int = 10) -> list[str]:
    scored = []

    def work(sym):
        return sym, compute_tenx_metrics(fetch_quarterly_yahoo(sym))

    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(work, s) for s in symbols]
        for fut in as_completed(futs):
            sym, m = fut.result()
            score, _, tags = tenx_score(m, mcaps.get(sym))
            if score is None:
                continue
            scored.append((score, sym, m, tags))

    scored.sort(reverse=True)
    lines = []
    for score, sym, m, tags in scored[:n]:
        yoy = m.get("q_rev_yoy")
        accel = m.get("rev_accel") if m.get("rev_accel") is not None else m.get("seq_accel")
        bits = [f"{score:.0f}", f"rev {yoy*100:+.0f}% YoY" if yoy is not None else ""]
        if accel is not None:
            bits.append(f"accel {accel*100:+.0f}pt")
        warn = " ⚠️" if any("tiny" in t for t in tags) else ""
        lines.append(f"{sym} ({names.get(sym, sym)}): " + " · ".join(b for b in bits if b) + warn)
    return lines


def main() -> None:
    api_key = os.getenv("FMP_API_KEY")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    mkt = market_risk.get_market_context(api_key) if api_key else {}
    hist = fetch_entry_history_yahoo()
    meter = compute_entry_meter(mkt or {}, None,
                                hist.get("spx_rsi14"), hist.get("vix_pctile_1y"))

    parts = [f"📈 NextPicks digest — {now}", ""]
    if meter["score"] is not None:
        parts += [f"🎯 S&P 500 entry meter: {meter['score']:.0f}/100 · {meter['zone']}",
                  f"(0 = extreme fear = better entries · 100 = extreme greed)",
                  meter["stance"], ""]
    else:
        parts += ["🎯 Entry meter: no market data this run.", ""]

    symbols, names = build_universe()
    mcaps = fetch_mcaps(symbols, api_key)
    top = radar_top(symbols, names, mcaps)
    if top:
        parts.append("🚀 10x Radar top 10 (exploding quarterly revenues):")
        parts += [f"{i+1:>2}. {line}" for i, line in enumerate(top)]
    else:
        parts.append("🚀 10x Radar: no quarterly data retrieved this run.")
    parts += ["", "Reported data only — research candidates, not advice."]

    text = "\n".join(parts)
    print(text)

    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if token and chat:
        try:
            r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                              json={"chat_id": chat, "text": text}, timeout=15)
            print(f"\n[telegram] status {r.status_code}")
        except Exception as e:  # noqa: BLE001
            print(f"\n[telegram] send failed: {e}")
    else:
        print("\n[telegram] secrets not set — digest printed to log only.")


if __name__ == "__main__":
    main()
