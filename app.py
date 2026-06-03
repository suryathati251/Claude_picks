"""
Stock Watchlist — Streamlit App  (fundamentals-ranked edition)

Live prices via FMP (stable API). Ranking is now driven by REPORTED FUNDAMENTALS
(earnings yield, ROIC, revenue growth, FCF yield) via fundamentals.py — not by
analyst price targets. Analyst targets are kept only as a reference column, and
are sanity-checked against the live 52-week range so split/stale mismatches
(e.g. a pre-split NFLX target vs a post-split price) get flagged, never ranked.

Cache TTL is 24h so we make at most a few network fetches per ticker per day.
Stale-data fallback: if a fetch fails, we keep showing the last good value.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import logging
import os
from typing import Optional, Tuple

import pandas as pd
import requests
import streamlit as st

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Config (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Watchlist",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Watchlist import — guarded so a bad/stale watchlist_data.py shows a clear
# message instead of a raw traceback.
# ---------------------------------------------------------------------------
try:
    from watchlist_data import WATCHLIST as _BASE_WATCHLIST, SECTOR_ORDER as _BASE_SECTORS
except ImportError:
    try:
        import watchlist_data as _wd
        _found = [n for n in dir(_wd) if not n.startswith("_")]
    except Exception as _e:  # noqa: BLE001
        _found = [f"(watchlist_data.py failed to import: {_e})"]
    st.error(
        "**`watchlist_data.py` is missing `WATCHLIST` / `SECTOR_ORDER`.**\n\n"
        "The file deployed in this repo imports, but doesn't define those names — so it isn't "
        "the full watchlist file (it was likely truncated or a different version got committed).\n\n"
        f"**Names actually found in watchlist_data.py:** `{_found}`\n\n"
        "**Fix:** re-commit the complete `watchlist_data.py` (the one defining `WATCHLIST = [...]` "
        "and `SECTOR_ORDER = [...]`), then redeploy."
    )
    st.stop()

from watchlist_growth import GROWTH_WATCHLIST
from fundamentals import (
    fetch_fundamentals,
    compute_composite_scores,
    target_is_sane,
    FACTORS,
    GROWTH_FACTORS,
)
import market_risk

# Merge the additive growth picks onto the original list (nothing removed).
# Dedupe by ticker in case of overlap, original entry wins.
_seen = {item["ticker"] for item in _BASE_WATCHLIST}
WATCHLIST = _BASE_WATCHLIST + [g for g in GROWTH_WATCHLIST if g["ticker"] not in _seen]
SECTOR_ORDER = _BASE_SECTORS + [s for s in ["Hypergrowth"] if s not in _BASE_SECTORS]

# FMP stable API (legacy /api/v3/ was deprecated for accounts after Aug 31, 2025)
FMP_BASE = "https://financialmodelingprep.com/stable"
CACHE_TTL_SECONDS = 24 * 60 * 60   # 24h — one fetch per ticker per day
STALE_MAX_HOURS = 24 * 7           # keep showing data up to a week if fetches keep failing

def get_api_key() -> Optional[str]:
    try:
        return st.secrets["FMP_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.getenv("FMP_API_KEY")

# ---------------------------------------------------------------------------
# Data fetch with persistent (process-level) last-good cache
# ---------------------------------------------------------------------------
@st.cache_resource
def get_persistent_store() -> dict:
    """Process-level dict: ticker -> {data, fetched_at}. Survives reruns;
    resets on Streamlit Cloud container restart (deploys / idle eviction)."""
    return {}

@st.cache_resource
def get_fundamentals_store() -> dict:
    """Separate process-level cache for fundamentals (slower-changing than price)."""
    return {}

def _fetch_profile_fmp(symbol: str, api_key: str) -> dict:
    """Raw FMP fetch — raises RuntimeError on any failure."""
    url = f"{FMP_BASE}/profile"
    r = requests.get(url, params={"symbol": symbol, "apikey": api_key}, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    try:
        data = r.json()
    except ValueError:
        raise RuntimeError(f"Non-JSON response: {r.text[:200]}")
    if isinstance(data, dict) and "Error Message" in data:
        raise RuntimeError(f"FMP: {data['Error Message'][:300]}")
    if isinstance(data, list) and data:
        record = data[0]
    elif isinstance(data, dict) and data.get("symbol"):
        record = data
    else:
        raise RuntimeError("Empty response")

    return {
        "symbol": record.get("symbol", symbol),
        "price": record.get("price"),
        "changePercentage": record.get("changesPercentage") or record.get("changePercentage"),
        "marketCap": record.get("mktCap") or record.get("marketCap"),
        "currency": record.get("currency", "USD"),
        "range": record.get("range", ""),
    }

def _cached_fetch(store: dict, symbol: str, fetch_fn, force_refresh: bool
                  ) -> Tuple[Optional[dict], Optional[str], Optional[datetime]]:
    """Generic 24h-cache + stale-fallback wrapper around a network fetch_fn(symbol)."""
    now = datetime.now()
    cached = store.get(symbol)
    if cached and not force_refresh:
        age = (now - cached["fetched_at"]).total_seconds()
        if age < CACHE_TTL_SECONDS:
            return cached["data"], None, cached["fetched_at"]
    try:
        data = fetch_fn(symbol)
        store[symbol] = {"data": data, "fetched_at": now}
        return data, None, now
    except Exception as e:
        if cached:
            age_h = (now - cached["fetched_at"]).total_seconds() / 3600
            if age_h < STALE_MAX_HOURS:
                return cached["data"], f"using cache ({str(e)[:80]})", cached["fetched_at"]
        return None, f"no cached value: {str(e)[:200]}", None

def fetch_profile(symbol: str, api_key: str, force_refresh: bool = False):
    return _cached_fetch(get_persistent_store(), symbol,
                         lambda s: _fetch_profile_fmp(s, api_key), force_refresh)

def fetch_fund(symbol: str, api_key: str, force_refresh: bool = False):
    # fetch_fundamentals returns a dict (never raises); treat empty-of-all-None as miss
    def _fn(s):
        d = fetch_fundamentals(s, api_key)
        if all(v is None for v in d.values()):
            raise RuntimeError("no fundamentals fields mapped")
        return d
    return _cached_fetch(get_fundamentals_store(), symbol, _fn, force_refresh)

def fetch_all(api_key: str, force_refresh: bool = False):
    """Fetch profiles + fundamentals concurrently for the whole watchlist."""
    profiles, fundamentals, fetched_at, errors = {}, {}, {}, {}
    fresh, stale, failed = [], [], []

    total = len(WATCHLIST) * 2  # profile + fundamentals per ticker
    progress = st.progress(0.0, text=f"Fetching {len(WATCHLIST)} tickers (price + fundamentals)...")
    completed = 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        prof_futs = {executor.submit(fetch_profile, item["ticker"], api_key, force_refresh): item["ticker"]
                     for item in WATCHLIST}
        fund_futs = {executor.submit(fetch_fund, item["ticker"], api_key, force_refresh): item["ticker"]
                     for item in WATCHLIST}

        for future in as_completed(list(prof_futs) + list(fund_futs)):
            is_profile = future in prof_futs
            symbol = prof_futs[future] if is_profile else fund_futs[future]
            try:
                data, warn, fa = future.result()
            except Exception as e:
                data, warn, fa = None, f"thread error: {e}", None

            if is_profile:
                if data is not None:
                    profiles[symbol] = data
                    fetched_at[symbol] = fa
                    (stale if warn else fresh).append(symbol)
                    if warn:
                        errors[symbol] = warn
                else:
                    failed.append(symbol)
                    errors[symbol] = warn or "unknown error"
            else:
                fundamentals[symbol] = data or {}

            completed += 1
            progress.progress(completed / total, text=f"{completed}/{total} fetched")
    progress.empty()
    return profiles, fundamentals, fetched_at, fresh, stale, failed, errors

# ---------------------------------------------------------------------------
# Build dataframe / formatting
# ---------------------------------------------------------------------------
CURRENCY_SYMBOL = {"USD": "$", "INR": "₹", "JPY": "¥", "EUR": "€", "GBP": "£", "CAD": "C$"}

def fmt_price(v, currency="USD"):
    if v is None or pd.isna(v):
        return "—"
    sym = CURRENCY_SYMBOL.get(currency, currency + " ")
    return f"{sym}{v:,.2f}"

def fmt_mcap(v, currency="USD"):
    if v is None or pd.isna(v):
        return "—"
    sym = CURRENCY_SYMBOL.get(currency, currency + " ")
    if v >= 1e12: return f"{sym}{v/1e12:.2f}T"
    if v >= 1e9:  return f"{sym}{v/1e9:.2f}B"
    if v >= 1e6:  return f"{sym}{v/1e6:.0f}M"
    return f"{sym}{v:,.0f}"

def build_dataframe(profiles: dict, fundamentals: dict, scores: dict, growth_scores: dict) -> pd.DataFrame:
    rows = []
    for item in WATCHLIST:
        tk = item["ticker"]
        p = profiles.get(tk) or {}
        price = p.get("price")
        day_pct = p.get("changePercentage")
        mcap = p.get("marketCap")
        currency = p.get("currency", "USD")
        range_str = p.get("range", "")
        try:
            lo, hi = [float(x.strip()) for x in range_str.split("-")]
        except Exception:
            lo, hi = None, None

        target = item.get("target")
        # Sanity guard: a static target far outside the live 52w range is almost
        # always a split / stale-data artifact (the NFLX 358-vs-83 bug). Flag it
        # and DON'T compute an upside that would otherwise dominate the board.
        target_ok = target_is_sane(target, lo, hi)
        upside = None
        if price is not None and target and target_ok:
            upside = ((target - price) / price) * 100

        sc = scores.get(tk) or {}
        gsc = growth_scores.get(tk) or {}
        fund = fundamentals.get(tk) or {}

        rows.append({
            "Ticker": tk,
            "Region": item["region"],
            "Name": item["name"],
            "Sector": item["sector"],
            "Price": price,
            "Day %": day_pct,
            "52w Low": lo,
            "52w High": hi,
            "52w Pos %": ((price - lo) / (hi - lo) * 100) if (price and lo and hi and hi > lo) else None,
            "Mkt Cap": mcap,
            "Currency": currency,
            # --- fundamentals-first columns ---
            "Score": (sc.get("score") * 100) if sc.get("score") is not None else None,
            "Coverage": sc.get("coverage", 0),
            "Growth Score": (gsc.get("score") * 100) if gsc.get("score") is not None else None,
            "Growth Cov": gsc.get("coverage", 0),
            "Earnings Yld %": (fund.get("earnings_yield") * 100) if fund.get("earnings_yield") is not None else None,
            "ROIC %": (fund.get("roic") * 100) if fund.get("roic") is not None else None,
            "Rev Growth %": (fund.get("rev_growth") * 100) if fund.get("rev_growth") is not None else None,
            "Gross Mgn %": (fund.get("gross_margin") * 100) if fund.get("gross_margin") is not None else None,
            "Rule40": fund.get("rule_of_40"),
            "FCF Yld %": (fund.get("fcf_yield") * 100) if fund.get("fcf_yield") is not None else None,
            # --- analyst target kept only as reference ---
            "Target": target,
            "Target OK": target_ok,
            "Upside %": upside,
            "Thesis": item["thesis"],
            "Catalyst": item["catalyst"],
            "Risk": item["risk"],
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("📈 Stock Watchlist")
st.caption(
    f"{len(WATCHLIST)} tickers across {len(SECTOR_ORDER)} sectors. Ranked by **reported "
    f"fundamentals** (earnings yield · ROIC · revenue growth · FCF yield), not analyst targets. "
    f"Live data from FMP (cached 24h). Targets/thesis are static reference — edit `watchlist_data.py`."
)

api_key = get_api_key()
if not api_key:
    st.error(
        "Missing FMP API key. Set `FMP_API_KEY` in Streamlit secrets (production) or as "
        "an environment variable (local). See README.md."
    )
    st.stop()

if "force_refresh" not in st.session_state:
    st.session_state.force_refresh = False

col_a, col_b = st.columns([1, 5])
with col_a:
    if st.button("🔄 Force refresh", use_container_width=True,
                 help="Bypass the 24h cache and refetch all tickers. Costs ~3 FMP requests per ticker."):
        st.session_state.force_refresh = True
        st.rerun()
with col_b:
    st.caption(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
               f"cache: {CACHE_TTL_SECONDS // 3600}h · stale fallback: {STALE_MAX_HOURS // 24}d")

force = st.session_state.force_refresh
st.session_state.force_refresh = False

profiles, fundamentals, fetched_at, fresh, stale, failed, errors = fetch_all(api_key, force_refresh=force)
scores = compute_composite_scores(fundamentals, FACTORS)
growth_scores = compute_composite_scores(fundamentals, GROWTH_FACTORS)

# ---------------------------------------------------------------------------
# Market-conditions dashboard — CONTEXT, not buy/sell signals.
# ---------------------------------------------------------------------------
mkt = market_risk.get_market_context(api_key)
with st.container():
    st.subheader("🌡️ Market conditions")
    st.caption(
        "Context for **how** to deploy (position size · scaling in/out · rebalancing) — "
        "**not** a buy/sell signal. Timing tops and bottoms reliably isn't possible; this just "
        "describes the environment you're investing into."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("S&P 500", f"{mkt['index_price']:,.0f}" if mkt["index_price"] else "—",
              mkt["trend"] if mkt["trend"] != "unknown" else None)
    c2.metric("vs 200-day avg",
              f"{mkt['pct_from_ma200']:+.1f}%" if mkt["pct_from_ma200"] is not None else "—")
    c3.metric("Index 52w position",
              f"{mkt['index_52w_pos']:.0f}%" if mkt["index_52w_pos"] is not None else "—")
    c4.metric("VIX (volatility)",
              f"{mkt['vix']:.1f}" if mkt["vix"] else "—",
              mkt["vix_label"] if mkt["vix_label"] != "unknown" else None)
    if mkt["vix_context"]:
        st.caption(f"**Volatility regime — {mkt['vix_label']}:** {mkt['vix_context']}")
    for note in mkt["notes"]:
        st.info(note)
st.divider()

# Status banner
if stale or failed:
    parts = []
    if fresh: parts.append(f"✅ {len(fresh)} fresh")
    if stale: parts.append(f"♻️ {len(stale)} stale (showing cached values)")
    if failed: parts.append(f"❌ {len(failed)} failed (no data)")
    st.warning(" · ".join(parts))
    with st.expander("🔍 Show per-ticker fetch details"):
        if stale:
            st.markdown("**Showing cached/stale data for:**")
            for sym in stale:
                age = ""
                if sym in fetched_at and fetched_at[sym]:
                    mins = int((datetime.now() - fetched_at[sym]).total_seconds() / 60)
                    age = f" (last good fetch: {mins} min ago)"
                st.code(f"{sym}: {errors.get(sym, 'using cached')}{age}", language=None)
        if failed:
            st.markdown("**No data available for:**")
            for sym in failed:
                st.code(f"{sym}: {errors.get(sym, 'unknown error')}", language=None)
elif fresh:
    st.success(f"✅ All {len(fresh)} tickers loaded (some may be from today's cached values).")

df = build_dataframe(profiles, fundamentals, scores, growth_scores)
df["Stale"] = df["Ticker"].isin(set(stale))

# Flag any target the sanity guard rejected (the split/stale bug catcher)
flagged = df[(df["Target"].notna()) & (~df["Target OK"])]
if len(flagged):
    names = ", ".join(f"{r.Ticker} (target {r.Target:g} vs 52w high {r._asdict()['52w High']:g})"
                      for r in flagged.itertuples())
    st.error(
        f"⚠️ {len(flagged)} static target(s) look implausible vs the live 52-week range and were "
        f"excluded from upside — likely a stock split or stale entry in `watchlist_data.py`: {names}"
    )

# Summary metrics — now fundamentals-driven
loaded = df["Price"].notna().sum()
scored = df[df["Score"].notna()]
avg_score = scored["Score"].mean() if len(scored) else 0
top_row = scored.sort_values("Score", ascending=False).head(1)
top_name = top_row["Ticker"].iloc[0] if len(top_row) else "—"
top_score = top_row["Score"].iloc[0] if len(top_row) else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Tickers loaded", f"{loaded}/{len(WATCHLIST)}")
m2.metric("With fundamentals", len(scored))
m3.metric("Avg fundamental score", f"{avg_score:.0f}/100")
m4.metric("Top score", f"{top_name} {top_score:.0f}")

st.divider()

available_sectors = [s for s in SECTOR_ORDER if s in df["Sector"].unique()]
fcol1, fcol2 = st.columns([2, 3])
with fcol1:
    rank_mode = st.radio(
        "Rank by",
        options=["Value / Quality", "Growth / Asymmetric", "Blended"],
        horizontal=True,
        help="Value = cheap + high-return-on-capital (avoids value traps). "
             "Growth = hypergrowth characteristics (higher risk; most won't 10x). "
             "Blended = average of both.",
    )
with fcol2:
    selected_sectors = st.multiselect(
        "Filter by sector",
        options=available_sectors,
        default=available_sectors,
        label_visibility="collapsed",
        placeholder="Filter by sector...",
    )

view = df[df["Sector"].isin(selected_sectors)].copy() if selected_sectors else df.copy()

# Blended score for sorting / display
view["Blend"] = view[["Score", "Growth Score"]].mean(axis=1, skipna=True)
sort_key = {"Value / Quality": "Score", "Growth / Asymmetric": "Growth Score", "Blended": "Blend"}[rank_mode]
view = view.sort_values(sort_key, ascending=False, na_position="last")

def render_display(row):
    stale_mark = " ♻️" if row.get("Stale") else ""
    target_str = fmt_price(row["Target"], row["Currency"]) if pd.notna(row["Target"]) else "—"
    if pd.notna(row["Target"]) and not row["Target OK"]:
        target_str += " ⚠️"
    return pd.Series({
        "Ticker": f"{row['Ticker']}{stale_mark}  ({row['Region']})",
        "Name": row["Name"],
        "Sector": row["Sector"],
        "Value": f"{row['Score']:.0f}" if pd.notna(row["Score"]) else "—",
        "Growth": f"{row['Growth Score']:.0f}" if pd.notna(row["Growth Score"]) else "—",
        "Cov": f"{int(row['Coverage'])}/{len(FACTORS)}" if pd.notna(row["Coverage"]) else "—",
        "Price": fmt_price(row["Price"], row["Currency"]),
        "Day %": f"{row['Day %']:+.2f}%" if pd.notna(row["Day %"]) else "—",
        "Earn Yld": f"{row['Earnings Yld %']:.1f}%" if pd.notna(row["Earnings Yld %"]) else "—",
        "ROIC": f"{row['ROIC %']:.1f}%" if pd.notna(row["ROIC %"]) else "—",
        "Rev Grw": f"{row['Rev Growth %']:+.1f}%" if pd.notna(row["Rev Growth %"]) else "—",
        "Gross Mgn": f"{row['Gross Mgn %']:.0f}%" if pd.notna(row["Gross Mgn %"]) else "—",
        "Rule40": f"{row['Rule40']:.0f}" if pd.notna(row["Rule40"]) else "—",
        "FCF Yld": f"{row['FCF Yld %']:.1f}%" if pd.notna(row["FCF Yld %"]) else "—",
        "Mkt Cap": fmt_mcap(row["Mkt Cap"], row["Currency"]),
        "Target": target_str,
        "Upside %": f"{row['Upside %']:+.1f}%" if pd.notna(row["Upside %"]) else "—",
        "Thesis": row["Thesis"],
    })

display = view.apply(render_display, axis=1)

def color_score(val: str) -> str:
    if val == "—": return "color: #6b7280;"
    try:
        n = float(val)
    except ValueError:
        return ""
    if n >= 70: return "color: #047857; font-weight: 600;"
    if n >= 45: return "color: #0369a1;"
    return "color: #6b7280;"

def color_day(val: str) -> str:
    if val == "—": return "color: #6b7280;"
    try:
        n = float(val.replace("%", "").replace("+", ""))
    except ValueError:
        return ""
    return "color: #059669;" if n >= 0 else "color: #dc2626;"

def color_growth(val: str) -> str:
    if val == "—": return "color: #6b7280;"
    try:
        n = float(val.replace("%", "").replace("+", ""))
    except ValueError:
        return ""
    return "color: #059669;" if n >= 0 else "color: #dc2626;"

styler = (
    display.style
    .map(color_score, subset=["Value"])
    .map(color_score, subset=["Growth"])
    .map(color_day, subset=["Day %"])
    .map(color_growth, subset=["Rev Grw"])
)

st.dataframe(
    styler,
    use_container_width=True,
    hide_index=True,
    height=min(60 + 36 * len(display), 900),
    column_config={
        "Thesis": st.column_config.TextColumn(width="large"),
        "Name": st.column_config.TextColumn(width="medium"),
    },
)

st.divider()
csv = view.to_csv(index=False).encode()
st.download_button(
    label="📥 Download current view as CSV",
    data=csv,
    file_name=f"watchlist_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv",
)

with st.expander("ℹ️ Notes & caveats", expanded=False):
    st.markdown(
        f"""
- **Not financial advice.** Ranking reflects reported fundamentals, which describe the past;
  they don't predict returns. A high score is a starting point for research, not a buy signal.
- **Two scores, pick your lens.** *Value/Quality* = cheap + high return-on-capital (Greenblatt-style;
  avoids value traps). *Growth/Asymmetric* = hypergrowth characteristics (revenue growth, gross
  margin, Rule-of-40). **The growth score does NOT find "the next Nvidia."** Nothing reliably does —
  for every 40-bagger, hundreds of look-alikes failed, and early Nvidia would have *failed* the value
  screen. Use the growth list as a small, diversified *basket* of candidates to research, sized so any
  single one going to zero doesn't hurt you. That's how asymmetric bets work when you can't predict the winner.
- **Market conditions ≠ timing.** The dashboard describes the environment (trend regime, volatility,
  how stretched the index is). It is deliberately NOT a buy/sell signal, because reliably timing tops
  and bottoms doesn't work. Use it to calibrate process — position size, scaling in over tranches,
  rebalance bands — never as a trigger.
- **Data source:** Financial Modeling Prep (stable API). **Field names vary by plan** — run once
  and check the log for `no field mapped for ...` warnings, then adjust the candidate lists in
  `fundamentals.py`.
- **API budget:** ~3 calls/ticker (profile + 2 fundamentals) = ~{len(WATCHLIST) * 3} on a cold
  start, vs the free tier's ~250/day. The 24h cache keeps you under it; **force refresh** spends
  the full ~{len(WATCHLIST) * 3} at once, so use sparingly.
- **Stale fallback:** failed fetches show the last good value (♻️) up to {STALE_MAX_HOURS // 24} days old.
"""
    )

