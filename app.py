"""
Stock Watchlist — Streamlit App
Live prices via FMP (stable API). Cache TTL is 24h so we make at most one
network fetch per ticker per day, well under FMP free tier's 250/day limit.
Stale-data fallback: if a fetch fails, we keep showing the last good value.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import os
from typing import Optional, Tuple

import pandas as pd
import requests
import streamlit as st

from watchlist_data import WATCHLIST, SECTOR_ORDER

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Watchlist",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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

def fetch_profile(symbol: str, api_key: str, force_refresh: bool = False) -> Tuple[Optional[dict], Optional[str], Optional[datetime]]:
    """
    Returns (data, warning_or_error, fetched_at).
    - data: profile dict (may be stale fallback)
    - warning_or_error: None on fresh success; string when stale or missing
    - fetched_at: original fetch timestamp of the returned data
    """
    store = get_persistent_store()
    now = datetime.now()

    cached = store.get(symbol)
    if cached and not force_refresh:
        age = (now - cached["fetched_at"]).total_seconds()
        if age < CACHE_TTL_SECONDS:
            return cached["data"], None, cached["fetched_at"]

    try:
        data = _fetch_profile_fmp(symbol, api_key)
        store[symbol] = {"data": data, "fetched_at": now}
        return data, None, now
    except Exception as e:
        if cached:
            age_h = (now - cached["fetched_at"]).total_seconds() / 3600
            if age_h < STALE_MAX_HOURS:
                return cached["data"], f"using cache ({str(e)[:80]})", cached["fetched_at"]
        return None, f"no cached value: {str(e)[:200]}", None

def fetch_all_prices(api_key: str, force_refresh: bool = False):
    profiles, fetched_at, errors = {}, {}, {}
    fresh, stale, failed = [], [], []

    progress = st.progress(0.0, text=f"Fetching prices for {len(WATCHLIST)} tickers...")
    completed = 0
    # 8 concurrent FMP calls — well within their per-second limits
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_profile, item["ticker"], api_key, force_refresh): item["ticker"]
                   for item in WATCHLIST}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                data, warn, fa = future.result()
            except Exception as e:
                data, warn, fa = None, f"thread error: {e}", None
            if data is not None:
                profiles[symbol] = data
                fetched_at[symbol] = fa
                if warn:
                    stale.append(symbol)
                    errors[symbol] = warn
                else:
                    fresh.append(symbol)
            else:
                failed.append(symbol)
                errors[symbol] = warn or "unknown error"
            completed += 1
            progress.progress(completed / len(WATCHLIST), text=f"{completed}/{len(WATCHLIST)} fetched")
    progress.empty()
    return profiles, fetched_at, fresh, stale, failed, errors

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

def build_dataframe(profiles: dict) -> pd.DataFrame:
    rows = []
    for item in WATCHLIST:
        p = profiles.get(item["ticker"]) or {}
        price = p.get("price")
        day_pct = p.get("changePercentage")
        mcap = p.get("marketCap")
        currency = p.get("currency", "USD")
        range_str = p.get("range", "")
        try:
            lo, hi = [float(x.strip()) for x in range_str.split("-")]
        except Exception:
            lo, hi = None, None

        upside = None
        if price is not None and item.get("target"):
            upside = ((item["target"] - price) / price) * 100

        rows.append({
            "Ticker": item["ticker"],
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
            "Target": item.get("target"),
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
    f"{len(WATCHLIST)} tickers across {len(SECTOR_ORDER)} sectors. "
    f"Live prices from FMP (cached 24h). Analyst targets and thesis are static — edit `watchlist_data.py`."
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
                 help="Bypass the 24h cache and fetch all tickers fresh. Costs ~74 FMP requests."):
        st.session_state.force_refresh = True
        st.rerun()
with col_b:
    st.caption(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
               f"cache: {CACHE_TTL_SECONDS // 3600}h · stale fallback: {STALE_MAX_HOURS // 24}d")

force = st.session_state.force_refresh
st.session_state.force_refresh = False

profiles, fetched_at, fresh, stale, failed, errors = fetch_all_prices(api_key, force_refresh=force)

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

df = build_dataframe(profiles)
stale_set = set(stale)
df["Stale"] = df["Ticker"].isin(stale_set)

# Summary metrics
loaded = df["Price"].notna().sum()
named_targets = df[df["Target"].notna() & df["Price"].notna()]
avg_upside = named_targets["Upside %"].mean() if len(named_targets) else 0
top_row = named_targets.sort_values("Upside %", ascending=False).head(1)
top_name = top_row["Ticker"].iloc[0] if len(top_row) else "—"
top_pct = top_row["Upside %"].iloc[0] if len(top_row) else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Tickers loaded", f"{loaded}/{len(WATCHLIST)}")
m2.metric("With targets", len(named_targets))
m3.metric("Avg upside (vs target)", f"{avg_upside:+.1f}%")
m4.metric("Top upside", f"{top_name} {top_pct:+.1f}%")

st.divider()

available_sectors = [s for s in SECTOR_ORDER if s in df["Sector"].unique()]
selected_sectors = st.multiselect(
    "Filter by sector",
    options=available_sectors,
    default=available_sectors,
    label_visibility="collapsed",
    placeholder="Filter by sector...",
)

view = df[df["Sector"].isin(selected_sectors)].copy() if selected_sectors else df.copy()
view = view.sort_values("Upside %", ascending=False, na_position="last")

def render_display(row):
    stale_mark = " ♻️" if row.get("Stale") else ""
    return pd.Series({
        "Ticker": f"{row['Ticker']}{stale_mark}  ({row['Region']})",
        "Name": row["Name"],
        "Sector": row["Sector"],
        "Price": fmt_price(row["Price"], row["Currency"]),
        "Day %": f"{row['Day %']:+.2f}%" if pd.notna(row["Day %"]) else "—",
        "52w Range": (
            f"{fmt_price(row['52w Low'], row['Currency'])} – {fmt_price(row['52w High'], row['Currency'])}"
            if pd.notna(row["52w Low"]) and pd.notna(row["52w High"]) else "—"
        ),
        "52w Pos": f"{row['52w Pos %']:.0f}%" if pd.notna(row["52w Pos %"]) else "—",
        "Mkt Cap": fmt_mcap(row["Mkt Cap"], row["Currency"]),
        "Target": fmt_price(row["Target"], row["Currency"]) if pd.notna(row["Target"]) else "—",
        "Upside %": f"{row['Upside %']:+.1f}%" if pd.notna(row["Upside %"]) else "—",
        "Thesis": row["Thesis"],
        "Catalyst": row["Catalyst"],
    })

display = view.apply(render_display, axis=1)

def color_upside(val: str) -> str:
    if val == "—": return "color: #6b7280;"
    try:
        n = float(val.replace("%", "").replace("+", ""))
    except ValueError:
        return ""
    if n < 0:   return "color: #dc2626;"
    if n >= 40: return "color: #047857; font-weight: 600;"
    if n >= 15: return "color: #0369a1;"
    return "color: #6b7280;"

def color_day(val: str) -> str:
    if val == "—": return "color: #6b7280;"
    try:
        n = float(val.replace("%", "").replace("+", ""))
    except ValueError:
        return ""
    return "color: #059669;" if n >= 0 else "color: #dc2626;"

styler = (
    display.style
    .map(color_upside, subset=["Upside %"])
    .map(color_day, subset=["Day %"])
)

st.dataframe(
    styler,
    use_container_width=True,
    hide_index=True,
    height=min(60 + 36 * len(display), 900),
    column_config={
        "Thesis": st.column_config.TextColumn(width="large"),
        "Catalyst": st.column_config.TextColumn(width="medium"),
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
- **Not financial advice.** Analyst targets are baked in from public sources at the time
  `watchlist_data.py` was last edited; they don't auto-refresh.
- **Data source:** Financial Modeling Prep (stable API). Free tier ≈ 250 requests/day.
- **Cache strategy:** {CACHE_TTL_SECONDS // 3600}h TTL. With {len(WATCHLIST)} tickers, a cold start costs
  {len(WATCHLIST)} requests — well under the daily limit. The cache lives in the Streamlit Cloud
  container's memory and resets on deploys / idle eviction.
- **Force refresh** bypasses the cache and refetches everything (costs {len(WATCHLIST)} requests).
  Use sparingly.
- **Stale fallback:** if a fetch fails, we show the last good value (♻️) up to {STALE_MAX_HOURS // 24} days old
  instead of blanking the row.
- **Editing the watchlist:** edit `watchlist_data.py`, commit + push.
- **Negative upside** means the price has run past the static target — re-research before acting.
"""
    )
