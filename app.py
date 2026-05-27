"""
Stock Watchlist — Streamlit App
Fetches live prices from FMP on each page load; thesis/targets are static (edit watchlist_data.py).
Deploy to Streamlit Community Cloud (streamlit.io). See README.md for setup.
"""
import os
from datetime import datetime
from typing import Optional

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

FMP_BASE = "https://financialmodelingprep.com/api/v3"

def get_api_key() -> Optional[str]:
    # Try Streamlit secrets first (production), then env var (local dev)
    try:
        return st.secrets["FMP_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.getenv("FMP_API_KEY")

# ---------------------------------------------------------------------------
# Data fetch (cached for 15 minutes to stay friendly to FMP free tier)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60 * 15, show_spinner=False)
def fetch_profile(symbol: str, api_key: str) -> Optional[dict]:
    try:
        url = f"{FMP_BASE}/profile/{symbol}"
        r = requests.get(url, params={"apikey": api_key}, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if isinstance(data, list) and data:
            return data[0]
        if isinstance(data, dict) and data.get("symbol"):
            return data
        return None
    except Exception:
        return None

def fetch_all_prices(api_key: str) -> dict:
    out = {}
    progress = st.progress(0.0, text=f"Fetching live prices for {len(WATCHLIST)} tickers...")
    for i, item in enumerate(WATCHLIST):
        out[item["ticker"]] = fetch_profile(item["ticker"], api_key)
        progress.progress((i + 1) / len(WATCHLIST), text=f"{i+1}/{len(WATCHLIST)} fetched")
    progress.empty()
    return out

# ---------------------------------------------------------------------------
# Build dataframe
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
        day_pct = p.get("changesPercentage") or p.get("changePercentage")
        mcap = p.get("mktCap") or p.get("marketCap")
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
    "High-upside / reasonably-valued candidates across AI-semis, biotech, energy, financials, "
    "consumer, and international. Live prices from FMP; analyst targets and thesis are static "
    "(edit `watchlist_data.py` to update)."
)

api_key = get_api_key()
if not api_key:
    st.error(
        "Missing FMP API key. Set `FMP_API_KEY` in Streamlit secrets (production) or as "
        "an environment variable (local). See README.md."
    )
    st.stop()

# Refresh control
col_a, col_b, col_c = st.columns([1, 1, 4])
with col_a:
    if st.button("🔄 Refresh prices", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
with col_b:
    st.caption(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

profiles = fetch_all_prices(api_key)
df = build_dataframe(profiles)

# Summary metrics
loaded = df["Price"].notna().sum()
named_targets = df[df["Target"].notna() & df["Price"].notna()]
avg_upside = named_targets["Upside %"].mean() if len(named_targets) else 0
top_upside_row = named_targets.sort_values("Upside %", ascending=False).head(1)
top_upside_name = top_upside_row["Ticker"].iloc[0] if len(top_upside_row) else "—"
top_upside_pct = top_upside_row["Upside %"].iloc[0] if len(top_upside_row) else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Tickers loaded", f"{loaded}/{len(WATCHLIST)}")
m2.metric("With targets", len(named_targets))
m3.metric("Avg upside (vs target)", f"{avg_upside:+.1f}%")
m4.metric("Top upside", f"{top_upside_name} {top_upside_pct:+.1f}%")

st.divider()

# Sector filter
available_sectors = [s for s in SECTOR_ORDER if s in df["Sector"].unique()]
selected_sectors = st.multiselect(
    "Filter by sector",
    options=available_sectors,
    default=available_sectors,
    label_visibility="collapsed",
    placeholder="Filter by sector...",
)

view = df[df["Sector"].isin(selected_sectors)].copy() if selected_sectors else df.copy()

# Default sort: highest upside first
view = view.sort_values("Upside %", ascending=False, na_position="last")

# Format display columns (keep numeric copies for sorting)
def render_display(row):
    return pd.Series({
        "Ticker": f"{row['Ticker']}  ({row['Region']})",
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

# Color the Upside column
def color_upside(val: str) -> str:
    if val == "—":
        return "color: #6b7280;"
    try:
        n = float(val.replace("%", "").replace("+", ""))
    except ValueError:
        return ""
    if n < 0:        return "color: #dc2626;"
    if n >= 40:      return "color: #047857; font-weight: 600;"
    if n >= 15:      return "color: #0369a1;"
    return "color: #6b7280;"

def color_day(val: str) -> str:
    if val == "—":
        return "color: #6b7280;"
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

# Export
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
        """
- **Not financial advice.** Analyst targets are baked in from public sources at the time
  `watchlist_data.py` was last edited; they don't auto-refresh. Live data is price/day%/range/mcap only.
- **Cache:** Prices cache for 15 minutes. Hit the **Refresh** button to force a re-fetch.
- **FMP free tier** has rate limits (~250 req/day). With 20 tickers and a 15-minute cache,
  one user can refresh ~12 times per day without trouble.
- **Editing the watchlist:** add/remove tickers in `watchlist_data.py`, commit + push,
  Streamlit Cloud will auto-redeploy.
- **Negative upside** means the price has run past the static target — re-research that
  name before acting; the original "undervalued" thesis no longer holds.
"""
    )
