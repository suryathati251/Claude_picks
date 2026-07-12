# Stock Watchlist (Streamlit)

Daily-refreshing stock watchlist. Live prices from Financial Modeling Prep (FMP); analyst targets and thesis are stored in `watchlist_data.py` and edited by hand.

Open the deployed URL in your browser each morning — fresh prices on every page load, no cron required.

---

## What you get

- 20 tickers across AI/semis, biotech, energy, financials, consumer/comm, and international.
- Live: price, day %, 52-week range, position-in-range %, market cap.
- Static: analyst price target, thesis, catalyst, risk note.
- Computed live: % upside to target.
- Sector filter, sort by clicking column headers, CSV export, manual refresh button.
- **🚀 10x Radar** — scans the watchlist **plus the S&P 500 / Nasdaq-100** (~600 names, `tenx_universe.py`)
  for the exploding-revenue profile of past 10-baggers (Nvidia '23, Micron/SanDisk memory-cycle turns):
  quarterly revenue YoY **level + acceleration**, margin inflection (operating leverage), market-cap
  headroom, and momentum confirmation (`tenx_radar.py`). Runs on free Yahoo quarterly data — zero FMP
  quota — cached 21 days, filling ~25 tickers per page load (`TENX_SCAN_BUDGET` env to change) with a
  "Scan next batch" button to speed it up. Screens reported data only; not investment advice.
- **🎯 S&P 500 entry meter** (`entry_meter.py`) — a 0–100 fear/greed gauge for "invest during fear,
  not greed": drawdown from the 52-week high, VIX level + 1-year percentile, stretch vs the 200-day
  average, 52-week range position, RSI(14), and watchlist breadth, each normalized and averaged, with
  a zone-based deployment stance (accumulate into extreme fear · plain DCA in neutral · patience with
  extra cash in extreme greed). Timing caveats included; not investment advice.
- **Tab navigation** — 📊 Watchlist · 🚀 10x Radar · 🎯 Market & Entry · 🔎 Lookup. The radar's
  market-scan only loads when its tab is open, so the default view stays fast.
- **Radar context columns** — **P/S** (market cap ÷ TTM revenue: how much of the explosion is already
  priced), **Δ** (10x-score change vs ~a week ago), **🆕** (entered the top 10 since then), and
  **Earnings** (next report date, ⚠️ within 7 days; Yahoo, cached 3d, displayed rows only).
- **Score history** (`history.json`) — daily snapshots of 10x scores, watchlist composites, and the
  entry meter (180-day retention) power the Δ/🆕 flags and the meter trend line. Note: Streamlit
  Cloud's disk resets on redeploys, so history re-accumulates after each deploy.
- **Insider column** — open-market buys/sells by officers & directors over 90 days (🟢 net buying ·
  🟣 3+ buys, zero sells, near the 52-week low). Own FMP budget (`INSIDER_BUDGET`, default 8/load,
  cached 14d); if your FMP plan doesn't include insider data the column stays blank without wasting
  quota. Display-only — never moves the Score.
- **Daily digest** (`digest.py` + `.github/workflows/daily-digest.yml`) — weekday GitHub Action
  (13:30 UTC) that posts the entry-meter reading + 10x Radar top 10 to Telegram. Setup: repo
  **Settings → Secrets and variables → Actions** → add `FMP_API_KEY`, `TELEGRAM_BOT_TOKEN` (from
  @BotFather) and `TELEGRAM_CHAT_ID` (from @userinfobot). Without the Telegram secrets the digest
  still prints in each run's Actions log. Test it via Actions → Daily digest → Run workflow.

---

## Deploy to Streamlit Community Cloud (free)

### 1. Get an FMP API key
1. Sign up at <https://site.financialmodelingprep.com/developer> (free tier is fine).
2. Copy your API key from the dashboard.

### 2. Push this folder to a new GitHub repo
```bash
cd streamlit-watchlist
git init
git add .
git commit -m "Initial watchlist"
gh repo create stock-watchlist --public --source=. --remote=origin --push
```
(Or create the repo on github.com and `git push` manually. **Must be public** for the free Streamlit tier — keep your API key in Streamlit secrets, never in code.)

### 3. Deploy on Streamlit Cloud
1. Go to <https://streamlit.io/cloud> and sign in with GitHub.
2. Click **New app**.
3. Pick your repo, branch `main`, main file `app.py`.
4. Click **Advanced settings → Secrets** and paste:
   ```toml
   FMP_API_KEY = "your-actual-key-here"
   ```
5. Click **Deploy**.

You'll get a URL like `https://stock-watchlist-yourname.streamlit.app`. Bookmark it.

### 4. (Optional) Local development
```bash
cd streamlit-watchlist
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml and paste your FMP key
streamlit run app.py
```
Opens at <http://localhost:8501>.

---

## Editing the watchlist

Edit `watchlist_data.py` — each entry is a dict with `ticker`, `region`, `name`, `sector`, `target`, `thesis`, `catalyst`, `risk`.

```python
{
    "ticker": "NVDA",
    "region": "US",
    "name": "Nvidia",
    "sector": "AI/Semis",
    "target": 180,
    "thesis": "...",
    "catalyst": "...",
    "risk": "...",
},
```

Set `"target": None` if you don't have a firm consensus number — the row will still show live price, but the Upside column will be blank.

Commit and push; Streamlit Cloud auto-redeploys within ~30 seconds.

### Non-US tickers
Use FMP's exchange suffix convention:
- India (NSE): `RELIANCE.NS`, `POLYCAB.NS`
- India (BSE): `RELIANCE.BO`
- Japan (Tokyo): `7203.T` (Toyota)
- UK (LSE): `BARC.L`
- Europe ADRs trade on NYSE/NASDAQ without a suffix.

---

## Want a true daily cron (push notifications)?

This setup uses **on-demand fetching** — fresh prices when you open the page, no scheduled job. If you want morning emails or Slack pings without opening the URL, add a GitHub Actions workflow at `.github/workflows/daily-digest.yml` that runs a Python script at, say, 6 a.m. ET, calls FMP for the same tickers, and posts to SMTP/Slack. Ask for that variant if you want it.

---

## Rate limits

FMP free tier: ~250 requests/day. The app caches prices for 15 minutes, so each refresh costs 20 requests (one per ticker). You can refresh ~12 times/day before hitting limits. For more, either upgrade FMP, increase the cache TTL in `app.py`, or trim the watchlist.

---

## Disclaimer

Not financial advice. Analyst targets are baked in from public sources at the time `watchlist_data.py` was last edited and do not auto-refresh. Negative upside means the price has run past the static target — re-research before acting.
