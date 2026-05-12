# Steam Sale Lift

**Measuring the causal effect of Steam seasonal sales on revenue, reviews, and player engagement.**

Self-scraped dataset of ~3,000 games × ~5 years of daily price, review, and player-count history — analyzed with DiD + CUPED, RDD, and Synthetic Control. Free hosting, $0 cost.

---

## TL;DR

Steam sales are everywhere, but do they actually lift revenue — or just pull demand forward? This project builds a ground-up causal inference pipeline from public Steam data to answer that question rigorously. No LLMs, no agents: just applied data science.

---

## Architecture

```
[Steam APIs + SteamSpy + SteamCharts + SteamDB]
            │
            ▼
   [Python scrapers — rate-limited, cached, idempotent]
   src/scrape/{steam_api, steamspy, steamcharts, steamdb}.py
            │
            ▼
   [Raw JSON → data/raw/]
            │
            ▼
   [Postgres (Neon free tier)]
   raw.* → stg.* → mart.*
            │
            ▼
   [Causal analyses — notebooks/ + src/analysis/]
   DiD + CUPED | RDD | Synthetic Control
            │
            ▼
   [Streamlit dashboard on HF Spaces + 3 executive memos]
```

---

## Methods

| Hypothesis | Method | Status |
|---|---|---|
| H1: Sales → net 30-day revenue lift? | Two-way FE DiD + CUPED | Phase 4 |
| H2: Discounts >50% → review score drop? | Regression Discontinuity | Phase 4 |
| H3: Indie > AAA marginal effect? | Synthetic Control | Phase 4 |

---

## Quickstart

### 1. Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (`pip install uv`)
- A free [Steam Web API key](https://steamcommunity.com/dev/apikey)
- A free [Neon Postgres](https://neon.tech) database

### 2. Setup

```bash
git clone https://github.com/YOUR_USERNAME/steam-sale-lift
cd steam-sale-lift
uv sync
cp .env.example .env
# Edit .env with your STEAM_API_KEY and DATABASE_URL
```

### 3. Scrape (runs on your laptop — ~2 weeks of overnight runs for full dataset)

```bash
make scrape-universe   # Day 1–2: get top 3,000 games by owner count
make scrape-metadata   # Day 3–5: Steam Store details (~5hr overnight)
make scrape-reviews    # Day 6–7: paginate reviews per game
make scrape-players    # Week 2 Day 1–3: SteamSpy + SteamCharts
make scrape-prices     # Week 2 Day 4–5: SteamDB price history
```

### 4. Load to Postgres

```bash
make load   # applies schema, upserts all data idempotently
```

### 5. Analyze

```bash
make analyze   # executes all notebooks
```

### 6. Dashboard

```bash
make dashboard   # starts Streamlit locally
```

---

## Data Dictionary

### mart.dim_games

| Column | Type | Description |
|---|---|---|
| appid | BIGINT | Steam app ID (primary key) |
| name | TEXT | Game name |
| release_date | DATE | Release date (may be null for early access) |
| developer | TEXT | Primary developer |
| publisher | TEXT | Primary publisher |
| base_price_cents | INT | Launch price in US cents |
| is_indie | BOOLEAN | True if indie publisher + Indie genre tag |
| is_aaa | BOOLEAN | True if publisher is in top-50 publisher list |
| primary_genre | TEXT | First genre tag from Steam |
| genres | TEXT[] | All genre tags |
| owners_lower | INT | SteamSpy estimated owners lower bound |
| owners_upper | INT | SteamSpy estimated owners upper bound |

### mart.fct_prices_daily

| Column | Type | Description |
|---|---|---|
| appid | BIGINT | FK → dim_games |
| date | DATE | Price date |
| price_cents | INT | Price in US cents that day |
| discount_pct | NUMERIC | Discount % from base price (0 = full price) |
| is_on_sale | BOOLEAN | Generated: discount_pct > 0 |
| sale_event_id | TEXT | FK → dim_sale_events (null if not during a known sale) |

### mart.fct_reviews

| Column | Type | Description |
|---|---|---|
| review_id | BIGINT | Steam recommendation ID |
| appid | BIGINT | FK → dim_games |
| review_date | DATE | Date review was submitted |
| is_positive | BOOLEAN | Thumbs up/down |
| playtime_at_review_min | INT | Playtime at time of review (minutes) |
| helpful_votes | INT | Number of helpful votes |

### mart.fct_players_monthly

| Column | Type | Description |
|---|---|---|
| appid | BIGINT | FK → dim_games |
| year_month | TEXT | e.g. "April 2024" |
| avg_players | NUMERIC | Average concurrent players that month |
| peak_players | INT | Peak concurrent players that month |

---

## Limitations

1. **No direct sales data.** Steam doesn't expose units sold. Review velocity and player counts are proxies only.
2. **Selection into sales.** Games that go on sale are not random — they may already be trending down. DiD + synthetic control partially address this but cannot fully eliminate it.
3. **SteamDB price coverage.** Not all games have complete price history on SteamDB.
4. **SteamSpy accuracy.** Ownership estimates are rough ranges, not exact figures.
5. **Review review-bombing.** Polarized review events (unrelated to sales) can contaminate the review-score outcome.

---

## Tech Stack

- **Scraping**: `httpx`, `tenacity`, `beautifulsoup4`
- **Storage**: Neon Postgres (free tier)
- **Transforms**: raw SQL via Makefile
- **Analysis**: `pandas`, `statsmodels`, `linearmodels`, `scipy`
- **Causal**: `DoWhy`, `EconML`, `pysyncon`
- **Dashboard**: `Streamlit`, `plotly`
- **CI**: GitHub Actions
- **Reproducibility**: `uv`, `Makefile`

---

## License

MIT. Data scraped from public Steam APIs under their terms of service.
