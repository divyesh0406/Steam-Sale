# Steam Sale Lift

**Measuring the causal effect of Steam's Winter 2024 sale on game reviews — using DiD, RDD, and Synthetic Control.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://steam-sale-lift.streamlit.app)

---

## What this project does

Steam runs major seasonal sales (Summer, Winter, Spring) where games are discounted for 10–15 days. Do those discounts actually cause more reviews — a proxy for engagement — or is any uptick just seasonal noise?

This project builds a ground-up causal inference pipeline from public Steam data to answer that question with three independent methods. No LLMs, no agents: just applied data science.

**Target event:** Steam Winter 2024 Sale — Dec 19 2024 to Jan 2 2025  
**Dataset:** 2,185 games, 205,163 daily review observations (76-day window)  
**Treatment proxy:** SteamSpy `discount > 0` snapshot → 232 treated, 1,953 control

---

## Results

| Hypothesis | Method | Finding |
|---|---|---|
| H1: Does sale participation increase reviews? | DiD + CUPED | ATT = −0.032 log reviews, p = 0.293 — **not significant** |
| H2: Does crossing 50% discount boost reviews? | Regression Discontinuity | Jump = +4.2 log reviews, p = 0.322 — **not significant** |
| H3: Is the lift larger for indie vs AAA? | Synthetic Control | Indie p = 0.000 ✓, AAA p = 0.500 — **heterogeneity confirmed** |

**Key finding:** No average effect across the full game population. But indie games show a statistically significant review lift vs their synthetic counterfactuals; AAA games do not. The result is partially driven by MiSide — a viral indie title — which warrants caution.

---

## Architecture

```
Public Steam APIs (SteamSpy, Steam Store, SteamCharts)
        │
        ▼
Python scrapers  (src/scrape/)
        │
        ▼
Raw JSON  →  data/raw/
        │
        ▼
Neon Postgres  (raw → stg → mart)
src/transform/stg_and_marts.sql
        │
        ▼
Causal analysis  (notebooks/ + src/analysis/)
DiD + CUPED  |  RDD  |  Synthetic Control
        │
        ▼
Streamlit dashboard  (dashboard/app.py)
```

---

## Methods

### H1 — Difference-in-Differences + CUPED

2×2 DiD comparing treated games (currently discounted per SteamSpy) to untreated games across pre-sale (Nov 19 – Dec 18) and sale windows (Dec 19 – Jan 2). CUPED (Deng et al. 2013) uses pre-period reviews as a control variate, reducing variance by **94%**. Permutation-based inference (1,000 shuffles).

### H2 — Regression Discontinuity

Sharp RDD on discount depth. Running variable: SteamSpy discount %. Cutoff: 50% — Steam's alleged featured-placement visibility threshold. Local linear regression with HC1 robust SEs. Bandwidth sensitivity (5–30 pp) and McCrary density test for manipulation.

### H3 — Synthetic Control

Abadie et al. convex weights (scipy SLSQP). Top 5 indie + top 5 AAA treated games vs a 200-game donor pool of non-treated games. Weights chosen to minimise pre-period MSPE. Placebo-based inference: p-value = fraction of 50 fake treated units with post/pre RMSPE ratio ≥ treated.

---

## Quickstart

### Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) — `pip install uv`
- A free [Neon Postgres](https://neon.tech) database (512 MB free tier is sufficient)

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/steam-sale-lift
cd steam-sale-lift
uv sync
cp .env.example .env
# Edit .env: set DATABASE_URL to your Neon connection string
```

### Scrape & load

```bash
make scrape-universe   # fetch top 3,000 games by owner count
make scrape-metadata   # Steam Store metadata (~5 hr overnight)
make scrape-players    # SteamSpy + SteamCharts player history
make load              # upsert all raw JSON into Postgres
make transform         # apply stg + mart SQL views
```

> **Note:** Review data in this project is synthetic — generated from SteamSpy lifetime review totals distributed across time using exponential decay and sale multipliers. The Steam Reviews API doesn't support efficient historical date-range queries for high-volume games.

### Analyze

```bash
# Recompute synthetic control (runs ~3 min, saves results/h3_sc_results.json)
uv run python scripts/run_sc_analysis.py

# Re-execute all notebooks
make analyze
```

### Dashboard

```bash
uv run streamlit run dashboard/app.py
```

---

## Project structure

```
steam-sale-lift/
├── data/raw/               # scraped JSON (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb                         # EDA + treatment assignment
│   ├── 02_hypothesis_1_did.ipynb            # DiD + CUPED
│   ├── 03_hypothesis_2_rdd.ipynb            # RDD
│   └── 04_hypothesis_3_synthetic_control.ipynb  # SC (loads pre-computed)
├── results/                # parquets, figures, JSON results (gitignored)
├── src/
│   ├── scrape/             # steam_api, steamspy, steamcharts scrapers
│   ├── load/               # ingest pipeline → Postgres
│   ├── transform/          # stg_and_marts.sql
│   └── analysis/           # cuped.py, did.py, rdd.py, synthetic_control.py
├── scripts/
│   └── run_sc_analysis.py  # standalone SC script (avoids notebook timeout)
├── dashboard/
│   └── app.py              # Streamlit dashboard
├── pyproject.toml
└── Makefile
```

---

## Limitations

1. **Synthetic reviews.** Review data is generated from SteamSpy lifetime totals, not real Steam timestamps. Causal estimates reflect the simulated data-generating process, not observed user behaviour.
2. **Treatment proxy.** `discount > 0` is a current SteamSpy snapshot, not a verified Winter 2024 participation record.
3. **Small RDD sample.** Only 58 games within ±15 pp of the 50% cutoff — underpowered for H2.
4. **MiSide outlier.** The indie SC result is heavily influenced by one viral game with poor pre-period synthetic fit (pre-RMSPE = 0.68).
5. **Selection into sales.** Games that go on sale are not random — DiD and SC partially address this but cannot fully eliminate selection bias.

---

## Tech stack

| Layer | Tools |
|---|---|
| Scraping | `httpx`, `tenacity`, `beautifulsoup4` |
| Storage | Neon Postgres (free tier), `psycopg` v3 |
| Transforms | raw SQL |
| Analysis | `pandas`, `numpy`, `statsmodels`, `scipy` |
| Dashboard | `streamlit` |
| Reproducibility | `uv`, `Makefile` |

---

## License

MIT. Data scraped from public Steam APIs under their terms of service.
