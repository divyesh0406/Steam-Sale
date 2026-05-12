# Steam Sale Lift
## A Complete Build Guide: From Scraping to Causal Inference, Hosted for $0

**Project goal.** Measure the *causal* effect of Steam sales on revenue, review scores, and player engagement using a self-built dataset of ~10,000 games × ~5 years of daily price, review, and player-count history — analyzed with A/B-style hypothesis testing, CUPED, difference-in-differences, regression discontinuity, and synthetic control.

**What this project signals to a hiring manager.** End-to-end applied data science: you can build a dataset from nothing, design a star-schema warehouse, write production-quality SQL, generate falsifiable hypotheses, choose the right causal method for each, defend your assumptions, and communicate the result to a non-technical audience. No LLMs, no agents, no fine-tuning — just rigorous applied DS.

**Total time.** ~6–8 weeks at 10–15 hrs/week. Cost: $0.

---

## Table of Contents

1. Architecture Overview
2. Tech Stack & Why Each Choice
3. Phase 1 — Data Collection (Weeks 1–2)
4. Phase 2 — Warehouse & Transformations (Week 2)
5. Phase 3 — EDA & Hypothesis Generation (Week 3)
6. Phase 4 — The Causal Analyses (Weeks 4–6)
7. Phase 5 — Communication Layer (Week 7)
8. Free Hosting Plan
9. Resume Bullet Drafts
10. Common Pitfalls

---

## 1. Architecture Overview

```
[Steam APIs + SteamDB + SteamSpy]
            │
            ▼
   [Python scrapers w/ rate limiting + cache]
            │
            ▼
   [Raw JSON dumps → /data/raw/]
            │
            ▼
   [Postgres (Neon free tier)]
   ┌──────────────────────────┐
   │ raw.*    (landing)       │
   │ stg.*    (cleaned)       │
   │ mart.*   (analytics)     │
   └──────────────────────────┘
            │
            ▼
   [Analysis notebooks + Python modules]
   ┌──────────────────────────┐
   │ DiD, RDD, CUPED, PSM,    │
   │ Synthetic Control        │
   └──────────────────────────┘
            │
            ▼
   [Streamlit dashboard on HF Spaces]
   [Three executive memos in /reports]
   [Medium / TDS blog post]
```

Three layers: **collection → warehouse → analysis**. Each runs independently and is reproducible with a single `make` command.

---

## 2. Tech Stack & Why Each Choice

| Layer | Tool | Why |
|---|---|---|
| Scraping | `httpx` + `tenacity` | Async-capable, modern, retries with exponential backoff |
| Request cache | `hishel` or local SQLite | Don't re-hit the same URL during dev |
| Storage | **Neon Postgres free tier** | 0.5 GB, no credit card, scale-to-zero, branching |
| Transforms | `dbt-core` (free) or raw SQL via Makefile | Industry standard; demonstrates analytics engineering |
| Analysis | `pandas`, `statsmodels`, `linearmodels`, `scipy` | Standard applied stats stack |
| Causal libs | `DoWhy`, `EconML`, `pysyncon` | DoWhy for graph-based reasoning, EconML for double ML, pysyncon for synthetic control |
| Dashboard | Streamlit | Free Community Cloud or HF Spaces |
| Repo | GitHub | Free public repos |
| CI | GitHub Actions | Free for public repos; run tests + lint on every push |
| Reproducibility | `uv` (or Poetry) + `Makefile` | Single `make all` rebuilds everything |

**Deliberately not on this list:** LangChain, agents, vector DBs, fine-tuned models. The whole point is a pure applied DS portfolio piece.

---

## 3. Phase 1 — Data Collection (Weeks 1–2)

### 3.1 What you're collecting

| Source | What you get | Endpoint / method |
|---|---|---|
| Steam Web API (`api.steampowered.com`) | Game list, app metadata | `ISteamApps/GetAppList` |
| Steam Store API (`store.steampowered.com/api`) | Genres, tags, price, release date, developer | `appdetails?appids={id}` |
| Steam Reviews API | Review text, score, timestamp, playtime-at-review | `appreviews/{id}?json=1&filter=all` |
| SteamSpy (`steamspy.com/api.php`) | Owners estimate, daily player counts | `request=appdetails&appid={id}` |
| Steam Charts | Concurrent player history | HTML scrape — `steamcharts.com/app/{id}` |
| SteamDB | Daily price history, sale event tags | HTML scrape (use respectfully) |

**Critical caveats from current sources:**
- The Steam Web API is **aggressively rate-limited** in 2026. Practical guidance from current scraping tutorials: max ~200 requests per 5 minutes, 1–2 second delay between requests. Going faster gets you shadow-banned with crazy rate limits.
- Cache aggressively — game metadata rarely changes, cache it for 24h+.
- Set the `birthtime` cookie when scraping store pages to bypass age-check redirects on mature games.
- Get a Steam Web API key for free at `steamcommunity.com/dev/apikey` — requires Steam Guard enabled.

### 3.2 Repo layout

```
steam-sale-lift/
├── README.md
├── Makefile
├── pyproject.toml
├── .env.example
├── .github/workflows/ci.yml
├── data/
│   ├── raw/          # never committed; gitignored
│   └── samples/      # tiny committed samples for tests
├── src/
│   ├── scrape/
│   │   ├── steam_api.py
│   │   ├── steamspy.py
│   │   ├── steamcharts.py
│   │   └── steamdb.py
│   ├── load/
│   │   ├── schema.sql
│   │   └── ingest.py
│   ├── transform/    # dbt models or raw SQL
│   └── analysis/
│       ├── eda.py
│       ├── cuped.py
│       ├── did.py
│       ├── rdd.py
│       └── synthetic_control.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_hypothesis_1_did.ipynb
│   ├── 03_hypothesis_2_rdd.ipynb
│   └── 04_hypothesis_3_synthetic_control.ipynb
├── reports/
│   ├── memo_1_revenue_lift.md
│   ├── memo_2_review_scores.md
│   └── memo_3_genre_heterogeneity.md
├── dashboard/
│   └── streamlit_app.py
└── tests/
```

### 3.3 The polite scraper template

```python
# src/scrape/steam_api.py
import httpx
import time
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from pathlib import Path
import json
import hashlib

logger = logging.getLogger(__name__)
CACHE_DIR = Path("data/raw/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class SteamClient:
    def __init__(self, api_key: str, requests_per_5min: int = 180):
        self.api_key = api_key
        self.client = httpx.Client(
            timeout=30.0,
            headers={"User-Agent": "SteamSaleLift/1.0 (academic research)"},
        )
        self.delay = 300 / requests_per_5min  # ~1.67s between calls
        self.last_call = 0.0

    def _throttle(self):
        elapsed = time.time() - self.last_call
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_call = time.time()

    def _cache_key(self, url: str, params: dict) -> Path:
        h = hashlib.sha256(f"{url}{json.dumps(params, sort_keys=True)}".encode()).hexdigest()[:16]
        return CACHE_DIR / f"{h}.json"

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
    def get(self, url: str, params: dict | None = None, use_cache: bool = True) -> dict:
        params = params or {}
        cache_path = self._cache_key(url, params)
        if use_cache and cache_path.exists():
            return json.loads(cache_path.read_text())

        self._throttle()
        r = self.client.get(url, params=params)
        if r.status_code == 429:
            logger.warning("Hit 429 — backing off 5 min")
            time.sleep(300)
            raise RuntimeError("rate limited")
        r.raise_for_status()
        data = r.json()
        cache_path.write_text(json.dumps(data))
        return data
```

**Key design choices to highlight in your README:**
1. **Throttling per call**, not just per session — protects you across script restarts
2. **Exponential backoff via tenacity** — handles transient 5xx and network flakes
3. **Local file cache keyed by URL+params** — re-running the script is free
4. **Aggressive 429 handling** — stop, wait 5 minutes, retry once, then fail loudly
5. **Realistic User-Agent** — never lie about what you are

### 3.4 Scraping plan

**Week 1, Day 1–2: Universe.** Hit `GetAppList` once; you'll get ~150K appids. Filter to "games" only (excluding software, DLC, soundtracks). Target the **top 10,000 by review count or estimated owners** — anything below that has too few reviews/players for statistical work. Save to `raw.app_list`.

**Week 1, Day 3–5: Metadata.** For each appid, hit Steam Store API `appdetails`. Save raw JSON. ~10K calls × 1.7s = ~5 hours — run overnight. You'll get genres, tags, base price, release date, developer, publisher.

**Week 1, Day 6–7: Reviews.** For each appid, paginate through `appreviews` with `filter=all` and `num_per_page=100`. Cap at 1,000 reviews/game (more than enough for stats; reduces volume 10×). Save timestamp, playtime-at-review, score, helpful votes.

**Week 2, Day 1–3: Player counts.** SteamSpy gives current owners + 2-week activity; Steam Charts (HTML scrape) gives historical concurrent players going back to launch. Pair them.

**Week 2, Day 4–5: Price history.** SteamDB has daily price history per game. Note: scraping SteamDB at scale is delicate. Use lower concurrency (1 req every 3s), respect robots.txt, and consider their public sale event listings as a more polite alternative. Also: many sale events are public knowledge (Summer Sale, Winter Sale, Autumn Sale, Spring Sale, Lunar New Year Sale) with known dates — you can hard-code these and just need each game's *participation* status.

**Week 2, Day 6–7: Idempotent ingestion + checkpointing.** Convert raw JSON to Postgres rows. Re-running the scraper should never duplicate data; use `INSERT ... ON CONFLICT DO UPDATE` everywhere.

### 3.5 Storage budget check

Rough numbers:
- 10,000 games × ~10 KB metadata = 100 MB
- 10,000 games × 1,000 reviews × ~500 bytes = ~5 GB raw, but only ~500 MB after pruning text
- 10,000 games × 1,800 days × 50 bytes (price+player_count) = ~900 MB

The Neon free tier is **0.5 GB per branch**. Two strategies:

1. **Sample down.** Use 3,000 games instead of 10,000. Cuts everything proportionally and is plenty for statistical power.
2. **Aggregate before storing.** Don't store raw review text in Postgres — only score, timestamp, playtime. Pre-aggregate price-day to weekly averages. Store the raw JSON dumps as compressed Parquet in your repo's `data/raw/` (gitignored) or on free Cloudflare R2 (10 GB free).

The Supabase free tier also gives 500 MB and lets you have 2 active projects. Either works.

---

## 4. Phase 2 — Warehouse & Transformations (Week 2, parallel)

### 4.1 Schema (star, kept simple)

```sql
-- mart.dim_games
CREATE TABLE mart.dim_games (
    appid BIGINT PRIMARY KEY,
    name TEXT NOT NULL,
    release_date DATE,
    developer TEXT,
    publisher TEXT,
    base_price_cents INT,
    is_indie BOOLEAN,        -- derived: solo or <5-person developer
    is_aaa BOOLEAN,          -- derived: publisher in top-50 publisher list
    primary_genre TEXT,
    genres TEXT[],           -- multi-valued
    tags TEXT[]
);

-- mart.fct_prices_daily
CREATE TABLE mart.fct_prices_daily (
    appid BIGINT REFERENCES mart.dim_games(appid),
    date DATE,
    price_cents INT,
    discount_pct NUMERIC(5,2),
    is_on_sale BOOLEAN,
    sale_event_id TEXT,      -- 'summer_2024', 'winter_2024', NULL otherwise
    PRIMARY KEY (appid, date)
);

-- mart.fct_reviews
CREATE TABLE mart.fct_reviews (
    review_id BIGINT PRIMARY KEY,
    appid BIGINT REFERENCES mart.dim_games(appid),
    review_date DATE,
    is_positive BOOLEAN,
    playtime_at_review_min INT,
    helpful_votes INT
);

-- mart.fct_players_daily
CREATE TABLE mart.fct_players_daily (
    appid BIGINT REFERENCES mart.dim_games(appid),
    date DATE,
    avg_concurrent_players INT,
    peak_concurrent_players INT,
    PRIMARY KEY (appid, date)
);

-- mart.dim_sale_events
CREATE TABLE mart.dim_sale_events (
    sale_event_id TEXT PRIMARY KEY,
    name TEXT,
    start_date DATE,
    end_date DATE,
    sale_type TEXT  -- 'major' (summer/winter), 'minor' (lunar/spring), 'themed'
);
```

### 4.2 Indexes you'll actually use

```sql
CREATE INDEX idx_prices_appid_date ON mart.fct_prices_daily(appid, date);
CREATE INDEX idx_prices_sale_event ON mart.fct_prices_daily(sale_event_id) WHERE sale_event_id IS NOT NULL;
CREATE INDEX idx_reviews_appid_date ON mart.fct_reviews(appid, review_date);
CREATE INDEX idx_players_appid_date ON mart.fct_players_daily(appid, date);
```

### 4.3 Useful analytics views

```sql
CREATE VIEW mart.v_game_sale_windows AS
SELECT
    g.appid,
    g.name,
    g.primary_genre,
    s.sale_event_id,
    s.start_date,
    s.end_date,
    s.start_date - INTERVAL '30 days' AS pre_window_start,
    s.end_date + INTERVAL '30 days' AS post_window_end,
    EXISTS (
        SELECT 1 FROM mart.fct_prices_daily p
        WHERE p.appid = g.appid
          AND p.sale_event_id = s.sale_event_id
          AND p.is_on_sale
    ) AS participated
FROM mart.dim_games g CROSS JOIN mart.dim_sale_events s;
```

---

## 5. Phase 3 — EDA & Hypothesis Generation (Week 3)

A standard EDA notebook — but the goal is to generate **3–4 sharp, falsifiable hypotheses** that the rest of the project answers.

### 5.1 EDA checklist

- Distribution of discount depths by genre and time of year
- Player count seasonality — STL decomposition (trend + seasonal + residual)
- Review score distributions overall, by genre, and around sale events
- Sale participation rates by indie vs. AAA
- Time-series plots: price + player count + cumulative reviews for 5 hand-picked games (e.g., a hit indie, a hit AAA, a flop, an early-access, a long-tail catalog game)
- Histogram of "days since release" at first sale — when do games first go on sale?

### 5.2 The four hypotheses

| # | Hypothesis | Method |
|---|---|---|
| H1 | Sales cause net-positive revenue lift over the following 30 days (vs. just pulling forward demand) | DiD + CUPED |
| H2 | Deeper discounts (>50% off) cause statistically significant drops in review scores | RDD |
| H3 | The marginal effect of Summer Sale participation is larger for indie games than AAA | Synthetic control + heterogeneity |
| H4 | Repeat sales within 90 days have diminishing returns | PSM (optional) |

You'll execute H1, H2, H3 deeply. H4 is a stretch goal.

---

## 6. Phase 4 — The Causal Analyses (Weeks 4–6)

This is the meat of the project. Each gets its own notebook + a Python module in `src/analysis/`.

### 6.1 Hypothesis 1 — Difference-in-differences with CUPED

**Setup.**
- Treatment group: games that participated in Summer Sale 2024
- Control group: matched games (same primary genre, similar baseline price ±20%, similar 30-day pre-sale player count ±30%) that did NOT participate
- Pre-period: 30 days before sale start; post-period: 30 days after sale end
- Outcome: estimated daily revenue per game = `price_paid × estimated_units_sold`. Steam doesn't give you sales data, so use **player count delta as a proxy for engagement-driven revenue** OR **review velocity as a proxy for purchase velocity** (a fixed % of buyers leave reviews; this ratio is fairly stable per genre).

**Two-way fixed effects DiD:**

```python
# src/analysis/did.py
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS

def run_twfe_did(panel_df: pd.DataFrame, outcome: str) -> dict:
    """
    panel_df has columns: appid, date, treated, post, <outcome>, log_pre_players
    Returns ATT estimate with cluster-robust SE.
    """
    df = panel_df.set_index(['appid', 'date'])
    df['treat_post'] = df['treated'] * df['post']

    model = PanelOLS.from_formula(
        f'{outcome} ~ treat_post + EntityEffects + TimeEffects',
        data=df,
        check_rank=False
    )
    fit = model.fit(cov_type='clustered', cluster_entity=True)
    return {
        'att': fit.params['treat_post'],
        'se': fit.std_errors['treat_post'],
        'ci_low': fit.conf_int().loc['treat_post', 'lower'],
        'ci_high': fit.conf_int().loc['treat_post', 'upper'],
        'n_treated': df['treated'].sum(),
        'n_obs': len(df),
    }
```

**Add CUPED for variance reduction.** This is the Microsoft/Booking/Netflix shibboleth — implementing it from scratch demonstrates real understanding.

```python
# src/analysis/cuped.py
import numpy as np
import pandas as pd

def cuped_adjust(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Adjust outcome y using pre-experiment covariate x.
    Returns adjusted outcome and theta.
    Reference: Deng et al. 2013 (Microsoft).
    """
    theta = np.cov(y, x)[0, 1] / np.var(x)
    y_adj = y - theta * (x - x.mean())
    return y_adj, theta

def variance_reduction(y_raw: np.ndarray, y_cuped: np.ndarray) -> float:
    return 1 - (y_cuped.var() / y_raw.var())
```

**Critical diagnostic: the parallel trends plot.**

```python
# Plot pre-period averages for treated vs. control by week
# Eyeball test: do they move together before the sale?
# If yes, DiD is credible. If no, use synthetic control instead.
```

**Deliverables for H1:**
1. ATT estimate with 95% CI
2. Parallel-trends plot
3. CUPED variance reduction stat (typical: 20–40%)
4. Naïve "treated mean - control mean" estimate vs. DiD estimate — the contrast IS the story
5. Robustness: permute treatment assignment 1,000 times; show your real estimate is in the tail

### 6.2 Hypothesis 2 — Regression Discontinuity on discount depth

**Setup.** Steam's front page algorithmically features sales above a threshold (~50% off gets premium real estate; below 50% does not). This creates a **sharp visibility cutoff**: a game at 49% off and one at 51% off are nearly identical, but only one gets featured. This is your natural experiment.

**Observable outcome:** review velocity in the 14 days after sale start, as a function of discount %.

```python
# src/analysis/rdd.py
import numpy as np
import statsmodels.api as sm

def sharp_rdd(
    df: pd.DataFrame,
    running_var: str = 'discount_pct',
    cutoff: float = 50.0,
    outcome: str = 'reviews_14d_post',
    bandwidth: float = 15.0,
) -> dict:
    """
    Local linear regression on either side of cutoff.
    """
    band = df[(df[running_var] >= cutoff - bandwidth) &
              (df[running_var] <= cutoff + bandwidth)].copy()
    band['above'] = (band[running_var] >= cutoff).astype(int)
    band['centered'] = band[running_var] - cutoff
    band['centered_above'] = band['centered'] * band['above']

    X = sm.add_constant(band[['above', 'centered', 'centered_above']])
    y = band[outcome]
    model = sm.OLS(y, X).fit(cov_type='HC1')

    return {
        'effect_at_cutoff': model.params['above'],
        'se': model.bse['above'],
        'ci_low': model.conf_int().loc['above', 0],
        'ci_high': model.conf_int().loc['above', 1],
        'bandwidth': bandwidth,
        'n_obs': len(band),
    }

def bandwidth_sensitivity(df, bandwidths=[5, 10, 15, 20, 25, 30]):
    """Sweep bandwidths and check estimate stability."""
    return [sharp_rdd(df, bandwidth=bw) for bw in bandwidths]
```

**Required robustness checks:**
1. **Bandwidth sweep** — estimate at bandwidth = [5, 10, 15, 20, 25] %. The effect should be stable.
2. **Manipulation test (McCrary density test)** — is there a suspicious spike in density right above the cutoff? If yes, developers are gaming the threshold and RDD is invalid.
3. **Polynomial robustness** — re-run with quadratic instead of linear; results shouldn't change much.
4. **Placebo cutoffs** — try cutoff = 30%, 70%. Effects should be null at fake cutoffs.
5. **Bootstrap 95% CI** — 1,000 resamples for confidence intervals.

### 6.3 Hypothesis 3 — Synthetic Control for genre heterogeneity

**Setup.** Pick 5 high-profile indie games and 5 AAA games that participated in a major sale. For each, build a synthetic control = weighted combination of non-participating games that closely matches the treated game's pre-sale player-count trajectory. Compare actual post-sale to synthetic post-sale.

```python
# src/analysis/synthetic_control.py
from pysyncon import Dataprep, Synth
import pandas as pd

def fit_synthetic_control(
    df: pd.DataFrame,
    treated_appid: int,
    donor_pool_appids: list[int],
    pre_period: tuple[str, str],
    post_period: tuple[str, str],
    outcome: str = 'avg_concurrent_players',
):
    dataprep = Dataprep(
        foo=df,
        predictors=[outcome],
        predictors_op='mean',
        time_predictors_prior=pd.date_range(*pre_period),
        dependent=outcome,
        unit_variable='appid',
        time_variable='date',
        treatment_identifier=treated_appid,
        controls_identifier=donor_pool_appids,
        time_optimize_ssr=pd.date_range(*pre_period),
    )
    synth = Synth()
    synth.fit(dataprep=dataprep)

    return {
        'weights': synth.weights(),
        'gap': synth.gaps,  # actual - synthetic
        'rmse_pre': synth.pre_rmspe,
    }
```

**Required:**
- **Placebo tests:** apply the method to 50 non-treated games; the "effect" should be null on average. If the actual effect is in the top 5% of placebo distribution, p ≈ 0.05.
- **Gap plots** for each treated game.
- **Heterogeneity table:** average effect for indie vs. AAA, with bootstrap CIs.

### 6.4 Hypothesis 4 — Propensity Score Matching (optional)

If you have time. Match games with repeat sales (within 90 days) to games with single sales, using propensity scores estimated from genre, base price, age, and pre-period engagement. Compare downstream outcomes. Add Rosenbaum bounds for sensitivity analysis.

---

## 7. Phase 5 — Communication Layer (Week 7)

This is where 95% of portfolio projects die. Don't.

### 7.1 Three executive memos

One per hypothesis. Each is **one page, in `/reports/`**, written for a non-technical PM. Structure:

```
TITLE: Does Putting a Game on Sale Actually Increase 30-Day Revenue?

TL;DR (3 lines)
- Yes, but only for games priced above $20. Indie games priced $5–$15 see no
  net lift; sales just pull forward demand we'd have captured anyway.
- 95% CI on the AAA effect: +14% to +22% over 30 days.
- Recommendation: skip discount participation for sub-$15 catalog titles.

CONTEXT
[2 sentences on the question]

METHOD
[3 sentences: what we did, with what data, why this approach]

KEY FINDING
[A chart + 2 sentences]

NAÏVE vs. CAUSAL
[The naïve estimate said X. The causal estimate says Y. Here's why.]

LIMITATIONS
[What we cannot conclude. This section signals seniority.]

NEXT STEPS
[2 things a real product team would test next.]
```

### 7.2 Streamlit dashboard

```python
# dashboard/streamlit_app.py — outline only
import streamlit as st
import pandas as pd
from src.analysis.did import run_twfe_did

st.title("Steam Sale Lift — Causal Effect Explorer")

tab1, tab2, tab3 = st.tabs(["Game Explorer", "Sale Event Effects", "Methodology"])

with tab1:
    appid = st.selectbox("Pick a game", get_top_games())
    fig = plot_price_reviews_players_overlay(appid)
    st.plotly_chart(fig)
    st.subheader("Estimated counterfactual")
    st.write("What would have happened without the sale, per synthetic control:")
    st.plotly_chart(plot_synthetic_control(appid))

with tab2:
    sale = st.selectbox("Sale event", ['Summer 2024', 'Winter 2024', ...])
    st.metric("Average revenue lift (DiD)", "+18.2%", "±3.4pp")
    st.metric("Average review-score lift (RDD)", "-0.8%", "±0.3pp")
    st.dataframe(genre_heterogeneity_table(sale))

with tab3:
    st.markdown(open('reports/methodology.md').read())
```

### 7.3 The blog post

Write **one** blog post on Medium / Towards Data Science / your own GitHub Pages site. Lead with **Hypothesis 2 (review scores around the discount cutoff)** — it's the most surprising and visual result. ~1,500 words, 4–5 charts, link to the dashboard and GitHub repo.

This is the single highest-ROI piece of communication for visibility. A good TDS post can drive thousands of views to your portfolio.

### 7.4 The README

Your repo README is more important than your code. Mandatory sections:

1. **One-paragraph TL;DR** — what, why, what you found
2. **Architecture diagram** (an image)
3. **Headline results** (2–3 charts inline)
4. **Methods table** — hypothesis, method, finding, CI
5. **How to reproduce** — `make setup && make scrape && make analyze`
6. **Data dictionary** — every column in every mart table
7. **Limitations** — top 5 things this analysis can't tell you
8. **Tech stack badges**

---

## 8. Free Hosting Plan

Total cost target: **$0/month**. Here's the stack.

| Component | Service | Free tier | Why this one |
|---|---|---|---|
| Code | GitHub | Unlimited public repos | Standard |
| CI | GitHub Actions | 2,000 min/mo for public repos | Run lint + tests on push |
| Database | **Neon Postgres** | 0.5 GB storage, scale-to-zero, no credit card | Best DX, branching, no expiration |
| Backup DB | Supabase | 500 MB, 2 active projects | Backup option / staging |
| Object storage | Cloudflare R2 | 10 GB free, no egress fees | For raw JSON dumps if you want them online |
| Dashboard | **Hugging Face Spaces** | 2 vCPU, 16 GB RAM | More generous than Streamlit Cloud; supports Streamlit + Gradio |
| Alt dashboard | Streamlit Community Cloud | ~1 GB RAM, always-on for public apps | Easier setup, but less RAM |
| Blog | Medium / Dev.to / GitHub Pages | Free | TDS submission is the highest-prestige free option |
| Domain (optional) | None / Cloudflare | Free subdomain on hosting platform | Skip a custom domain at first |

### 8.1 Why HF Spaces over Streamlit Cloud

Hugging Face Spaces' free CPU-basic tier gives 2 CPU cores and 16 GB RAM, much more generous than Streamlit Cloud's ~1 GB. The trade-off: HF Spaces sleeps after ~48 hours of inactivity but wakes on visit; Streamlit Cloud keeps public apps always-on but with much less RAM. For a portfolio dashboard that needs to load a few hundred MB of pre-computed results into memory, **HF Spaces wins**.

### 8.2 Free-tier survival tactics

**Database — staying under 0.5 GB:**
- Don't store raw review text in Postgres. Store score, timestamp, playtime only.
- Pre-aggregate `fct_prices_daily` to weekly averages for the long tail of catalog games (only keep daily for the top 1,000 by review count).
- Drop reviews older than 5 years.
- Use `TIMESTAMP` not `TIMESTAMPTZ`, smaller indexes.
- `VACUUM FULL` after big imports to reclaim space.

**Compute — 5,000 game scrape on a free tier without timeouts:**
- Scrape **locally on your laptop** to a SQLite file (`data/raw/local.db`). Ingestion to Neon only after you have a clean Parquet snapshot.
- Don't run scrapers on free hosting — they'll get killed for timeouts. Hosting is for the dashboard only.

**Dashboard — keeping it snappy on weak hardware:**
- Pre-compute everything. The Streamlit app should never run a DiD or fit a synthetic control on demand. Save fitted results to Parquet/JSON; the app just loads them.
- Cache aggressively with `@st.cache_data`.
- Lazy-load charts (only render the active tab).

### 8.3 Step-by-step deployment

```bash
# 1. Set up Neon
# - sign up at neon.tech (no credit card)
# - create project "steam-sale-lift"
# - grab the connection string → put in .env

# 2. Local dev
git clone <your-repo>
cd steam-sale-lift
uv sync
make scrape    # runs overnight on your laptop
make load      # ingests to Neon
make analyze   # runs all notebooks, saves results to /results/

# 3. Deploy dashboard to HF Spaces
# - create Space at huggingface.co/spaces (Streamlit SDK)
# - link to your GitHub repo (or push directly via git)
# - add Neon DATABASE_URL as a Space Secret
# - HF auto-builds and deploys on push

# 4. Set up CI
# .github/workflows/ci.yml — runs ruff + pytest on every PR
```

---

## 9. Resume Bullet Drafts

Use one or two of these depending on space.

**Long version (full bullet, ~3 lines):**

> *Built end-to-end causal analytics platform on 18M+ rows of self-scraped Steam pricing, review, and player-count data; estimated the causal effect of seasonal sales on 30-day engagement using two-way fixed-effects difference-in-differences with CUPED variance reduction (28% SE reduction), regression discontinuity at the discount-visibility threshold, and synthetic control with placebo-based inference; documented findings in three executive memos and an interactive Streamlit dashboard.*

**Short version (one line, for tight resumes):**

> *Designed and shipped a causal analytics pipeline (Python, Postgres, Streamlit) on a self-scraped 10K-game Steam dataset; estimated heterogeneous sale effects via DiD + CUPED, RDD, and synthetic control with bootstrap inference.*

**Quantified-impact version (best for product DS):**

> *Engineered a 10K-game Steam dataset (price, reviews, players) from public APIs with idempotent ingestion to Postgres; identified that naïve estimates overstate seasonal-sale revenue lift by ~3× vs. DiD-with-CUPED estimates, and quantified an indie/AAA heterogeneous treatment effect via synthetic control with placebo-based p-values.*

---

## 10. Common Pitfalls

**Selection effects you must address head-on.** Games go on sale *because* they're underperforming or because publishers expect a sale to help. This is the single biggest threat to any naïve analysis. The whole point of the causal toolkit is to handle this — make sure your write-up *names this threat explicitly* and explains how each method addresses it. Hiring managers reading carefully will check for this.

**Don't let scraping become the whole project.** Cap data collection at 2 weeks. If you're still scraping in week 4, you've over-engineered. Stop and analyze what you have.

**Don't fit fancy ML models.** The temptation to throw XGBoost / causal forests at everything is real. Resist. Linear DiD with proper SEs is more credible to a senior DS than a black-box causal forest with vague "feature importances."

**Don't skip the diagnostic plots.** Parallel trends, McCrary density, placebo tests, gap plots — these are the credibility receipts. A DiD without a parallel-trends plot is not a DiD; it's wishful thinking.

**Don't silently aggregate away variation.** If you smooth daily prices to weekly averages, document it. If you cap reviews per game at 1,000, document it. The pre-registration mindset (decide your method before you see the result) is what separates DS from data-dredging.

**Don't claim what you can't claim.** "Sales cause revenue lift" and "sales are correlated with revenue lift" are different sentences. Use the right one. The "Limitations" section of every memo is what makes you sound senior.

**Don't ignore multiple testing.** If you run 50 hypothesis tests across genres and pick the significant ones, you've p-hacked. Use FDR correction (Benjamini-Hochberg) or pre-register your top 3 hypotheses.

---

## Appendix: Two-Week Sprint Plan If You're Short On Time

If you only have 2 weeks instead of 8, do this:

- **Days 1–2:** Scrape ~1,000 games (Steam Store API + reviews only — skip player counts and SteamDB)
- **Days 3–4:** Load to Neon, build minimal mart layer
- **Days 5–7:** Pick **only Hypothesis 2 (RDD on discount depth → review scores)**. Do it deeply with all robustness checks.
- **Days 8–10:** Streamlit dashboard with one page: the RDD result, with bandwidth slider
- **Days 11–12:** Write the blog post on TDS
- **Days 13–14:** Polish README, deploy to HF Spaces, ship

A single rigorous RDD analysis on a self-built dataset, deployed and written up well, beats three half-baked projects every time.

---

*End of guide. Ship it.*
