.PHONY: all setup scrape load analyze test lint clean

# ── Top-level targets ─────────────────────────────────────────────────────────

all: setup scrape load analyze

setup:
	uv sync
	cp -n .env.example .env || true
	@echo ">>> Edit .env with your STEAM_API_KEY and DATABASE_URL before continuing."

# ── Scraping ──────────────────────────────────────────────────────────────────

scrape: scrape-universe scrape-metadata scrape-reviews scrape-players scrape-prices

scrape-universe:
	uv run python -m src.scrape.steam_api universe

scrape-metadata:
	uv run python -m src.scrape.steam_api metadata

scrape-reviews:
	uv run python -m src.scrape.steam_api reviews

scrape-players:
	uv run python -m src.scrape.steamspy
	uv run python -m src.scrape.steamcharts

scrape-prices:
	uv run python -m src.scrape.steamdb

# ── Load & transform ──────────────────────────────────────────────────────────

load:
	uv run python -m src.load.ingest

# ── Analysis ──────────────────────────────────────────────────────────────────

analyze:
	uv run jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb
	uv run jupyter nbconvert --to notebook --execute notebooks/02_hypothesis_1_did.ipynb
	uv run jupyter nbconvert --to notebook --execute notebooks/03_hypothesis_2_rdd.ipynb
	uv run jupyter nbconvert --to notebook --execute notebooks/04_hypothesis_3_synthetic_control.ipynb

# ── Dashboard ─────────────────────────────────────────────────────────────────

dashboard:
	uv run streamlit run dashboard/streamlit_app.py

# ── Dev / CI ──────────────────────────────────────────────────────────────────

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

test:
	uv run pytest

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
