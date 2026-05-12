"""
Steam Web API + Steam Store API + Steam Reviews API scrapers.

Run modes (via __main__):
  python -m src.scrape.steam_api universe   — fetch all appids, filter to top games
  python -m src.scrape.steam_api metadata   — fetch store metadata per game
  python -m src.scrape.steam_api reviews    — paginate reviews per game
"""

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")

CACHE_DIR = Path("data/raw/cache")
RAW_DIR = Path("data/raw")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Known major Steam sale events with hard-coded date ranges (UTC).
# Participation per game is determined from price-history data, not scraped event pages.
KNOWN_SALE_EVENTS: list[dict] = [
    {"id": "summer_2019", "name": "Summer Sale 2019", "start": "2019-06-25", "end": "2019-07-09", "type": "major"},
    {"id": "winter_2019", "name": "Winter Sale 2019", "start": "2019-12-19", "end": "2020-01-02", "type": "major"},
    {"id": "spring_2020", "name": "Spring Sale 2020", "start": "2020-03-26", "end": "2020-03-30", "type": "minor"},
    {"id": "summer_2020", "name": "Summer Sale 2020", "start": "2020-06-25", "end": "2020-07-09", "type": "major"},
    {"id": "autumn_2020", "name": "Autumn Sale 2020", "start": "2020-11-25", "end": "2020-12-01", "type": "minor"},
    {"id": "winter_2020", "name": "Winter Sale 2020", "start": "2020-12-22", "end": "2021-01-05", "type": "major"},
    {"id": "lunar_2021", "name": "Lunar New Year 2021", "start": "2021-02-11", "end": "2021-02-15", "type": "minor"},
    {"id": "spring_2021", "name": "Spring Sale 2021", "start": "2021-03-11", "end": "2021-03-22", "type": "minor"},
    {"id": "summer_2021", "name": "Summer Sale 2021", "start": "2021-06-24", "end": "2021-07-08", "type": "major"},
    {"id": "autumn_2021", "name": "Autumn Sale 2021", "start": "2021-11-24", "end": "2021-11-30", "type": "minor"},
    {"id": "winter_2021", "name": "Winter Sale 2021", "start": "2021-12-22", "end": "2022-01-05", "type": "major"},
    {"id": "lunar_2022", "name": "Lunar New Year 2022", "start": "2022-02-03", "end": "2022-02-07", "type": "minor"},
    {"id": "spring_2022", "name": "Spring Sale 2022", "start": "2022-03-10", "end": "2022-03-21", "type": "minor"},
    {"id": "summer_2022", "name": "Summer Sale 2022", "start": "2022-06-23", "end": "2022-07-07", "type": "major"},
    {"id": "autumn_2022", "name": "Autumn Sale 2022", "start": "2022-11-22", "end": "2022-11-29", "type": "minor"},
    {"id": "winter_2022", "name": "Winter Sale 2022", "start": "2022-12-22", "end": "2023-01-05", "type": "major"},
    {"id": "lunar_2023", "name": "Lunar New Year 2023", "start": "2023-01-19", "end": "2023-01-23", "type": "minor"},
    {"id": "spring_2023", "name": "Spring Sale 2023", "start": "2023-03-16", "end": "2023-03-23", "type": "minor"},
    {"id": "summer_2023", "name": "Summer Sale 2023", "start": "2023-06-29", "end": "2023-07-13", "type": "major"},
    {"id": "autumn_2023", "name": "Autumn Sale 2023", "start": "2023-11-21", "end": "2023-11-28", "type": "minor"},
    {"id": "winter_2023", "name": "Winter Sale 2023", "start": "2023-12-21", "end": "2024-01-04", "type": "major"},
    {"id": "lunar_2024", "name": "Lunar New Year 2024", "start": "2024-02-08", "end": "2024-02-12", "type": "minor"},
    {"id": "spring_2024", "name": "Spring Sale 2024", "start": "2024-03-14", "end": "2024-03-21", "type": "minor"},
    {"id": "summer_2024", "name": "Summer Sale 2024", "start": "2024-06-27", "end": "2024-07-11", "type": "major"},
    {"id": "autumn_2024", "name": "Autumn Sale 2024", "start": "2024-11-27", "end": "2024-12-03", "type": "minor"},
    {"id": "winter_2024", "name": "Winter Sale 2024", "start": "2024-12-19", "end": "2025-01-02", "type": "major"},
    {"id": "lunar_2025", "name": "Lunar New Year 2025", "start": "2025-01-23", "end": "2025-01-27", "type": "minor"},
    {"id": "spring_2025", "name": "Spring Sale 2025", "start": "2025-03-13", "end": "2025-03-20", "type": "minor"},
    {"id": "summer_2025", "name": "Summer Sale 2025", "start": "2025-06-26", "end": "2025-07-10", "type": "major"},
]


class SteamClient:
    """
    Thread-safe, cache-backed HTTP client for Steam APIs.

    Throttles to ~180 req/5min by default (safely under the 200 req/5min limit).
    Caches responses on disk keyed by URL + params, so re-running is free.
    """

    BASE_API = "https://api.steampowered.com"
    BASE_STORE = "https://store.steampowered.com/api"

    def __init__(self, api_key: str, requests_per_5min: int = 180):
        if not api_key:
            raise ValueError("STEAM_API_KEY is required. Get one at steamcommunity.com/dev/apikey")
        self.api_key = api_key
        self.client = httpx.Client(
            timeout=30.0,
            headers={"User-Agent": "SteamSaleLift/1.0 (academic research, mittals@usc.edu)"},
            # Set birthtime cookie to bypass age-gate on mature content pages
            cookies={"birthtime": "631152000", "lastagecheckage": "1-0-1990"},
        )
        self.delay = 300.0 / requests_per_5min  # seconds between calls
        self.last_call: float = 0.0

    def _throttle(self) -> None:
        elapsed = time.time() - self.last_call
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_call = time.time()

    def _cache_key(self, url: str, params: dict) -> Path:
        h = hashlib.sha256(
            f"{url}{json.dumps(params, sort_keys=True)}".encode()
        ).hexdigest()[:16]
        return CACHE_DIR / f"{h}.json"

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
    def get(self, url: str, params: dict | None = None, use_cache: bool = True) -> dict:
        params = params or {}
        cache_path = self._cache_key(url, params)
        if use_cache and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        self._throttle()
        r = self.client.get(url, params=params)

        if r.status_code == 429:
            logger.warning("Hit 429 rate limit — backing off 5 minutes")
            time.sleep(300)
            raise RuntimeError("rate limited — retrying")

        if r.status_code == 403:
            logger.error("403 Forbidden — check your API key or IP ban")
            raise RuntimeError("403 Forbidden")

        r.raise_for_status()
        data = r.json()
        cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data

    # ── Universe ──────────────────────────────────────────────────────────────

    def get_app_list(self) -> list[dict]:
        """Fetch all Steam appids. Returns list of {appid, name}."""
        data = self.get(
            f"{self.BASE_API}/ISteamApps/GetAppList/v2/",
            params={"key": self.api_key},
        )
        return data["applist"]["apps"]

    # ── Metadata ──────────────────────────────────────────────────────────────

    def get_app_details(self, appid: int) -> dict | None:
        """Fetch store metadata for a single appid. Returns None if not a game."""
        data = self.get(
            f"{self.BASE_STORE}/appdetails",
            params={"appids": appid, "cc": "us", "l": "en"},
        )
        app_data = data.get(str(appid), {})
        if not app_data.get("success"):
            return None
        details = app_data["data"]
        if details.get("type") != "game":
            return None
        return details

    # ── Reviews ───────────────────────────────────────────────────────────────

    def get_reviews(self, appid: int, max_reviews: int = 1000) -> list[dict]:
        """
        Paginate through all reviews for a game using the cursor API.
        Stops at max_reviews (default 1,000). Returns list of review dicts.
        """
        reviews: list[dict] = []
        cursor = "*"
        base_url = f"https://store.steampowered.com/appreviews/{appid}"

        while len(reviews) < max_reviews:
            params = {
                "json": "1",
                "filter": "all",
                "language": "all",
                "num_per_page": "100",
                "cursor": cursor,
                "purchase_type": "all",
            }
            # Reviews endpoint doesn't need API key but is still rate-limited
            data = self.get(base_url, params=params, use_cache=(cursor == "*"))

            if data.get("success") != 1:
                break

            batch = data.get("reviews", [])
            if not batch:
                break

            reviews.extend(batch)
            next_cursor = data.get("cursor", "")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor

        return reviews[:max_reviews]


# ── Scraping entrypoints ───────────────────────────────────────────────────────

def _load_api_key() -> str:
    key = os.getenv("STEAM_API_KEY", "")
    if not key or key == "your_steam_api_key_here":
        logger.error(
            "STEAM_API_KEY not set. Edit your .env file.\n"
            "Get a key at: https://steamcommunity.com/dev/apikey"
        )
        sys.exit(1)
    return key


def scrape_universe(client: SteamClient, max_games: int = 3000) -> None:
    """
    Step 1: Fetch all appids, rank by SteamSpy estimated owners,
    and save the top `max_games` to data/raw/universe.json.

    We use SteamSpy's /all endpoint (returns ~35K games sorted by owners desc)
    to rank, then cross-reference with Steam's full app list to get accurate names.
    """
    universe_path = RAW_DIR / "universe.json"
    if universe_path.exists():
        logger.info("Universe already scraped — delete data/raw/universe.json to re-run")
        return

    logger.info("Fetching full Steam app list...")
    all_apps = client.get_app_list()
    app_map = {a["appid"]: a["name"] for a in all_apps if a.get("appid")}
    logger.info(f"Total apps on Steam: {len(app_map):,}")

    # SteamSpy /all gives page-by-page results sorted by owners desc.
    # This is the fastest way to get a ranked list without hitting per-app endpoints.
    logger.info("Fetching SteamSpy ranked game list (multiple pages)...")
    ranked: list[dict] = []
    for page in range(0, 10):  # up to 10 pages × ~1000 games = 10K candidates
        spy_data = client.get(
            "https://steamspy.com/api.php",
            params={"request": "all", "page": str(page)},
        )
        if not spy_data:
            break
        for appid_str, info in spy_data.items():
            try:
                appid = int(appid_str)
            except ValueError:
                continue
            ranked.append({
                "appid": appid,
                "name": info.get("name", app_map.get(appid, "")),
                "owners": info.get("owners", "0 .. 0"),  # range string e.g. "200000 .. 500000"
                "positive": info.get("positive", 0),
                "negative": info.get("negative", 0),
            })
        logger.info(f"  Page {page}: {len(spy_data)} entries, {len(ranked)} total so far")
        if len(ranked) >= max_games * 2:
            break
        time.sleep(1.5)  # SteamSpy rate limit is gentler but still needs respect

    # SteamSpy /all is already sorted by estimated owners desc.
    # Filter to only appids that appear in Steam's official game list.
    game_appids = set(app_map.keys())
    ranked_games = [r for r in ranked if r["appid"] in game_appids]
    top_games = ranked_games[:max_games]

    universe_path.write_text(
        json.dumps(top_games, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"Saved {len(top_games)} games to {universe_path}")


def scrape_metadata(client: SteamClient) -> None:
    """
    Step 2: For each game in the universe, fetch Steam Store appdetails.
    Saves one JSON file per game to data/raw/metadata/{appid}.json.
    Idempotent: skips games already fetched.
    """
    universe_path = RAW_DIR / "universe.json"
    if not universe_path.exists():
        logger.error("Run 'scrape universe' first.")
        sys.exit(1)

    games = json.loads(universe_path.read_text(encoding="utf-8"))
    meta_dir = RAW_DIR / "metadata"
    meta_dir.mkdir(exist_ok=True)

    todo = [g for g in games if not (meta_dir / f"{g['appid']}.json").exists()]
    logger.info(f"Fetching metadata for {len(todo)} games (skipping {len(games) - len(todo)} cached)")

    errors = 0
    for game in tqdm(todo, desc="metadata"):
        appid = game["appid"]
        details = client.get_app_details(appid)
        out_path = meta_dir / f"{appid}.json"
        if details is None:
            # Not a game or unavailable — save a tombstone so we don't retry
            out_path.write_text(json.dumps({"appid": appid, "skipped": True}), encoding="utf-8")
            errors += 1
        else:
            out_path.write_text(json.dumps(details, ensure_ascii=False), encoding="utf-8")

    logger.info(f"Metadata done. {errors} non-game/unavailable appids.")


def scrape_reviews(client: SteamClient, max_reviews_per_game: int = 1000) -> None:
    """
    Step 3: For each game, paginate reviews (capped at max_reviews_per_game).
    Saves to data/raw/reviews/{appid}.json. Idempotent.
    """
    universe_path = RAW_DIR / "universe.json"
    if not universe_path.exists():
        logger.error("Run 'scrape universe' first.")
        sys.exit(1)

    games = json.loads(universe_path.read_text(encoding="utf-8"))
    review_dir = RAW_DIR / "reviews"
    review_dir.mkdir(exist_ok=True)

    todo = [g for g in games if not (review_dir / f"{g['appid']}.json").exists()]
    logger.info(f"Fetching reviews for {len(todo)} games")

    for game in tqdm(todo, desc="reviews"):
        appid = game["appid"]
        reviews = client.get_reviews(appid, max_reviews=max_reviews_per_game)
        out = {
            "appid": appid,
            "total_fetched": len(reviews),
            "reviews": [
                {
                    "review_id": r["recommendationid"],
                    "is_positive": r["voted_up"],
                    "timestamp": r["timestamp_created"],
                    "playtime_at_review_min": r.get("author", {}).get("playtime_at_review", 0),
                    "helpful_votes": r.get("votes_up", 0),
                    "weighted_vote_score": r.get("weighted_vote_score", 0),
                }
                for r in reviews
            ],
        }
        (review_dir / f"{appid}.json").write_text(
            json.dumps(out, ensure_ascii=False), encoding="utf-8"
        )

    logger.info("Reviews done.")


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.scrape.steam_api [universe|metadata|reviews]")
        sys.exit(1)

    mode = sys.argv[1].lower()
    api_key = _load_api_key()
    max_games = int(os.getenv("MAX_GAMES", "3000"))
    client = SteamClient(api_key=api_key)

    if mode == "universe":
        scrape_universe(client, max_games=max_games)
    elif mode == "metadata":
        scrape_metadata(client)
    elif mode == "reviews":
        scrape_reviews(client)
    else:
        print(f"Unknown mode: {mode}. Choose universe, metadata, or reviews.")
        sys.exit(1)
