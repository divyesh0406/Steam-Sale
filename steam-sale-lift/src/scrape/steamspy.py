"""
SteamSpy scraper — per-game owner estimates and player activity.

SteamSpy is more lenient than Steam on rate limits but still requires politeness.
We use ~1 req/1.5s and cache aggressively.

Run: python -m src.scrape.steamspy
"""

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

RAW_DIR = Path("data/raw")
CACHE_DIR = Path("data/raw/cache")


class SteamSpyClient:
    BASE_URL = "https://steamspy.com/api.php"

    def __init__(self, delay: float = 1.5):
        self.client = httpx.Client(
            timeout=30.0,
            headers={"User-Agent": "SteamSaleLift/1.0 (academic research)"},
        )
        self.delay = delay
        self.last_call: float = 0.0

    def _throttle(self) -> None:
        elapsed = time.time() - self.last_call
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_call = time.time()

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
    def get_app(self, appid: int) -> dict | None:
        cache_path = CACHE_DIR / f"spy_{appid}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        self._throttle()
        r = self.client.get(self.BASE_URL, params={"request": "appdetails", "appid": str(appid)})

        if r.status_code == 429:
            logger.warning("SteamSpy 429 — backing off 60s")
            time.sleep(60)
            raise RuntimeError("rate limited")

        r.raise_for_status()
        data = r.json()
        if not data or data.get("appid") is None:
            return None

        cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data


def scrape_steamspy() -> None:
    universe_path = RAW_DIR / "universe.json"
    if not universe_path.exists():
        logger.error("Run 'scrape universe' first (python -m src.scrape.steam_api universe).")
        sys.exit(1)

    games = json.loads(universe_path.read_text(encoding="utf-8"))
    out_dir = RAW_DIR / "steamspy"
    out_dir.mkdir(exist_ok=True)

    todo = [g for g in games if not (out_dir / f"{g['appid']}.json").exists()]
    logger.info(f"Fetching SteamSpy details for {len(todo)} games")

    client = SteamSpyClient()
    errors = 0
    for game in tqdm(todo, desc="steamspy"):
        appid = game["appid"]
        data = client.get_app(appid)
        if data is None:
            errors += 1
            (out_dir / f"{appid}.json").write_text(
                json.dumps({"appid": appid, "skipped": True}), encoding="utf-8"
            )
        else:
            (out_dir / f"{appid}.json").write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8"
            )

    logger.info(f"SteamSpy done. {errors} errors.")


if __name__ == "__main__":
    scrape_steamspy()
