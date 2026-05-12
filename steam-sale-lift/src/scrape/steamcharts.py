"""
SteamCharts scraper — historical concurrent player counts.

SteamCharts publishes monthly peak and avg concurrent player data going back to launch.
We scrape the JSON endpoint they use internally (cleaner than HTML parsing).

Rate: 1 req/3s, no API key needed.
Run: python -m src.scrape.steamcharts
"""

import json
import logging
import re
import sys
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")

RAW_DIR = Path("data/raw")
CACHE_DIR = Path("data/raw/cache")


class SteamChartsClient:
    def __init__(self, delay: float = 3.0):
        self.client = httpx.Client(
            timeout=30.0,
            headers={
                "User-Agent": "SteamSaleLift/1.0 (academic research)",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        self.delay = delay
        self.last_call: float = 0.0

    def _throttle(self) -> None:
        elapsed = time.time() - self.last_call
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_call = time.time()

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=5, max=120))
    def get_player_history(self, appid: int) -> list[dict] | None:
        """
        Scrape monthly player history from steamcharts.com/app/{appid}.
        Returns list of {year_month, avg_players, peak_players} or None if unavailable.
        """
        cache_path = CACHE_DIR / f"charts_{appid}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        self._throttle()
        url = f"https://steamcharts.com/app/{appid}"
        r = self.client.get(url)

        if r.status_code == 404:
            return None
        if r.status_code == 429:
            logger.warning(f"SteamCharts 429 for {appid} — backing off 2 min")
            time.sleep(120)
            raise RuntimeError("rate limited")
        if r.status_code != 200:
            logger.warning(f"SteamCharts HTTP {r.status_code} for appid {appid}")
            return None

        soup = BeautifulSoup(r.text, "lxml")
        table = soup.find("table", {"id": "main-chart"})
        if not table:
            return None

        rows = []
        for tr in table.find("tbody").find_all("tr"):  # type: ignore[union-attr]
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cells) < 3:
                continue
            # Columns: Month, Avg. Players, Gain, % Gain, Peak Players
            month_str = cells[0]   # e.g. "April 2024"
            avg_raw = cells[1].replace(",", "")
            peak_raw = cells[4].replace(",", "") if len(cells) > 4 else "0"

            try:
                rows.append({
                    "year_month": month_str,
                    "avg_players": float(avg_raw) if avg_raw not in ("-", "") else None,
                    "peak_players": int(peak_raw) if peak_raw not in ("-", "") else None,
                })
            except ValueError:
                continue

        if not rows:
            return None

        cache_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        return rows


def scrape_steamcharts() -> None:
    universe_path = RAW_DIR / "universe.json"
    if not universe_path.exists():
        logger.error("Run 'python -m src.scrape.steam_api universe' first.")
        sys.exit(1)

    games = json.loads(universe_path.read_text(encoding="utf-8"))
    out_dir = RAW_DIR / "steamcharts"
    out_dir.mkdir(exist_ok=True)

    todo = [g for g in games if not (out_dir / f"{g['appid']}.json").exists()]
    logger.info(f"Fetching SteamCharts history for {len(todo)} games")

    client = SteamChartsClient()
    errors = 0
    for game in tqdm(todo, desc="steamcharts"):
        appid = game["appid"]
        history = client.get_player_history(appid)
        if history is None:
            errors += 1
            (out_dir / f"{appid}.json").write_text(
                json.dumps({"appid": appid, "skipped": True}), encoding="utf-8"
            )
        else:
            (out_dir / f"{appid}.json").write_text(
                json.dumps({"appid": appid, "history": history}, ensure_ascii=False),
                encoding="utf-8",
            )

    logger.info(f"SteamCharts done. {errors} unavailable games.")


if __name__ == "__main__":
    scrape_steamcharts()
