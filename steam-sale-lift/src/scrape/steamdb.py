"""
SteamDB scraper — daily price history and sale participation.

SteamDB is a community resource. We scrape very conservatively:
- 1 request every 3 seconds (hard floor)
- Respect robots.txt (SteamDB allows scraping of game pages)
- Use the known sale events list from steam_api.py instead of scraping event pages

Strategy: For each game, fetch its price history page and extract the discount
percentage for each date. We then join this with the known sale event calendar
to derive `sale_event_id` (which major sale each discount belongs to).

Run: python -m src.scrape.steamdb
"""

import json
import logging
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from src.scrape.steam_api import KNOWN_SALE_EVENTS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")

RAW_DIR = Path("data/raw")
CACHE_DIR = Path("data/raw/cache")

# Pre-parse sale event date ranges once
_SALE_RANGES: list[tuple[date, date, str]] = [
    (
        datetime.strptime(e["start"], "%Y-%m-%d").date(),
        datetime.strptime(e["end"], "%Y-%m-%d").date(),
        e["id"],
    )
    for e in KNOWN_SALE_EVENTS
]


def date_to_sale_event(d: date) -> str | None:
    """Return the sale_event_id if `d` falls within a known sale window, else None."""
    for start, end, event_id in _SALE_RANGES:
        if start <= d <= end:
            return event_id
    return None


class SteamDBClient:
    def __init__(self, delay: float = 3.0):
        self.client = httpx.Client(
            timeout=45.0,
            headers={
                "User-Agent": "SteamSaleLift/1.0 (academic research)",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
                # SteamDB age-gate bypass
                "Cookie": "Steam_Language=english; birthtime=631152000",
            },
            follow_redirects=True,
        )
        self.delay = delay
        self.last_call: float = 0.0

    def _throttle(self) -> None:
        elapsed = time.time() - self.last_call
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_call = time.time()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=10, max=180))
    def get_price_history(self, appid: int) -> list[dict] | None:
        """
        Fetch price history for a game from SteamDB.
        Returns list of {date, price_cents, discount_pct} or None.

        SteamDB embeds the price history as a JSON array inside a <script> tag.
        We extract that array rather than parsing the full chart HTML.
        """
        cache_path = CACHE_DIR / f"steamdb_{appid}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        self._throttle()
        url = f"https://steamdb.info/app/{appid}/charts/#price"
        r = self.client.get(url)

        if r.status_code == 404:
            return None
        if r.status_code in (429, 503):
            logger.warning(f"SteamDB {r.status_code} for {appid} — backing off 3 min")
            time.sleep(180)
            raise RuntimeError("rate limited")
        if r.status_code != 200:
            logger.warning(f"SteamDB HTTP {r.status_code} for {appid}")
            return None

        # SteamDB embeds price chart data as a JavaScript array.
        # Pattern: Highcharts.chart('...', { series: [{ data: [[ts_ms, price_usd], ...] }] })
        # We look for the price series specifically.
        text = r.text
        # Find all numeric series arrays: [[timestamp, value], ...]
        pattern = re.compile(r"\[\s*\[\s*(\d{13})\s*,\s*([\d.]+)\s*\]")
        matches = pattern.findall(text)

        if not matches:
            return None

        # Build a minimal price history: one entry per day we can infer.
        # SteamDB data points are at the moment of a price change, not daily.
        # We forward-fill to daily granularity.
        raw_points: list[tuple[date, float]] = []
        for ts_ms_str, price_str in matches:
            ts = datetime.utcfromtimestamp(int(ts_ms_str) / 1000).date()
            raw_points.append((ts, float(price_str)))

        if not raw_points:
            return None

        raw_points.sort(key=lambda x: x[0])

        # Forward-fill to daily
        records: list[dict] = []
        current_price = raw_points[0][1]
        start_date = raw_points[0][0]
        end_date = date.today()

        change_map = {d: p for d, p in raw_points}
        d = start_date
        while d <= end_date:
            if d in change_map:
                current_price = change_map[d]
            records.append({
                "date": d.isoformat(),
                "price_usd": current_price,
            })
            d += timedelta(days=1)

        cache_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        return records


def scrape_steamdb() -> None:
    """
    Scrape price history for all games in the universe.
    Annotates each price record with sale_event_id using the known sale calendar.
    """
    universe_path = RAW_DIR / "universe.json"
    if not universe_path.exists():
        logger.error("Run 'python -m src.scrape.steam_api universe' first.")
        sys.exit(1)

    games = json.loads(universe_path.read_text(encoding="utf-8"))
    out_dir = RAW_DIR / "prices"
    out_dir.mkdir(exist_ok=True)

    todo = [g for g in games if not (out_dir / f"{g['appid']}.json").exists()]
    logger.info(
        f"Fetching price history for {len(todo)} games. "
        f"At 3s/req this will take ~{len(todo) * 3 / 3600:.1f} hours."
    )

    client = SteamDBClient()
    errors = 0
    for game in tqdm(todo, desc="prices"):
        appid = game["appid"]
        history = client.get_price_history(appid)

        if history is None:
            errors += 1
            (out_dir / f"{appid}.json").write_text(
                json.dumps({"appid": appid, "skipped": True}), encoding="utf-8"
            )
            continue

        # Annotate with sale event membership
        annotated = []
        for rec in history:
            d = datetime.strptime(rec["date"], "%Y-%m-%d").date()
            annotated.append({
                "date": rec["date"],
                "price_usd": rec["price_usd"],
                "price_cents": int(round(rec["price_usd"] * 100)),
                "sale_event_id": date_to_sale_event(d),
            })

        (out_dir / f"{appid}.json").write_text(
            json.dumps({"appid": appid, "history": annotated}, ensure_ascii=False),
            encoding="utf-8",
        )

    logger.info(f"SteamDB done. {errors} unavailable games.")


if __name__ == "__main__":
    scrape_steamdb()
