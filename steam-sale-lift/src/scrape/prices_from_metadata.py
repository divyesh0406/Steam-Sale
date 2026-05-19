"""
Build synthetic price history from Steam Store metadata + known sale event calendar.

Since SteamDB blocks automated scraping, we construct price history as follows:
- Base price: taken from each game's Steam Store appdetails (price_overview.initial)
- Sale periods: for each known sale event, SteamSpy appdetails includes a
  `discount` field if the game was on sale when we scraped it. For historical
  sales we mark every game as "potentially on sale" during known sale windows
  and rely on the discount_pct=0 default for non-sale days.

This gives us the price backbone needed for the analysis:
  - Full-price days: price = base_price, discount_pct = 0
  - Known sale windows: flagged with sale_event_id (discount_pct inferred from
    SteamSpy current discount if available, else left as NULL for the model to
    handle via the RDD running variable)

Output: data/raw/prices/{appid}.json  (same format as steamdb.py would produce)

Run: python -m src.scrape.prices_from_metadata
"""

import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from tqdm import tqdm

from src.scrape.steam_api import KNOWN_SALE_EVENTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")

# Date range to generate: 2019-01-01 to today
START_DATE = date(2019, 1, 1)
END_DATE = date.today()

# Pre-parse sale windows once
_SALE_RANGES: list[tuple[date, date, str]] = [
    (
        datetime.strptime(e["start"], "%Y-%m-%d").date(),
        datetime.strptime(e["end"], "%Y-%m-%d").date(),
        e["id"],
    )
    for e in KNOWN_SALE_EVENTS
]


def date_to_sale_event(d: date) -> str | None:
    for start, end, event_id in _SALE_RANGES:
        if start <= d <= end:
            return event_id
    return None


def build_price_history(appid: int, base_price_cents: int, release_date: date | None) -> list[dict]:
    """
    Generate daily price records from START_DATE (or release date) to END_DATE.
    All days get base_price; sale_event_id is set during known sale windows.
    discount_pct is left NULL — it will be filled in during analysis from
    SteamSpy current discount data where available.
    """
    start = max(START_DATE, release_date) if release_date else START_DATE
    records = []
    d = start
    while d <= END_DATE:
        records.append({
            "date": d.isoformat(),
            "price_cents": base_price_cents,
            "price_usd": round(base_price_cents / 100, 2),
            "sale_event_id": date_to_sale_event(d),
        })
        d += timedelta(days=1)
    return records


def main() -> None:
    universe_path = RAW_DIR / "universe.json"
    if not universe_path.exists():
        logger.error("Run 'python -m src.scrape.steam_api universe' first.")
        sys.exit(1)

    games = json.loads(universe_path.read_text(encoding="utf-8"))
    meta_dir = RAW_DIR / "metadata"
    out_dir = RAW_DIR / "prices"
    out_dir.mkdir(exist_ok=True)

    todo = [g for g in games if not (out_dir / f"{g['appid']}.json").exists()]
    logger.info(f"Building synthetic price history for {len(todo)} games")

    skipped = 0
    for game in tqdm(todo, desc="prices"):
        appid = game["appid"]
        meta_path = meta_dir / f"{appid}.json"

        if not meta_path.exists():
            skipped += 1
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("skipped"):
            skipped += 1
            continue

        # Base price in cents
        price_overview = meta.get("price_overview") or {}
        base_price_cents = price_overview.get("initial")
        if not base_price_cents:
            # Free-to-play or no price data — skip
            skipped += 1
            (out_dir / f"{appid}.json").write_text(
                json.dumps({"appid": appid, "skipped": True, "reason": "free_or_no_price"}),
                encoding="utf-8",
            )
            continue

        # Release date
        release_date = None
        release_raw = (meta.get("release_date") or {}).get("date", "")
        if release_raw:
            for fmt in ("%d %b, %Y", "%b %d, %Y", "%Y-%m-%d"):
                try:
                    release_date = datetime.strptime(release_raw, fmt).date()
                    break
                except ValueError:
                    continue

        history = build_price_history(appid, base_price_cents, release_date)
        (out_dir / f"{appid}.json").write_text(
            json.dumps({"appid": appid, "history": history}, ensure_ascii=False),
            encoding="utf-8",
        )

    logger.info(f"Done. {len(todo) - skipped} games with price history, {skipped} skipped.")


if __name__ == "__main__":
    main()
