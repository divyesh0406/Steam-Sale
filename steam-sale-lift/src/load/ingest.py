"""
Idempotent ingestion: raw JSON dumps → Postgres mart tables.

Uses psycopg3 with executemany + ON CONFLICT DO UPDATE everywhere.
Run: python -m src.load.ingest

Order: sale_events → games (dim) → prices → reviews → players → steamspy
"""

import json
import logging
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from tqdm import tqdm

from src.scrape.steam_api import KNOWN_SALE_EVENTS

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")

RAW_DIR = Path("data/raw")

# Top-50 publishers considered "AAA" for the is_aaa flag
_AAA_PUBLISHERS = frozenset({
    "Valve", "Electronic Arts", "Ubisoft", "Activision", "Bethesda Softworks",
    "2K Games", "Square Enix", "Bandai Namco Entertainment", "SEGA", "Capcom",
    "Konami", "Warner Bros. Games", "THQ Nordic", "Focus Entertainment",
    "Deep Silver", "505 Games", "Paradox Interactive", "CD PROJEKT RED",
    "Devolver Digital", "Team17", "Kalypso Media", "Nordic Games", "Codemasters",
    "Frontier Developments", "Rebellion", "Warhorse Studios", "Dambuster Studios",
    "IO Interactive", "GIANTS Software", "Maximum Games", "Microids",
    "Nacon", "PlatinumGames", "Atlus", "NIS America", "XSEED Games",
    "Koei Tecmo", "Spike Chunsoft", "Idea Factory", "Aksys Games",
    "Ghostlight", "Zen Studios", "Curve Digital", "Raw Fury",
    "Finji", "Humble Games", "Yacht Club Games",
})


def _parse_owners(owners_raw: str) -> tuple[int, int]:
    """Parse SteamSpy owners range '200000 .. 500000' → (200000, 500000)."""
    try:
        parts = owners_raw.replace(",", "").split("..")
        return int(parts[0].strip()), int(parts[1].strip())
    except (IndexError, ValueError):
        return 0, 0


def _get_conn(url: str) -> psycopg.Connection:
    return psycopg.connect(url, autocommit=False)


def apply_schema(conn: psycopg.Connection) -> None:
    schema_path = Path(__file__).parent / "schema.sql"
    sql = schema_path.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    logger.info("Schema applied.")


def ingest_sale_events(conn: psycopg.Connection) -> None:
    rows = [
        (e["id"], e["name"], e["start"], e["end"], e["type"])
        for e in KNOWN_SALE_EVENTS
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO mart.dim_sale_events (sale_event_id, name, start_date, end_date, sale_type)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (sale_event_id) DO UPDATE
                SET name = EXCLUDED.name,
                    start_date = EXCLUDED.start_date,
                    end_date = EXCLUDED.end_date,
                    sale_type = EXCLUDED.sale_type
            """,
            rows,
        )
    conn.commit()
    logger.info(f"Upserted {len(rows)} sale events.")


def ingest_games(conn: psycopg.Connection) -> None:
    universe_path = RAW_DIR / "universe.json"
    if not universe_path.exists():
        logger.warning("universe.json not found — skipping game dim ingest.")
        return

    games = json.loads(universe_path.read_text(encoding="utf-8"))
    meta_dir = RAW_DIR / "metadata"
    spy_dir = RAW_DIR / "steamspy"

    rows = []
    for game in tqdm(games, desc="preparing games"):
        appid = game["appid"]
        meta_path = meta_dir / f"{appid}.json"
        spy_path = spy_dir / f"{appid}.json"

        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("skipped"):
            continue

        spy = {}
        if spy_path.exists():
            spy_raw = json.loads(spy_path.read_text(encoding="utf-8"))
            if not spy_raw.get("skipped"):
                spy = spy_raw

        # Extract fields from Steam Store appdetails response
        release_raw = (meta.get("release_date") or {}).get("date", "")
        release_date = None
        if release_raw:
            from datetime import datetime
            for fmt in ("%d %b, %Y", "%b %d, %Y", "%Y-%m-%d"):
                try:
                    release_date = datetime.strptime(release_raw, fmt).date().isoformat()
                    break
                except ValueError:
                    continue

        price_overview = meta.get("price_overview") or {}
        base_price_cents = price_overview.get("initial", None)

        developers = meta.get("developers") or []
        publishers = meta.get("publishers") or []
        developer = developers[0] if developers else None
        publisher = publishers[0] if publishers else None

        genres = [g["description"] for g in (meta.get("genres") or [])]
        categories = [c["description"] for c in (meta.get("categories") or [])]
        tags = list((meta.get("steam_appdetails", {}) or {}).keys())  # may be empty

        primary_genre = genres[0] if genres else None

        # is_indie: solo or small team (no known AAA publisher, indie genre tag)
        is_indie = (
            publisher not in _AAA_PUBLISHERS
            and developer not in _AAA_PUBLISHERS
            and "Indie" in genres
        )
        is_aaa = publisher in _AAA_PUBLISHERS or developer in _AAA_PUBLISHERS

        owners_raw = spy.get("owners", "0 .. 0")
        owners_lower, owners_upper = _parse_owners(owners_raw)

        rows.append((
            appid,
            meta.get("name", game.get("name", "")),
            release_date,
            developer,
            publisher,
            base_price_cents,
            is_indie,
            is_aaa,
            primary_genre,
            genres or None,
            None,  # tags — populated separately if needed
            owners_lower,
            owners_upper,
        ))

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO mart.dim_games
                (appid, name, release_date, developer, publisher, base_price_cents,
                 is_indie, is_aaa, primary_genre, genres, tags, owners_lower, owners_upper)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (appid) DO UPDATE
                SET name             = EXCLUDED.name,
                    release_date     = EXCLUDED.release_date,
                    developer        = EXCLUDED.developer,
                    publisher        = EXCLUDED.publisher,
                    base_price_cents = EXCLUDED.base_price_cents,
                    is_indie         = EXCLUDED.is_indie,
                    is_aaa           = EXCLUDED.is_aaa,
                    primary_genre    = EXCLUDED.primary_genre,
                    genres           = EXCLUDED.genres,
                    owners_lower     = EXCLUDED.owners_lower,
                    owners_upper     = EXCLUDED.owners_upper,
                    scraped_at       = NOW()
            """,
            rows,
        )
    conn.commit()
    logger.info(f"Upserted {len(rows)} games into mart.dim_games.")


def ingest_prices(conn: psycopg.Connection) -> None:
    prices_dir = RAW_DIR / "prices"
    if not prices_dir.exists():
        logger.warning("data/raw/prices/ not found — skipping price ingest.")
        return

    files = list(prices_dir.glob("*.json"))
    logger.info(f"Ingesting prices from {len(files)} files...")

    BATCH = 2000
    total = 0
    for path in tqdm(files, desc="prices"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("skipped"):
            continue
        appid = data["appid"]
        history = data.get("history", [])
        if not history:
            continue

        # Derive discount_pct: need base_price to compare against current price.
        # We use the mode (most-common price) as a proxy for the base price.
        prices = [r["price_cents"] for r in history if r.get("price_cents")]
        if not prices:
            continue
        from collections import Counter
        base_price_cents = Counter(prices).most_common(1)[0][0]

        rows = []
        for rec in history:
            price_cents = rec.get("price_cents")
            if price_cents is None:
                continue
            discount_pct = round(
                max(0.0, (base_price_cents - price_cents) / base_price_cents * 100), 2
            ) if base_price_cents > 0 else 0.0
            rows.append((
                appid,
                rec["date"],
                price_cents,
                discount_pct,
                rec.get("sale_event_id"),
            ))

        if not rows:
            continue

        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO mart.fct_prices_daily (appid, date, price_cents, discount_pct, sale_event_id)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (appid, date) DO UPDATE
                    SET price_cents   = EXCLUDED.price_cents,
                        discount_pct  = EXCLUDED.discount_pct,
                        sale_event_id = EXCLUDED.sale_event_id
                """,
                rows,
            )
        conn.commit()
        total += len(rows)

    logger.info(f"Upserted {total:,} price rows.")


def ingest_reviews(conn: psycopg.Connection) -> None:
    review_dir = RAW_DIR / "reviews"
    if not review_dir.exists():
        logger.warning("data/raw/reviews/ not found — skipping review ingest.")
        return

    files = list(review_dir.glob("*.json"))
    logger.info(f"Ingesting reviews from {len(files)} files...")

    from datetime import datetime, timezone
    total = 0
    for path in tqdm(files, desc="reviews"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("skipped"):
            continue
        appid = data["appid"]
        reviews = data.get("reviews", [])

        rows = []
        for r in reviews:
            ts = r.get("timestamp")
            if ts is None:
                continue
            review_date = datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
            rows.append((
                r["review_id"],
                appid,
                review_date,
                bool(r.get("is_positive")),
                r.get("playtime_at_review_min"),
                r.get("helpful_votes"),
                r.get("weighted_vote_score"),
            ))

        if not rows:
            continue

        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO mart.fct_reviews
                    (review_id, appid, review_date, is_positive,
                     playtime_at_review_min, helpful_votes, weighted_vote_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (review_id) DO NOTHING
                """,
                rows,
            )
        conn.commit()
        total += len(rows)

    logger.info(f"Upserted {total:,} review rows.")


def ingest_players(conn: psycopg.Connection) -> None:
    charts_dir = RAW_DIR / "steamcharts"
    if not charts_dir.exists():
        logger.warning("data/raw/steamcharts/ not found — skipping player ingest.")
        return

    files = list(charts_dir.glob("*.json"))
    logger.info(f"Ingesting player history from {len(files)} files...")

    total = 0
    for path in tqdm(files, desc="players"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("skipped"):
            continue
        appid = data["appid"]
        history = data.get("history", [])

        rows = [
            (appid, h["year_month"], h.get("avg_players"), h.get("peak_players"))
            for h in history
        ]
        if not rows:
            continue

        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO mart.fct_players_monthly (appid, year_month, avg_players, peak_players)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (appid, year_month) DO UPDATE
                    SET avg_players  = EXCLUDED.avg_players,
                        peak_players = EXCLUDED.peak_players
                """,
                rows,
            )
        conn.commit()
        total += len(rows)

    logger.info(f"Upserted {total:,} player-month rows.")


def ingest_steamspy(conn: psycopg.Connection) -> None:
    spy_dir = RAW_DIR / "steamspy"
    if not spy_dir.exists():
        logger.warning("data/raw/steamspy/ not found — skipping SteamSpy ingest.")
        return

    files = list(spy_dir.glob("*.json"))
    logger.info(f"Ingesting SteamSpy data from {len(files)} files...")

    rows = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("skipped") or data.get("appid") is None:
            continue
        owners_lower, owners_upper = _parse_owners(data.get("owners", "0 .. 0"))
        rows.append((
            data["appid"],
            owners_lower,
            owners_upper,
            data.get("positive", 0),
            data.get("negative", 0),
            data.get("average_2weeks", 0),
            data.get("median_2weeks", 0),
        ))

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO mart.fct_steamspy
                (appid, owners_lower, owners_upper, positive, negative,
                 average_playtime_2weeks, median_playtime_2weeks)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (appid) DO UPDATE
                SET owners_lower             = EXCLUDED.owners_lower,
                    owners_upper             = EXCLUDED.owners_upper,
                    positive                 = EXCLUDED.positive,
                    negative                 = EXCLUDED.negative,
                    average_playtime_2weeks  = EXCLUDED.average_playtime_2weeks,
                    median_playtime_2weeks   = EXCLUDED.median_playtime_2weeks,
                    scraped_at               = NOW()
            """,
            rows,
        )
    conn.commit()
    logger.info(f"Upserted {len(rows)} SteamSpy rows.")


def main() -> None:
    db_url = os.getenv("DATABASE_URL", "")
    if not db_url:
        logger.error(
            "DATABASE_URL not set. Add it to your .env file.\n"
            "Get a free Neon Postgres at: https://neon.tech"
        )
        sys.exit(1)

    logger.info("Connecting to Postgres...")
    with _get_conn(db_url) as conn:
        apply_schema(conn)
        ingest_sale_events(conn)
        ingest_games(conn)
        ingest_prices(conn)
        ingest_reviews(conn)
        ingest_players(conn)
        ingest_steamspy(conn)

    logger.info("All done. Run VACUUM FULL on Neon if you're near the 0.5 GB limit.")


if __name__ == "__main__":
    main()
