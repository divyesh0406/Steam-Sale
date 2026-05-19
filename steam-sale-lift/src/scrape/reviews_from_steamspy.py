"""
Generate synthetic daily review history from SteamSpy lifetime totals.

Method:
  - Total lifetime reviews = SteamSpy positive + negative
  - Distribute across days since release using a decay curve:
      reviews/day ∝ exp(-k * age_days) — higher near launch, tapering off
  - Apply a sale multiplier: during each known sale window, reviews/day × sale_boost
  - Normalise so the sum across all days equals the lifetime total
  - Sample an integer count per day using a Poisson draw

Output: data/raw/reviews/{appid}.json  (same schema as steam_api.py scraper)
        {"appid": int, "total_fetched": int, "reviews": [{"timestamp": int, ...}]}

Run: python -m src.scrape.reviews_from_steamspy
"""

import json
import logging
import math
import os
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")

RAW_DIR = Path("data/raw")

# Full distribution window used for decay weights (wider = more realistic distribution)
DIST_START = date(2022, 1, 1)
DIST_END   = date(2026, 5, 18)
DIST_DAYS  = (DIST_END - DIST_START).days + 1

# Output window: only emit rows within this range to stay under 512 MB
# Covers Winter 2024 pre-period (30 days before) through post-period (30 days after)
# Nov 19 2024 – Feb 2 2025 = 76 days
WINDOW_START = date(2024, 11, 19)
WINDOW_END   = date(2025,  2,  2)
WINDOW_DAYS  = (WINDOW_END - WINDOW_START).days + 1

# Sale multipliers: boost review velocity during sale windows
# Format: (start_date, end_date, multiplier)
SALE_BOOSTS = [
    (date(2024,  3, 14), date(2024,  3, 21), 2.5),   # spring_2024
    (date(2024,  6, 27), date(2024,  7, 11), 4.0),   # summer_2024
    (date(2024, 11, 27), date(2024, 12,  3), 2.0),   # autumn_2024
    (date(2024, 12, 19), date(2025,  1,  2), 4.0),   # winter_2024
    (date(2025,  1, 23), date(2025,  1, 27), 1.5),   # lunar_2025
    (date(2025,  3, 13), date(2025,  3, 20), 2.5),   # spring_2025
    (date(2025,  6, 26), date(2025,  7, 10), 4.0),   # summer_2025
]

# Decay constant: half-life ~2 years (reviews taper off with age)
DECAY_K = math.log(2) / 730


def _build_day_weights(release_date: date) -> tuple[list[float], list[float]]:
    """
    Return (dist_weights, window_weights).
    dist_weights: weight for every day in [DIST_START, DIST_END] — used to
                  proportion how many lifetime reviews fall in each day.
    window_weights: the slice of dist_weights covering [WINDOW_START, WINDOW_END]
                    — used to allocate output rows.
    """
    dist_weights = []
    for i in range(DIST_DAYS):
        d = DIST_START + timedelta(days=i)
        if d < release_date:
            dist_weights.append(0.0)
            continue
        age_days = (d - release_date).days
        w = math.exp(-DECAY_K * age_days)
        for sale_start, sale_end, mult in SALE_BOOSTS:
            if sale_start <= d <= sale_end:
                w *= mult
                break
        dist_weights.append(w)

    # Slice to output window
    window_offset = (WINDOW_START - DIST_START).days
    window_weights = dist_weights[window_offset: window_offset + WINDOW_DAYS]
    return dist_weights, window_weights


def _distribute_reviews(total: int, weights: list[float], rng: random.Random) -> list[int]:
    """
    Allocate `total` reviews across days proportionally to `weights`,
    with Poisson noise to avoid perfectly smooth counts.
    """
    total_weight = sum(weights)
    if total_weight == 0:
        return [0] * len(weights)

    # Expected reviews per day
    expected = [w / total_weight * total for w in weights]

    # Poisson draw with rescaling to hit the exact total
    counts = [rng.poisson_int(e) for e in expected]

    # Rescale to match total exactly (add/subtract from busiest days)
    diff = total - sum(counts)
    if diff != 0:
        nonzero = sorted(
            [i for i, w in enumerate(weights) if w > 0],
            key=lambda i: -weights[i]
        )
        for i in nonzero:
            if diff == 0:
                break
            delta = min(abs(diff), max(1, counts[i] // 10))
            if diff > 0:
                counts[i] += delta
                diff -= delta
            else:
                adj = min(delta, counts[i])
                counts[i] -= adj
                diff += adj

    return counts


class _RNG:
    """Minimal Poisson sampler using standard library random."""
    def __init__(self, seed: int):
        self._r = random.Random(seed)

    def poisson_int(self, lam: float) -> int:
        if lam <= 0:
            return 0
        if lam > 50:
            # Normal approximation for large lambda
            val = self._r.gauss(lam, math.sqrt(lam))
            return max(0, round(val))
        # Knuth algorithm
        L = math.exp(-lam)
        k, p = 0, 1.0
        while p > L:
            k += 1
            p *= self._r.random()
        return k - 1


def generate_reviews(conn: psycopg.Connection) -> None:
    review_dir = RAW_DIR / "reviews"
    review_dir.mkdir(exist_ok=True)

    cur = conn.cursor()
    cur.execute("""
        SELECT g.appid, g.name, g.release_date,
               COALESCE(sp.positive, 0) + COALESCE(sp.negative, 0) AS total_reviews,
               COALESCE(sp.positive, 0) AS positive_count
        FROM mart.dim_games g
        LEFT JOIN mart.fct_steamspy sp USING (appid)
        WHERE g.release_date IS NOT NULL
          AND (COALESCE(sp.positive, 0) + COALESCE(sp.negative, 0)) > 0
        ORDER BY g.appid
    """)
    rows = cur.fetchall()
    logger.info(f"Generating synthetic reviews for {len(rows):,} games...")

    for appid, name, release_date, total_reviews, positive_count in tqdm(rows, desc="synthetic reviews"):
        out_path = review_dir / f"{appid}.json"
        rng = _RNG(seed=appid)

        dist_weights, window_weights = _build_day_weights(release_date)

        # Fraction of lifetime reviews that fall in the output window
        dist_total = sum(dist_weights)
        if dist_total == 0:
            out_path.write_text(
                json.dumps({"appid": appid, "total_fetched": 0, "reviews": []},
                           ensure_ascii=False), encoding="utf-8"
            )
            continue

        window_fraction = sum(window_weights) / dist_total
        window_total = round(total_reviews * window_fraction)

        counts = _distribute_reviews(window_total, window_weights, rng)

        # Positive rate from SteamSpy
        pos_rate = positive_count / total_reviews if total_reviews > 0 else 0.7

        reviews = []
        for i, count in enumerate(counts):
            if count == 0:
                continue
            day = WINDOW_START + timedelta(days=i)
            for j in range(count):
                hour = (j * 86400 // max(count, 1)) % 86400
                ts = int(datetime(day.year, day.month, day.day).timestamp()) + hour
                reviews.append({
                    "review_id": f"syn_{appid}_{i}_{j}",
                    "is_positive": rng._r.random() < pos_rate,
                    "timestamp": ts,
                    "playtime_at_review_min": max(0, round(rng._r.gauss(120, 80))),
                    "helpful_votes": 0,
                    "weighted_vote_score": 0,
                })

        out_path.write_text(
            json.dumps({"appid": appid, "total_fetched": len(reviews), "reviews": reviews},
                       ensure_ascii=False),
            encoding="utf-8",
        )

    logger.info(f"Done. Generated synthetic reviews for {len(rows):,} games.")


def main() -> None:
    db_url = os.getenv("DATABASE_URL", "")
    if not db_url:
        logger.error("DATABASE_URL not set.")
        sys.exit(1)
    with psycopg.connect(db_url) as conn:
        generate_reviews(conn)


if __name__ == "__main__":
    main()
