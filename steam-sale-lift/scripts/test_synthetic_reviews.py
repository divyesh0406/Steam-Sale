"""Quick sanity check on the synthetic review generator."""
import json, datetime as dt, os, psycopg
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
conn = psycopg.connect(os.environ["DATABASE_URL"])

import sys
sys.path.insert(0, ".")
from src.scrape.reviews_from_steamspy import _build_day_weights, _distribute_reviews, _RNG, WINDOW_START, WINDOW_DAYS, DIST_START, DIST_DAYS

# Test with Stardew Valley (appid 413150, ~1M lifetime reviews)
cur = conn.cursor()
cur.execute("""
    SELECT g.appid, g.name, g.release_date,
           sp.positive + sp.negative AS total_reviews, sp.positive
    FROM mart.dim_games g
    JOIN mart.fct_steamspy sp USING (appid)
    WHERE g.appid = 413150
""")
appid, name, release_date, total, positive = cur.fetchone()
print(f"Game: {name} (appid {appid})")
print(f"Lifetime reviews: {total:,} ({positive/total:.1%} positive)")
print(f"Release date: {release_date}")

rng = _RNG(seed=appid)
dist_weights, window_weights = _build_day_weights(release_date)
window_total = round(total * sum(window_weights) / sum(dist_weights))
counts = _distribute_reviews(window_total, window_weights, rng)
print(f"\nWindow total allocated: {sum(counts):,} (of {total:,} lifetime)")

# Check winter 2024 window
winter_start = dt.date(2024, 12, 19)
winter_end   = dt.date(2025,  1,  2)
pre_start    = dt.date(2024, 11, 19)
pre_end      = dt.date(2024, 12, 18)
post_start   = dt.date(2025,  1,  3)
post_end     = dt.date(2025,  2,  2)

def sum_window(start, end):
    total = 0
    for i in range(WINDOW_DAYS):
        d = WINDOW_START + dt.timedelta(days=i)
        if start <= d <= end:
            total += counts[i]
    return total

print(f"\nReviews in pre-period  (Nov 19 – Dec 18 2024): {sum_window(pre_start, pre_end):,}")
print(f"Reviews in sale window (Dec 19 2024 – Jan 2 2025): {sum_window(winter_start, winter_end):,}")
print(f"Reviews in post-period (Jan 3 – Feb 2 2025):  {sum_window(post_start, post_end):,}")

sale_lift = sum_window(winter_start, winter_end) / max(sum_window(pre_start, pre_end) / 30 * 15, 1) - 1
print(f"\nSale period lift vs baseline: {sale_lift:+.1%}")
conn.close()
