"""Check what fraction of our games have Winter 2024 review coverage."""
import json
import datetime as dt
from pathlib import Path
from collections import Counter

review_dir = Path("data/raw/reviews")
files = list(review_dir.glob("*.json"))

winter_start = dt.date(2024, 12, 19)
winter_end   = dt.date(2025,  1,  2)
pre_start    = dt.date(2024, 11, 19)

games_with_winter = 0
games_with_pre    = 0
games_total       = 0
sample_earliests  = []

for p in files[:200]:  # sample first 200
    d = json.loads(p.read_text())
    reviews = d.get("reviews", [])
    if not reviews:
        continue
    games_total += 1
    dates = [dt.datetime.fromtimestamp(r["timestamp"]).date() for r in reviews]
    earliest = min(dates)
    sample_earliests.append(earliest)
    if any(winter_start <= x <= winter_end for x in dates):
        games_with_winter += 1
    if any(pre_start <= x < winter_start for x in dates):
        games_with_pre += 1

print(f"Sample: {games_total} games")
print(f"Games with reviews IN Winter 2024 window: {games_with_winter} ({games_with_winter/games_total:.1%})")
print(f"Games with reviews in pre-period (Nov 19 - Dec 18): {games_with_pre} ({games_with_pre/games_total:.1%})")

sample_earliests.sort()
print(f"\nEarliest review date distribution (sample):")
year_months = Counter(f"{d.year}-{d.month:02d}" for d in sample_earliests)
for ym, cnt in sorted(year_months.items()):
    print(f"  {ym}: {cnt}")
