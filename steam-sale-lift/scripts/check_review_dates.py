import json
import datetime as dt
from pathlib import Path

p = Path("data/raw/reviews/291550.json")
if p.exists():
    d = json.loads(p.read_text())
    reviews = d.get("reviews", [])
    print(f"Total fetched: {d['total_fetched']}")
    dates = sorted([dt.datetime.fromtimestamp(r["timestamp"]).date() for r in reviews])
    if dates:
        print(f"Earliest: {dates[0]}")
        print(f"Latest:   {dates[-1]}")
        winter_2024 = [x for x in dates if dt.date(2024, 12, 19) <= x <= dt.date(2025, 1, 2)]
        print(f"Reviews in Winter 2024 window: {len(winter_2024)}")
    else:
        print("No reviews")
else:
    print("File not found")
