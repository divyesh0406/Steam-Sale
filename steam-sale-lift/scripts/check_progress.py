import json
from pathlib import Path

universe = json.loads(Path("data/raw/universe.json").read_text(encoding="utf-8"))
for i, g in enumerate(universe):
    if g["appid"] == 242760:
        pct = (i + 1) / len(universe) * 100
        elapsed_min = 15.2
        remaining_pct = 100 - pct
        eta_min = elapsed_min * remaining_pct / pct if pct > 0 else 999
        print(f"appid 242760 = {g['name']} at position {i+1}/{len(universe)}")
        print(f"Progress: {pct:.1f}%")
        print(f"ETA: ~{eta_min:.0f} more minutes at current rate")
        break
