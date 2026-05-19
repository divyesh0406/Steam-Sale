"""Check which appid the scraper is currently on and estimate progress."""
import json
import re
from pathlib import Path

log_path = Path(r"C:\Users\divye\AppData\Local\Temp\claude\d--Steam-Sale\87d4ae97-49b1-4831-8e0e-0602a931d24a\tasks\bgmmrzgkz.output")
universe = json.loads(Path("data/raw/universe.json").read_text(encoding="utf-8"))
appid_to_pos = {g["appid"]: (i, g["name"]) for i, g in enumerate(universe)}

# Find the last appid mentioned in the log
content = log_path.read_text(encoding="utf-8", errors="replace")
matches = re.findall(r"/appreviews/(\d+)\?", content)
if matches:
    last_appid = int(matches[-1])
    if last_appid in appid_to_pos:
        pos, name = appid_to_pos[last_appid]
        pct = (pos + 1) / len(universe) * 100
        elapsed_min = ((__import__('time').time() - 1747611000) / 60)  # seconds since ~21:30 start
        eta_min = elapsed_min * (100 - pct) / pct if pct > 0 else 999
        print(f"Currently on: appid {last_appid} = {name}")
        print(f"Position: {pos+1}/{len(universe)} ({pct:.1f}%)")
        print(f"ETA: ~{eta_min:.0f} more minutes")
    else:
        print(f"Last appid: {last_appid} (not in universe)")
