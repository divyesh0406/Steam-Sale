"""Patch cell cd00cb9ff1 in 04_hypothesis_3_synthetic_control.ipynb.
tc already has is_indie, is_aaa, sale_reviews — no need to re-merge games.
"""
import json
from pathlib import Path

nb_path = Path(__file__).parent.parent / "notebooks" / "04_hypothesis_3_synthetic_control.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))

# Also update the games SQL query cell to just fetch name for display purposes
fixed_cell2_source = [
    'tc = pd.read_parquet(RESULTS / "eda_treatment_control.parquet")\n',
    '\n',
    '# Fetch game names for display (separate from tc merge)\n',
    'games = pd.read_sql("""\n',
    '    SELECT appid, name FROM stg.games\n',
    '""", conn)\n',
    '\n',
    '# tc already has is_indie, is_aaa, sale_reviews from EDA step\n',
    'tc = tc.merge(games[["appid","name"]], on="appid", how="left", suffixes=("","_db"))\n',
    'if "name_db" in tc.columns:\n',
    '    tc["name"] = tc["name_db"].fillna(tc["name"])\n',
    '    tc.drop(columns=["name_db"], inplace=True)\n',
    '\n',
    '# Select treated games with most reviews during the sale (best signal)\n',
    'treated = tc[tc["treated"]].copy()\n',
    'treated_indie = (treated[treated["is_indie"] == True]\n',
    '                 .sort_values("sale_reviews", ascending=False)\n',
    '                 .head(5)["appid"].tolist())\n',
    'treated_aaa   = (treated[treated["is_aaa"] == True]\n',
    '                 .sort_values("sale_reviews", ascending=False)\n',
    '                 .head(5)["appid"].tolist())\n',
    '\n',
    'print(f"Top 5 indie treated: {treated_indie}")\n',
    'print(f"Top 5 AAA treated:   {treated_aaa}")\n',
    '\n',
    '# Donor pool: non-treated games with data in our panel\n',
    'non_treated_appids = tc[~tc["treated"]]["appid"].tolist()\n',
    'donor_pool = [a for a in non_treated_appids if a in panel["appid"].unique()]\n',
    'print(f"Donor pool size: {len(donor_pool):,}")\n',
]

# Also fix the name lookup in the SC fit loop to use tc instead of games
fixed_fit_source = [
    'SC_KWARGS = dict(\n',
    '    date_col="review_date",\n',
    '    outcome_col="log_reviews",\n',
    '    pre_end="2024-12-18",\n',
    '    post_start="2024-12-19",\n',
    '    post_end="2025-01-02",\n',
    ')\n',
    '\n',
    'name_map = dict(zip(tc["appid"], tc["name"]))\n',
    '\n',
    'indie_results, aaa_results = [], []\n',
    'for appid in treated_indie:\n',
    '    name = name_map.get(appid, str(appid))\n',
    '    r = fit_synthetic_control(panel, appid, donor_pool, **SC_KWARGS)\n',
    '    if "error" in r:\n',
    '        print(f"  {name}: {r[\'error\']}")\n',
    '    else:\n',
    '        print(f"  {name}: pre_RMSPE={r[\'pre_rmspe\']:.4f}  mean_gap={r[\'mean_gap_post\']:+.4f}")\n',
    '        indie_results.append({"name": name, **r})\n',
    '\n',
    'print()\n',
    'for appid in treated_aaa:\n',
    '    name = name_map.get(appid, str(appid))\n',
    '    r = fit_synthetic_control(panel, appid, donor_pool, **SC_KWARGS)\n',
    '    if "error" in r:\n',
    '        print(f"  {name}: {r[\'error\']}")\n',
    '    else:\n',
    '        print(f"  {name}: pre_RMSPE={r[\'pre_rmspe\']:.4f}  mean_gap={r[\'mean_gap_post\']:+.4f}")\n',
    '        aaa_results.append({"name": name, **r})\n',
]

patched = 0
for cell in nb["cells"]:
    cid = cell.get("id", "")
    if cid == "cd00cb9ff1":
        cell["source"] = fixed_cell2_source
        cell["outputs"] = []
        cell["execution_count"] = None
        patched += 1
    elif cid == "cd03c5b213":
        cell["source"] = fixed_fit_source
        cell["outputs"] = []
        cell["execution_count"] = None
        patched += 1

print(f"Patched {patched} cells.")
nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Saved.")
