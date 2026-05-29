"""
Run synthetic control analysis and save results to results/h3_sc_results.json.
Replaces notebook 04 execution for portability and speed.
"""
import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import psycopg
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.analysis.synthetic_control import fit_synthetic_control, placebo_test

load_dotenv()
RESULTS = Path(__file__).parent.parent / "results"
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})

PANEL_CACHE = RESULTS / "panel_daily.parquet"

# --- 1. Build panel (cache locally to avoid slow repeated DB reads) ---
if PANEL_CACHE.exists():
    print("Loading panel from cache...")
    panel = pd.read_parquet(PANEL_CACHE)
else:
    print("Connecting to DB and loading panel (first run — will cache)...")
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    rows = conn.execute("""
        SELECT appid, review_date::TEXT AS review_date, review_count
        FROM mart.fct_reviews_daily
        ORDER BY review_date, appid
    """).fetchall()
    conn.close()
    panel = pd.DataFrame(rows, columns=["appid", "review_date", "review_count"])
    panel.to_parquet(PANEL_CACHE, index=False)
    print(f"Cached panel to {PANEL_CACHE}")

panel["log_reviews"] = np.log1p(panel["review_count"])
print(f"Panel rows: {len(panel):,}  Games: {panel['appid'].nunique():,}")

# --- 2. Select treated / donor units ---
print("Loading treatment assignments...")
tc = pd.read_parquet(RESULTS / "eda_treatment_control.parquet")

panel_appids = set(panel["appid"].unique())
treated = tc[tc["treated"]].copy()
treated_indie = (treated[treated["is_indie"] == True]
                 .sort_values("sale_reviews", ascending=False)
                 .head(5)["appid"].tolist())
treated_aaa   = (treated[treated["is_aaa"] == True]
                 .sort_values("sale_reviews", ascending=False)
                 .head(5)["appid"].tolist())

print(f"Top 5 indie treated: {treated_indie}")
print(f"Top 5 AAA treated:   {treated_aaa}")

non_treated = tc[~tc["treated"]]["appid"].tolist()
donor_pool_full = [a for a in non_treated if a in panel_appids]
print(f"Donor pool size (full): {len(donor_pool_full):,}")

# Cap donors at 200 for tractable SLSQP (1886 variables causes solver to hang)
rng_cap = np.random.default_rng(42)
donor_pool = rng_cap.choice(donor_pool_full, size=min(200, len(donor_pool_full)), replace=False).tolist()
print(f"Donor pool size (capped): {len(donor_pool)}")

SC_KWARGS = dict(
    date_col="review_date",
    outcome_col="log_reviews",
    pre_end="2024-12-18",
    post_start="2024-12-19",
    post_end="2025-01-02",
)

name_map = dict(zip(tc["appid"], tc["name"]))

# --- 3. Fit synthetic controls ---
print("\nFitting indie synthetic controls...")
indie_results, aaa_results = [], []
for appid in treated_indie:
    name = name_map.get(appid, str(appid))
    r = fit_synthetic_control(panel, appid, donor_pool, **SC_KWARGS)
    if "error" in r:
        print(f"  {name}: {r['error']}")
    else:
        print(f"  {name}: pre_RMSPE={r['pre_rmspe']:.4f}  mean_gap={r['mean_gap_post']:+.4f}")
        indie_results.append({"name": name, **r})

print("\nFitting AAA synthetic controls...")
for appid in treated_aaa:
    name = name_map.get(appid, str(appid))
    r = fit_synthetic_control(panel, appid, donor_pool, **SC_KWARGS)
    if "error" in r:
        print(f"  {name}: {r['error']}")
    else:
        print(f"  {name}: pre_RMSPE={r['pre_rmspe']:.4f}  mean_gap={r['mean_gap_post']:+.4f}")
        aaa_results.append({"name": name, **r})

# --- 4. Gap plots ---
def plot_gaps(results, title, ax):
    for r in results:
        gs = r.get("gap_series")
        if gs is not None and len(gs):
            ax.plot(pd.to_datetime(gs.index), gs.values,
                    alpha=0.6, lw=1.5, label=r["name"][:25])
    ax.axhline(0, color="black", lw=1.5, ls="--")
    ax.axvline(pd.Timestamp("2024-12-19"), color="black", ls=":", lw=1)
    ax.set_ylabel("Actual − Synthetic (log reviews)")
    ax.set_title(title)
    ax.legend(fontsize=8)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
plot_gaps(indie_results, "Indie treated games", axes[0])
plot_gaps(aaa_results,   "AAA treated games",   axes[1])
plt.suptitle("Synthetic control gap plots (positive = sale lift above synthetic)", y=1.02)
plt.tight_layout()
plt.savefig(RESULTS / "fig_sc_gap_plots.png", dpi=150, bbox_inches="tight")
print("\nSaved fig_sc_gap_plots.png")

# --- 5. Placebo test ---
print("\nRunning placebo test (n=50)... this takes a few minutes")
all_treated = treated_indie + treated_aaa
placebos = placebo_test(panel, all_treated, donor_pool, n_placebos=50, seed=42, **SC_KWARGS)
print(f"Placebo tests completed: {placebos['n_placebos']}")

placebo_gaps = [r["mean_gap"] for r in placebos["placebo_results"]]
indie_mean_gap = np.mean([r["mean_gap_post"] for r in indie_results]) if indie_results else np.nan
aaa_mean_gap   = np.mean([r["mean_gap_post"] for r in aaa_results])   if aaa_results   else np.nan

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(placebo_gaps, bins=25, color="steelblue", edgecolor="white", alpha=0.7, label="Placebo gaps")
if not np.isnan(indie_mean_gap):
    ax.axvline(indie_mean_gap, color="green", lw=2, label=f"Indie mean gap = {indie_mean_gap:+.3f}")
if not np.isnan(aaa_mean_gap):
    ax.axvline(aaa_mean_gap, color="orange", lw=2, label=f"AAA mean gap = {aaa_mean_gap:+.3f}")
ax.axvline(0, color="black", lw=1, ls="--")
ax.set_xlabel("Mean post-sale gap (actual − synthetic)")
ax.set_title("Placebo distribution vs treated gaps")
ax.legend()
plt.tight_layout()
plt.savefig(RESULTS / "fig_sc_placebo.png", dpi=150)
print("Saved fig_sc_placebo.png")

# --- 6. p-values and save ---
p_indie = float(np.mean(np.array(placebo_gaps) >= indie_mean_gap)) if not np.isnan(indie_mean_gap) else None
p_aaa   = float(np.mean(np.array(placebo_gaps) >= aaa_mean_gap))   if not np.isnan(aaa_mean_gap)   else None

if p_indie is not None:
    print(f"\nIndie mean gap: {indie_mean_gap:+.4f}  one-sided p = {p_indie:.3f}")
if p_aaa is not None:
    print(f"AAA mean gap:   {aaa_mean_gap:+.4f}  one-sided p = {p_aaa:.3f}")

het_table = []
for tier, results in [("Indie", indie_results), ("AAA", aaa_results)]:
    for r in results:
        het_table.append({
            "tier": tier,
            "name": r["name"],
            "pre_rmspe": r["pre_rmspe"],
            "mean_gap_post": r["mean_gap_post"],
            "n_donors": r["n_donors"],
        })

het_df = pd.DataFrame(het_table)
if not het_df.empty:
    print("\nHeterogeneity table:")
    print(het_df.groupby("tier")[["pre_rmspe","mean_gap_post"]].agg(["mean","std"]).round(4))

h3_results = {
    "indie_mean_gap": float(indie_mean_gap) if not np.isnan(indie_mean_gap) else None,
    "aaa_mean_gap":   float(aaa_mean_gap)   if not np.isnan(aaa_mean_gap)   else None,
    "p_indie": p_indie,
    "p_aaa":   p_aaa,
    "n_indie_treated": len(indie_results),
    "n_aaa_treated":   len(aaa_results),
    "n_placebos": placebos["n_placebos"],
    "heterogeneity": het_df.to_dict(orient="records") if not het_df.empty else [],
}

with open(RESULTS / "h3_sc_results.json", "w") as f:
    json.dump(h3_results, f, indent=2)
print(f"\nSaved h3_sc_results.json")
print(json.dumps({k: v for k, v in h3_results.items() if k != "heterogeneity"}, indent=2))
