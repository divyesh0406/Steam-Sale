"""Create the three Phase 4 analysis notebooks."""
import json
from pathlib import Path

NOTEBOOKS_DIR = Path("notebooks")


def nb(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "cells": cells,
    }


def md(source):
    return {"cell_type": "markdown", "id": f"md{hash(source)%10**8:08x}",
            "metadata": {}, "source": source if isinstance(source, list) else [source]}


def code(source):
    return {"cell_type": "code", "id": f"cd{hash(source)%10**8:08x}",
            "metadata": {}, "execution_count": None,
            "outputs": [],
            "source": source if isinstance(source, list) else [source]}


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 02: H1 — DiD + CUPED
# ─────────────────────────────────────────────────────────────────────────────
nb02 = nb([
    md("# H1 — Difference-in-Differences with CUPED\n\n"
       "**Hypothesis:** Steam seasonal sales cause a net-positive review velocity lift "
       "in the 30 days following the sale.\n\n"
       "**Method:** 2×2 DiD with CUPED variance reduction. Treatment proxy: games "
       "currently discounted in SteamSpy (May 2026 snapshot). Outcome: log review count "
       "during/after the Winter 2024 sale vs. 30-day pre-period."),

    code("""\
import os, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from dotenv import load_dotenv
import psycopg

sys.path.insert(0, str(Path("..").resolve()))
from src.analysis.did import simple_did, did_with_cuped, permutation_test
from src.analysis.cuped import cuped_adjust, variance_reduction

load_dotenv()
RESULTS = Path("../results")
RESULTS.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
conn = psycopg.connect(os.environ["DATABASE_URL"])
print("Connected.")
"""),

    md("## 1. Load treatment/control panel"),

    code("""\
tc = pd.read_parquet(RESULTS / "eda_treatment_control.parquet")
print(f"Total units: {len(tc):,}")
print(f"Treated: {tc['treated'].sum():,}  Control: {(~tc['treated']).sum():,}")
tc[["pre_reviews","sale_reviews","post_reviews","log_lift","treated"]].describe().round(2)
"""),

    md("## 2. Naïve estimate vs DiD"),

    code("""\
tc["log_pre"]  = np.log1p(tc["pre_reviews"])
tc["log_post"] = np.log1p(tc["post_reviews"])

naive = simple_did(tc, "log_pre", "log_post", "treated")
print("=== Naïve DiD (no CUPED) ===")
print(f"  ATT:  {naive['att']:+.4f}")
print(f"  SE:   {naive['se']:.4f}")
print(f"  95%CI: [{naive['ci_low']:+.4f}, {naive['ci_high']:+.4f}]")
print(f"  Treated Δ: {naive['delta_treated']:+.4f}  Control Δ: {naive['delta_control']:+.4f}")
"""),

    md("## 3. CUPED-adjusted DiD"),

    code("""\
result = did_with_cuped(tc, "log_pre", "log_post", "treated")

print("=== CUPED-adjusted DiD ===")
print(f"  ATT:  {result['cuped_att']:+.4f}")
print(f"  SE:   {result['cuped_se']:.4f}")
print(f"  95%CI: [{result['cuped_ci_low']:+.4f}, {result['cuped_ci_high']:+.4f}]")
print(f"  theta (CUPED coefficient): {result['theta']:.4f}")
print(f"  Variance reduction: {result['variance_reduction']:.1%}")
print(f"  SE reduction: {1 - result['cuped_se']/result['naive_se']:.1%}")
sig = "SIGNIFICANT" if result['cuped_ci_low'] > 0 or result['cuped_ci_high'] < 0 else "not significant"
print(f"  Result: {sig} at 95% confidence")
"""),

    md("## 4. Parallel trends plot"),

    code("""\
import datetime as dt

monthly = pd.read_sql(\"\"\"
    SELECT
        DATE_TRUNC('month', review_date)::DATE AS month,
        appid,
        SUM(review_count) AS monthly_reviews
    FROM mart.fct_reviews_daily
    GROUP BY 1, 2
\"\"\", conn, parse_dates=["month"])

monthly_agg = (
    monthly
    .merge(tc[["appid","treated"]], on="appid")
    .groupby(["month","treated"])["monthly_reviews"]
    .mean()
    .reset_index()
    .rename(columns={"monthly_reviews": "avg_reviews"})
)

fig, ax = plt.subplots(figsize=(11, 4))
for treated_val, label, color in [(True, "Treated (on sale)", "steelblue"),
                                   (False, "Control (not on sale)", "coral")]:
    d = monthly_agg[monthly_agg["treated"] == treated_val]
    ax.plot(d["month"], d["avg_reviews"], label=label, color=color, lw=2, marker="o", ms=4)

ax.axvline(pd.Timestamp("2024-12-19"), color="black", ls="--", lw=1.5, label="Sale start")
ax.axvline(pd.Timestamp("2025-01-02"), color="black", ls=":",  lw=1.5, label="Sale end")
ax.set_ylabel("Avg monthly reviews per game")
ax.set_title("Parallel trends check — avg review velocity by treatment group")
ax.legend()
plt.tight_layout()
plt.savefig(RESULTS / "fig_did_parallel_trends.png", dpi=150)
plt.show()
print("If lines move together before Dec 2024, the parallel-trends assumption is credible.")
"""),

    md("## 5. Permutation test (randomisation inference)"),

    code("""\
perm = permutation_test(tc, "log_pre", "log_post", "treated",
                        n_permutations=1000, seed=42)

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(perm["null_distribution"], bins=50, color="steelblue", edgecolor="white", alpha=0.7)
ax.axvline(perm["observed_att"], color="crimson", lw=2,
           label=f"Observed ATT = {perm['observed_att']:+.4f}")
ax.set_xlabel("ATT under permuted treatment")
ax.set_ylabel("Count")
ax.set_title(f"Permutation test (n=1,000)  p = {perm['p_value']:.3f}")
ax.legend()
plt.tight_layout()
plt.savefig(RESULTS / "fig_did_permutation.png", dpi=150)
plt.show()
print(f"Two-sided p-value: {perm['p_value']:.3f}")
"""),

    md("## 6. Heterogeneity by game tier"),

    code("""\
for tier, mask_col in [("Indie", "is_indie"), ("AAA", "is_aaa")]:
    sub = tc[tc[mask_col] == True].copy() if mask_col in tc.columns else tc.copy()
    if len(sub[sub["treated"]]) < 5:
        print(f"{tier}: too few treated ({len(sub[sub['treated']])}) — skipping")
        continue
    r = did_with_cuped(sub, "log_pre", "log_post", "treated")
    print(f"{tier:6s}  ATT={r['cuped_att']:+.4f}  SE={r['cuped_se']:.4f}  "
          f"CI=[{r['cuped_ci_low']:+.4f}, {r['cuped_ci_high']:+.4f}]  "
          f"VR={r['variance_reduction']:.1%}  n_t={r['n_treated']}")
"""),

    md("## 7. Save results"),

    code("""\
h1_results = {
    "naive_att":  result["naive_att"],
    "naive_se":   result["naive_se"],
    "cuped_att":  result["cuped_att"],
    "cuped_se":   result["cuped_se"],
    "cuped_ci_low":  result["cuped_ci_low"],
    "cuped_ci_high": result["cuped_ci_high"],
    "variance_reduction": result["variance_reduction"],
    "permutation_p": perm["p_value"],
    "n_treated": result["n_treated"],
    "n_control": result["n_control"],
}
with open(RESULTS / "h1_did_results.json", "w") as f:
    json.dump(h1_results, f, indent=2)
print(json.dumps(h1_results, indent=2))
conn.close()
"""),
])

# ─────────────────────────────────────────────────────────────────────────────
# Notebook 03: H2 — RDD
# ─────────────────────────────────────────────────────────────────────────────
nb03 = nb([
    md("# H2 — Regression Discontinuity on Discount Depth\n\n"
       "**Hypothesis:** Games discounted ≥50% receive more reviews in the 14 days "
       "after sale start than games discounted <50%, due to Steam's front-page "
       "visibility threshold.\n\n"
       "**Running variable:** SteamSpy discount % (current snapshot).  "
       "**Cutoff:** 50%.  "
       "**Outcome:** Review count during Winter 2024 sale window (14 days)."),

    code("""\
import os, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

sys.path.insert(0, str(Path("..").resolve()))
from src.analysis.rdd import (sharp_rdd, bandwidth_sensitivity,
                               placebo_cutoffs, mccrary_density_test, bootstrap_rdd)

load_dotenv()
RESULTS = Path("../results")
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
print("Ready.")
"""),

    md("## 1. Load RDD sample"),

    code("""\
rdd = pd.read_parquet(RESULTS / "eda_rdd_sample.parquet")
print(f"Games with discount data: {len(rdd):,}")
print(f"Discount range: {rdd['discount_pct'].min()}% – {rdd['discount_pct'].max()}%")
print(f"≥50% off: {(rdd['discount_pct'] >= 50).sum():,}   <50% off: {(rdd['discount_pct'] < 50).sum():,}")

# Log-transform outcome for normality
rdd["log_reviews"] = np.log1p(rdd["reviews_14d_post"])
rdd["log_reviews_pre"] = np.log1p(rdd["reviews_30d_pre"])
print(rdd[["discount_pct","reviews_14d_post","log_reviews"]].describe().round(2))
"""),

    md("## 2. Main RDD estimate (bandwidth = 15%)"),

    code("""\
main = sharp_rdd(rdd, outcome="log_reviews", bandwidth=15.0)
print("=== RDD at 50% cutoff, BW=15 ===")
print(f"  Effect at cutoff: {main['effect_at_cutoff']:+.4f}")
print(f"  SE:   {main['se']:.4f}")
print(f"  95%CI: [{main['ci_low']:+.4f}, {main['ci_high']:+.4f}]")
print(f"  p-value: {main['p_value']:.3f}")
print(f"  N obs: {main['n_obs']}  (above={main['n_above']}, below={main['n_below']})")
sig = "SIGNIFICANT" if main['p_value'] < 0.05 else "not significant"
print(f"  Result: {sig} at 5%")
"""),

    md("## 3. RDD plot — fitted lines either side of cutoff"),

    code("""\
cutoff = 50.0
bw = 15.0
band = rdd[(rdd["discount_pct"] >= cutoff - bw) & (rdd["discount_pct"] <= cutoff + bw)].copy()

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(band["discount_pct"], band["log_reviews"], alpha=0.35, s=15, color="steelblue")

for above in [False, True]:
    sub = band[band["discount_pct"] >= cutoff] if above else band[band["discount_pct"] < cutoff]
    if len(sub) < 3:
        continue
    from numpy.polynomial import polynomial as P
    x = sub["discount_pct"].values
    y = sub["log_reviews"].values
    coeffs = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, np.polyval(coeffs, xs), color="crimson", lw=2.5)

ax.axvline(cutoff, color="black", ls="--", lw=1.5, label=f"Cutoff ({cutoff}%)")
ax.set_xlabel("Discount %")
ax.set_ylabel("log(1 + reviews during sale window)")
ax.set_title(f"RDD: Effect of crossing 50% discount threshold\\n"
             f"Jump = {main['effect_at_cutoff']:+.3f} ({'' if main['p_value']>=0.05 else 'p<0.05'})")
ax.legend()
plt.tight_layout()
plt.savefig(RESULTS / "fig_rdd_main.png", dpi=150)
plt.show()
"""),

    md("## 4. Bandwidth sensitivity"),

    code("""\
bw_results = bandwidth_sensitivity(rdd, outcome="log_reviews")

bws   = [r["bandwidth"] for r in bw_results if "error" not in r]
effs  = [r["effect_at_cutoff"] for r in bw_results if "error" not in r]
cis_l = [r["ci_low"]  for r in bw_results if "error" not in r]
cis_h = [r["ci_high"] for r in bw_results if "error" not in r]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(bws, effs, marker="o", color="steelblue", lw=2)
ax.fill_between(bws, cis_l, cis_h, alpha=0.2, color="steelblue")
ax.axhline(0, color="black", ls="--", lw=1)
ax.set_xlabel("Bandwidth (%)")
ax.set_ylabel("Effect at cutoff (log reviews)")
ax.set_title("Bandwidth sensitivity — RDD estimate stability")
plt.tight_layout()
plt.savefig(RESULTS / "fig_rdd_bandwidth.png", dpi=150)
plt.show()

for r in bw_results:
    if "error" not in r:
        sig = "*" if r["p_value"] < 0.05 else " "
        print(f"  BW={r['bandwidth']:4.0f}%  effect={r['effect_at_cutoff']:+.4f}  "
              f"SE={r['se']:.4f}  p={r['p_value']:.3f} {sig}")
"""),

    md("## 5. McCrary density test (manipulation check)"),

    code("""\
density = mccrary_density_test(rdd)
print("=== McCrary density test ===")
print(f"  N below cutoff: {density['n_below']}  N above: {density['n_above']}")
print(f"  Density below: {density['density_below']:.2f}  above: {density['density_above']:.2f}")
print(f"  Density ratio (above/below): {density['density_ratio']:.2f}")
concern = "YES — bunching may invalidate RDD" if density["manipulation_concern"] else "No concern"
print(f"  Manipulation concern: {concern}")

# Histogram
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(rdd["discount_pct"], bins=range(0, 101, 5), color="steelblue", edgecolor="white")
ax.axvline(50, color="crimson", ls="--", lw=2, label="Cutoff (50%)")
ax.set_xlabel("Discount %")
ax.set_ylabel("Games")
ax.set_title("Discount density — McCrary manipulation test")
ax.legend()
plt.tight_layout()
plt.savefig(RESULTS / "fig_rdd_mccrary.png", dpi=150)
plt.show()
"""),

    md("## 6. Placebo cutoffs"),

    code("""\
placebos = placebo_cutoffs(rdd, outcome="log_reviews")
print("=== Placebo cutoffs (should show null effects) ===")
for r in placebos:
    if "error" not in r:
        sig = "* SIGNIFICANT" if r["p_value"] < 0.05 else ""
        print(f"  Cutoff={r.get('cutoff', '?'):3.0f}%  BW=15  "
              f"effect={r['effect_at_cutoff']:+.4f}  p={r['p_value']:.3f} {sig}")

# Patch cutoff into results for display
placebo_results_with_c = []
for c, r in zip([20,30,40,60,70,80], placebos):
    if "error" not in r:
        placebo_results_with_c.append({**r, "cutoff": c})

fig, ax = plt.subplots(figsize=(9, 4))
real_cutoffs  = [main["effect_at_cutoff"]]
real_labels   = ["50% (real)"]
placebo_effs  = [r["effect_at_cutoff"] for r in placebo_results_with_c]
placebo_labels = [f"{r['cutoff']}%" for r in placebo_results_with_c]
all_labels = placebo_labels[:3] + real_labels + placebo_labels[3:]
all_effs   = placebo_effs[:3]   + real_cutoffs + placebo_effs[3:]
colors = ["steelblue"] * 3 + ["crimson"] + ["steelblue"] * 3
ax.bar(all_labels, all_effs, color=colors)
ax.axhline(0, color="black", lw=1)
ax.set_ylabel("Effect at cutoff (log reviews)")
ax.set_title("Placebo cutoff test — effect should be near zero at fake cutoffs")
plt.tight_layout()
plt.savefig(RESULTS / "fig_rdd_placebo.png", dpi=150)
plt.show()
"""),

    md("## 7. Bootstrap CI + save results"),

    code("""\
boot = bootstrap_rdd(rdd, outcome="log_reviews", bandwidth=15.0, n_bootstrap=500)
print(f"Bootstrap 95% CI: [{boot['bootstrap_ci_low']:+.4f}, {boot['bootstrap_ci_high']:+.4f}]")
print(f"Bootstrap SE: {boot['bootstrap_se']:.4f}")

h2_results = {
    "effect_at_cutoff": main["effect_at_cutoff"],
    "se": main["se"],
    "ci_low": main["ci_low"],
    "ci_high": main["ci_high"],
    "p_value": main["p_value"],
    "bootstrap_ci_low": boot["bootstrap_ci_low"],
    "bootstrap_ci_high": boot["bootstrap_ci_high"],
    "manipulation_concern": density["manipulation_concern"],
    "n_obs": main["n_obs"],
}
import json
with open(RESULTS / "h2_rdd_results.json", "w") as f:
    json.dump(h2_results, f, indent=2)
print(json.dumps(h2_results, indent=2))
"""),
])

# ─────────────────────────────────────────────────────────────────────────────
# Notebook 04: H3 — Synthetic Control
# ─────────────────────────────────────────────────────────────────────────────
nb04 = nb([
    md("# H3 — Synthetic Control: Indie vs AAA Heterogeneity\n\n"
       "**Hypothesis:** The marginal review lift from Winter 2024 sale participation "
       "is larger for indie games than AAA games.\n\n"
       "**Method:** Synthetic control with placebo-based inference. "
       "5 indie + 5 AAA treated games, donor pool = same-genre non-treated games."),

    code("""\
import os, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import psycopg

sys.path.insert(0, str(Path("..").resolve()))
from src.analysis.synthetic_control import fit_synthetic_control, placebo_test, compute_pvalue

load_dotenv()
RESULTS = Path("../results")
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
conn = psycopg.connect(os.environ["DATABASE_URL"])
print("Connected.")
"""),

    md("## 1. Build daily review panel"),

    code("""\
panel = pd.read_sql(\"\"\"
    SELECT r.appid, r.review_date::TEXT AS review_date, r.review_count
    FROM mart.fct_reviews_daily r
    JOIN stg.games g USING (appid)
    ORDER BY r.review_date, r.appid
\"\"\", conn)

panel["log_reviews"] = np.log1p(panel["review_count"])
print(f"Panel rows: {len(panel):,}")
print(f"Date range: {panel['review_date'].min()} – {panel['review_date'].max()}")
print(f"Unique games: {panel['appid'].nunique():,}")
"""),

    md("## 2. Select treated and donor units"),

    code("""\
tc = pd.read_parquet(RESULTS / "eda_treatment_control.parquet")
games = pd.read_sql(\"\"\"
    SELECT appid, name, primary_genre, is_indie, is_aaa, owners_mid
    FROM stg.games WHERE base_price_cents >= 500
\"\"\", conn)
tc = tc.merge(games, on="appid", how="left")

# Select treated games with most reviews during the sale (best signal)
treated = tc[tc["treated"]].copy()
treated_indie = (treated[treated["is_indie"] == True]
                 .sort_values("sale_reviews", ascending=False)
                 .head(5)["appid"].tolist())
treated_aaa   = (treated[treated["is_aaa"] == True]
                 .sort_values("sale_reviews", ascending=False)
                 .head(5)["appid"].tolist())

print(f"Top 5 indie treated: {treated_indie}")
print(f"Top 5 AAA treated:   {treated_aaa}")

# Donor pool: non-treated games with data in our panel
non_treated_appids = tc[~tc["treated"]]["appid"].tolist()
donor_pool = [a for a in non_treated_appids if a in panel["appid"].unique()]
print(f"Donor pool size: {len(donor_pool):,}")
"""),

    md("## 3. Fit synthetic controls"),

    code("""\
SC_KWARGS = dict(
    date_col="review_date",
    outcome_col="log_reviews",
    pre_end="2024-12-18",
    post_start="2024-12-19",
    post_end="2025-01-02",
)

indie_results, aaa_results = [], []
for appid in treated_indie:
    name = games[games["appid"] == appid]["name"].values[0] if appid in games["appid"].values else str(appid)
    r = fit_synthetic_control(panel, appid, donor_pool, **SC_KWARGS)
    if "error" in r:
        print(f"  {name}: {r['error']}")
    else:
        print(f"  {name}: pre_RMSPE={r['pre_rmspe']:.4f}  mean_gap={r['mean_gap_post']:+.4f}")
        indie_results.append({"name": name, **r})

print()
for appid in treated_aaa:
    name = games[games["appid"] == appid]["name"].values[0] if appid in games["appid"].values else str(appid)
    r = fit_synthetic_control(panel, appid, donor_pool, **SC_KWARGS)
    if "error" in r:
        print(f"  {name}: {r['error']}")
    else:
        print(f"  {name}: pre_RMSPE={r['pre_rmspe']:.4f}  mean_gap={r['mean_gap_post']:+.4f}")
        aaa_results.append({"name": name, **r})
"""),

    md("## 4. Gap plots"),

    code("""\
def plot_gaps(results, title, color, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
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
    return ax

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
plot_gaps(indie_results, "Indie treated games", "steelblue", axes[0])
plot_gaps(aaa_results,   "AAA treated games",   "coral",     axes[1])
plt.suptitle("Synthetic control gap plots (positive = sale lift above synthetic)", y=1.02)
plt.tight_layout()
plt.savefig(RESULTS / "fig_sc_gap_plots.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

    md("## 5. Placebo test"),

    code("""\
all_treated = treated_indie + treated_aaa
print("Running placebo test (n=50)...")
placebos = placebo_test(panel, all_treated, donor_pool, n_placebos=50, seed=42, **SC_KWARGS)
print(f"Placebo tests completed: {placebos['n_placebos']}")

placebo_gaps = [r["mean_gap"] for r in placebos["placebo_results"]]

# Compare treated mean gaps to placebo distribution
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
plt.show()

# p-values
if not np.isnan(indie_mean_gap):
    p_indie = np.mean(np.array(placebo_gaps) >= indie_mean_gap)
    print(f"Indie mean gap: {indie_mean_gap:+.4f}  one-sided p = {p_indie:.3f}")
if not np.isnan(aaa_mean_gap):
    p_aaa = np.mean(np.array(placebo_gaps) >= aaa_mean_gap)
    print(f"AAA mean gap:   {aaa_mean_gap:+.4f}  one-sided p = {p_aaa:.3f}")
"""),

    md("## 6. Heterogeneity table + save results"),

    code("""\
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
    print(het_df.groupby("tier")[["pre_rmspe","mean_gap_post"]].agg(["mean","std"]).round(4))

h3_results = {
    "indie_mean_gap": float(indie_mean_gap) if not np.isnan(indie_mean_gap) else None,
    "aaa_mean_gap":   float(aaa_mean_gap)   if not np.isnan(aaa_mean_gap)   else None,
    "n_indie_treated": len(indie_results),
    "n_aaa_treated":   len(aaa_results),
    "n_placebos": placebos["n_placebos"],
    "heterogeneity": het_df.to_dict(orient="records") if not het_df.empty else [],
}

import json
with open(RESULTS / "h3_sc_results.json", "w") as f:
    json.dump(h3_results, f, indent=2)
print(json.dumps({k: v for k, v in h3_results.items() if k != "heterogeneity"}, indent=2))
conn.close()
"""),
])

# ── Write notebooks ────────────────────────────────────────────────────────────
for fname, content in [
    ("02_hypothesis_1_did.ipynb", nb02),
    ("03_hypothesis_2_rdd.ipynb", nb03),
    ("04_hypothesis_3_synthetic_control.ipynb", nb04),
]:
    path = NOTEBOOKS_DIR / fname
    path.write_text(json.dumps(content, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Written: {path}")
