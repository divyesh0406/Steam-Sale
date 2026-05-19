"""Rewrite EDA notebook to target Winter 2024 sale instead of Summer 2024."""
import json
from pathlib import Path

nb_path = Path("notebooks/01_eda.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))


def set_code_cell(nb, idx, lines):
    cell = nb["cells"][idx]
    cell["source"] = lines
    cell["outputs"] = []
    cell["execution_count"] = None


def set_md_cell(nb, idx, lines):
    cell = nb["cells"][idx]
    cell["source"] = lines


# ── Cell 0: header ────────────────────────────────────────────────────────────
set_md_cell(nb, 0, [
    "# Steam Sale Lift — EDA & Hypothesis Generation\n",
    "\n",
    "**Goal:** Understand the data, identify data quality issues, and generate 3 sharp falsifiable hypotheses for the causal analyses.\n",
    "\n",
    "**Outputs:**\n",
    "- `results/eda_treatment_control.parquet` — Winter 2024 treatment/control assignment\n",
    "- `results/eda_rdd_sample.parquet` — discount-depth sample for RDD\n",
    "- `results/eda_summary_stats.json` — headline numbers for the README\n",
    "\n",
    "**Target sale event:** Steam Winter Sale 2024 (Dec 19 2024 – Jan 2 2025), chosen because\n",
    "reviews were scraped with `day_range=500` which fully covers this window and its pre/post buffers.\n",
])

# ── Cell 18: section 9 markdown ───────────────────────────────────────────────
set_md_cell(nb, 18, [
    "## 9. Define treatment and control groups for H1\n",
    "\n",
    "**Treatment definition:** Games that participated in the Steam Winter Sale 2024 (Dec 19 2024 – Jan 2 2025).\n",
    "\n",
    "**Proxy:** We use review velocity spike during the sale window. Games with elevated review counts\n",
    "during Winter 2024 relative to their 30-day pre-period baseline are classified as treated.\n",
    "\n",
    "**Note:** Review data was scraped with `day_range=500` (last ~500 days from May 2026), which covers\n",
    "the Winter 2024 sale window and its 30-day pre/post buffers. This is the most recent major sale\n",
    "with full review coverage in our dataset.\n",
])

# ── Cell 19: treatment window code ────────────────────────────────────────────
set_code_cell(nb, 19, [
    "# Build Winter 2024 treatment/control assignment\n",
    "# Treatment: paid game (base_price >= $5) with review activity during Winter 2024 sale window\n",
    "# Control:   paid game with no elevated review activity during that window\n",
    "\n",
    'WINTER_2024_START = pd.Timestamp("2024-12-19")\n',
    'WINTER_2024_END   = pd.Timestamp("2025-01-02")\n',
    "PRE_START  = WINTER_2024_START - pd.Timedelta(days=30)\n",
    "PRE_END    = WINTER_2024_START - pd.Timedelta(days=1)\n",
    "POST_START = WINTER_2024_END   + pd.Timedelta(days=1)\n",
    "POST_END   = WINTER_2024_END   + pd.Timedelta(days=30)\n",
    "\n",
    "# Load review windows for all paid games\n",
    "review_windows = pd.read_sql(\"\"\"\n",
    "    SELECT\n",
    "        r.appid,\n",
    "        SUM(CASE WHEN r.review_date BETWEEN %(pre_start)s AND %(pre_end)s\n",
    "                 THEN r.review_count ELSE 0 END)  AS pre_reviews,\n",
    "        SUM(CASE WHEN r.review_date BETWEEN %(post_start)s AND %(post_end)s\n",
    "                 THEN r.review_count ELSE 0 END)  AS post_reviews,\n",
    "        SUM(CASE WHEN r.review_date BETWEEN %(sale_start)s AND %(sale_end)s\n",
    "                 THEN r.review_count ELSE 0 END)  AS sale_reviews\n",
    "    FROM mart.fct_reviews_daily r\n",
    "    JOIN stg.games g USING (appid)\n",
    "    WHERE g.base_price_cents >= 500\n",
    "    GROUP BY r.appid\n",
    "\"\"\", conn, params={\n",
    '    "pre_start":  PRE_START.date(),\n',
    '    "pre_end":    PRE_END.date(),\n',
    '    "post_start": POST_START.date(),\n',
    '    "post_end":   POST_END.date(),\n',
    '    "sale_start": WINTER_2024_START.date(),\n',
    '    "sale_end":   WINTER_2024_END.date(),\n',
    "})\n",
    "\n",
    'print(f"Paid games with review data: {len(review_windows):,}")\n',
    'print(review_windows[["pre_reviews","sale_reviews","post_reviews"]].describe().round(1))\n',
])

# ── Cell 20: treatment assignment ─────────────────────────────────────────────
set_code_cell(nb, 20, [
    "# Treatment: games with sale_reviews > 0 AND elevated vs pre-period baseline\n",
    'review_windows["review_ratio"] = (\n',
    '    review_windows["sale_reviews"] /\n',
    '    (review_windows["pre_reviews"] / 30 * 15).clip(lower=1)\n',
    ")\n",
    'review_windows["treated"] = (\n',
    '    (review_windows["sale_reviews"] > 0) &\n',
    '    (review_windows["review_ratio"] >= 0.5)\n',
    ").astype(int)\n",
    "\n",
    'n_treated = review_windows["treated"].sum()\n',
    'n_control = (review_windows["treated"] == 0).sum()\n',
    'print(f"Treated (active during Winter 2024): {n_treated:,}")\n',
    'print(f"Control (paid, no sale activity):    {n_control:,}")\n',
    "if n_treated > 0:\n",
    '    print(f"Ratio: {n_control/n_treated:.1f}:1 control:treated")\n',
    "else:\n",
    '    print("WARNING: 0 treated games — review data may not cover this window yet")\n',
])

# ── Cell 24: parallel trends code ─────────────────────────────────────────────
set_code_cell(nb, 24, [
    "# Load monthly review velocity for treated vs control, 6 months before Winter 2024\n",
    'treated_appids = tc[tc["treated"] == 1]["appid"].tolist()\n',
    'control_appids = tc[tc["treated"] == 0]["appid"].tolist()\n',
    "\n",
    "import random\n",
    "random.seed(42)\n",
    "t_sample = random.sample(treated_appids, min(500, len(treated_appids)))\n",
    "c_sample = random.sample(control_appids, min(500, len(control_appids)))\n",
    "\n",
    "def get_monthly_reviews(appids, label):\n",
    "    if not appids:\n",
    '        return pd.DataFrame(columns=["month", "avg_daily_reviews", "group"])\n',
    "    df = pd.read_sql(\"\"\"\n",
    "        SELECT\n",
    "            DATE_TRUNC('month', review_date)::DATE AS month,\n",
    "            AVG(review_count)                      AS avg_daily_reviews\n",
    "        FROM mart.fct_reviews_daily\n",
    "        WHERE appid = ANY(%(ids)s)\n",
    "          AND review_date BETWEEN '2024-06-01' AND '2025-02-28'\n",
    "        GROUP BY 1\n",
    "        ORDER BY 1\n",
    '    \"\"\", conn, params={"ids": appids}, parse_dates=["month"])\n',
    '    df["group"] = label\n',
    "    return df\n",
    "\n",
    'trend_t = get_monthly_reviews(t_sample, "Treated")\n',
    'trend_c = get_monthly_reviews(c_sample, "Control")\n',
    "trends = pd.concat([trend_t, trend_c])\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(11, 4))\n",
    'for grp, color in [("Treated", "steelblue"), ("Control", "coral")]:\n',
    '    d = trends[trends["group"] == grp]\n',
    "    if not d.empty:\n",
    '        ax.plot(d["month"], d["avg_daily_reviews"], color=color, lw=2, label=grp, marker="o", ms=4)\n',
    "\n",
    'ax.axvline(pd.Timestamp("2024-12-19"), color="black", linestyle="--", label="Sale start")\n',
    'ax.axvline(pd.Timestamp("2025-01-02"), color="black", linestyle=":",  label="Sale end")\n',
    'ax.set_title("Parallel trends check — avg daily reviews per game (sample of 500 each)")\n',
    'ax.set_ylabel("Avg daily reviews per game")\n',
    "ax.legend()\n",
    "plt.tight_layout()\n",
    'plt.savefig(RESULTS / "fig_parallel_trends.png", dpi=150)\n',
    "plt.show()\n",
    'print("If treated and control move together before Dec 2024, DiD is credible.")\n',
])

# ── Cell 26: RDD query — update dates ─────────────────────────────────────────
rdd_src = nb["cells"][26]["source"]
new_rdd = []
for line in rdd_src:
    line = line.replace("'2024-06-27' AND '2024-07-25'", "'2024-12-19' AND '2025-01-16'")
    line = line.replace("'2024-05-28' AND '2024-06-26'", "'2024-11-19' AND '2024-12-18'")
    line = line.replace("most recent major sale start", "Winter 2024 sale start")
    new_rdd.append(line)
nb["cells"][26]["source"] = new_rdd
nb["cells"][26]["outputs"] = []
nb["cells"][26]["execution_count"] = None

# ── Cell 28: summary stats key rename ─────────────────────────────────────────
set_code_cell(nb, 28, [
    "summary = {\n",
    '    "total_games": len(games),\n',
    '    "indie_pct": round(float(games["is_indie"].mean()), 3),\n',
    '    "aaa_pct": round(float(games["is_aaa"].mean()), 3),\n',
    '    "median_base_price_usd": round(float(games["base_price_usd"].median()), 2),\n',
    '    "games_on_sale_snapshot": int(gs["discount"].gt(0).sum()),\n',
    '    "avg_discount_on_sale": round(float(gs[gs["discount"] > 0]["discount"].mean()), 1),\n',
    '    "treated_winter_2024": int(n_treated),\n',
    '    "control_winter_2024": int(n_control),\n',
    '    "rdd_sample_size": len(rdd_df),\n',
    "}\n",
    "\n",
    'with open(RESULTS / "eda_summary_stats.json", "w") as f:\n',
    "    json.dump(summary, f, indent=2)\n",
    "\n",
    "print(json.dumps(summary, indent=2))\n",
])

# ── Cell 29: hypotheses markdown ──────────────────────────────────────────────
hyp_src = "".join(nb["cells"][29]["source"])
hyp_src = hyp_src.replace("Summer 2024", "Winter 2024")
hyp_src = hyp_src.replace("summer_2024", "winter_2024")
hyp_src = hyp_src.replace("Summer Sale 2024", "Winter Sale 2024")
hyp_src = hyp_src.replace("June 27", "December 19")
hyp_src = hyp_src.replace("June 2024", "December 2024")
hyp_src = hyp_src.replace("July 11", "January 2")
nb["cells"][29]["source"] = [hyp_src]

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Done. Notebook updated with Winter 2024 dates. Total cells: {len(nb['cells'])}")
