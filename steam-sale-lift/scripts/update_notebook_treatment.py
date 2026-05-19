"""Update EDA notebook treatment cells to use SteamSpy discount as treatment proxy."""
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
    nb["cells"][idx]["source"] = lines


# ── Cell 18: section 9 markdown ───────────────────────────────────────────────
set_md_cell(nb, 18, [
    "## 9. Define treatment and control groups for H1\n",
    "\n",
    "**Treatment proxy:** Games currently on sale in the SteamSpy snapshot (May 2026, `discount > 0`).\n",
    "This is a cross-sectional proxy for 'games that go on sale' — games with a non-zero discount\n",
    "are systematically the kind of games that participate in Steam sales.\n",
    "\n",
    "**Control:** Paid games with no current discount (`discount = 0` or NULL).\n",
    "\n",
    "**Outcome:** Review velocity lift during Winter 2024 sale window vs. 30-day pre-period,\n",
    "computed from synthetic review data generated from SteamSpy lifetime totals.\n",
    "\n",
    "**Note on identification:** Because the treatment proxy (current discount) is observed *after*\n",
    "the outcome period (Winter 2024), this is a selection-on-observables design, not a DiD.\n",
    "The parallel trends assumption is replaced by conditional independence: games on sale now\n",
    "are similar in pre-trends to games not on sale, conditional on genre + price.\n",
])

# ── Cell 19: treatment window code ────────────────────────────────────────────
set_code_cell(nb, 19, [
    "# Treatment: games currently discounted in SteamSpy snapshot (proxy for 'sale participant')\n",
    "# Control:   paid games with no current discount\n",
    "\n",
    'WINTER_2024_START = pd.Timestamp("2024-12-19")\n',
    'WINTER_2024_END   = pd.Timestamp("2025-01-02")\n',
    "PRE_START  = WINTER_2024_START - pd.Timedelta(days=30)\n",
    "PRE_END    = WINTER_2024_START - pd.Timedelta(days=1)\n",
    "POST_START = WINTER_2024_END   + pd.Timedelta(days=1)\n",
    "POST_END   = WINTER_2024_END   + pd.Timedelta(days=30)\n",
    "\n",
    "# Load review windows + treatment assignment from SteamSpy discount\n",
    "review_windows = pd.read_sql(\"\"\"\n",
    "    SELECT\n",
    "        g.appid,\n",
    "        COALESCE(sp.discount, 0) > 0               AS treated,\n",
    "        COALESCE(sp.discount, 0)                    AS discount_pct,\n",
    "        SUM(CASE WHEN r.review_date BETWEEN %(pre_start)s AND %(pre_end)s\n",
    "                 THEN r.review_count ELSE 0 END)    AS pre_reviews,\n",
    "        SUM(CASE WHEN r.review_date BETWEEN %(post_start)s AND %(post_end)s\n",
    "                 THEN r.review_count ELSE 0 END)    AS post_reviews,\n",
    "        SUM(CASE WHEN r.review_date BETWEEN %(sale_start)s AND %(sale_end)s\n",
    "                 THEN r.review_count ELSE 0 END)    AS sale_reviews\n",
    "    FROM stg.games g\n",
    "    LEFT JOIN mart.fct_steamspy sp USING (appid)\n",
    "    LEFT JOIN mart.fct_reviews_daily r USING (appid)\n",
    "    WHERE g.base_price_cents >= 500\n",
    "    GROUP BY g.appid, sp.discount\n",
    "\"\"\", conn, params={\n",
    '    "pre_start":  PRE_START.date(),\n',
    '    "pre_end":    PRE_END.date(),\n',
    '    "post_start": POST_START.date(),\n',
    '    "post_end":   POST_END.date(),\n',
    '    "sale_start": WINTER_2024_START.date(),\n',
    '    "sale_end":   WINTER_2024_END.date(),\n',
    "})\n",
    "\n",
    'print(f"Total paid games: {len(review_windows):,}")\n',
    'print(f"Treated (discount > 0):  {review_windows[\'treated\'].sum():,} ({review_windows[\'treated\'].mean():.1%})")\n',
    'print(f"Control (discount = 0):  {(~review_windows[\'treated\']).sum():,}")\n',
    'print(review_windows[[\"pre_reviews\",\"sale_reviews\",\"post_reviews\"]].describe().round(1))\n',
])

# ── Cell 20: treatment assignment (simplified — already in query) ──────────────
set_code_cell(nb, 20, [
    "# Log-transform review counts for DiD outcome variable\n",
    "import numpy as np\n",
    "review_windows['log_pre']  = np.log1p(review_windows['pre_reviews'])\n",
    "review_windows['log_post'] = np.log1p(review_windows['post_reviews'])\n",
    "review_windows['log_lift'] = review_windows['log_post'] - review_windows['log_pre']\n",
    "\n",
    "n_treated = int(review_windows['treated'].sum())\n",
    "n_control = int((~review_windows['treated']).sum())\n",
    'print(f"Treated: {n_treated:,}  Control: {n_control:,}  Ratio: {n_control/n_treated:.1f}:1")\n',
    "\n",
    "print('\\n=== Log review lift by treatment status ===')\n",
    "print(review_windows.groupby('treated')['log_lift'].agg(['mean','std','count']).round(3))\n",
    "\n",
    "print('\\n=== Raw review lift by treatment status ===')\n",
    "review_windows['review_lift'] = review_windows['sale_reviews'] - review_windows['pre_reviews'] / 30 * 15\n",
    "print(review_windows.groupby('treated')['review_lift'].agg(['mean','median','count']).round(1))\n",
])

# ── Cell 28: summary stats ────────────────────────────────────────────────────
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

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Notebook updated with SteamSpy-discount treatment definition.")
