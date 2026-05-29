"""Rebuild notebook 04 to load pre-computed results instead of rerunning SC."""
import json
from pathlib import Path

nb_path = Path(__file__).parent.parent / "notebooks" / "04_hypothesis_3_synthetic_control.ipynb"

cells = [
    {
        "cell_type": "markdown",
        "id": "md000f7093",
        "metadata": {},
        "source": "# H3 — Synthetic Control: Indie vs AAA Heterogeneity\n\n**Hypothesis:** The marginal review lift from Winter 2024 sale participation is larger for indie games than AAA games.\n\n**Method:** Synthetic control (scipy SLSQP convex weights) with placebo-based inference. 5 indie + 5 AAA treated games, donor pool = 200 randomly-sampled non-treated games."
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "cd_setup",
        "metadata": {},
        "outputs": [],
        "source": "import json\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport matplotlib.image as mpimg\n\nRESULTS = Path('../results')\nplt.rcParams.update({'figure.dpi': 120, 'axes.spines.top': False, 'axes.spines.right': False})\nprint('Ready.')"
    },
    {
        "cell_type": "markdown",
        "id": "md_results",
        "metadata": {},
        "source": "## 1. Load pre-computed results\n\nResults were computed by `scripts/run_sc_analysis.py` (60 SLSQP fits: 5 indie + 5 AAA treated + 50 placebos)."
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "cd_load",
        "metadata": {},
        "outputs": [],
        "source": (
            "with open(RESULTS / 'h3_sc_results.json') as f:\n"
            "    h3 = json.load(f)\n"
            "\n"
            "print('=== Synthetic Control Results ===')\n"
            "print(f\"Indie mean gap (log reviews): {h3['indie_mean_gap']:+.4f}  p = {h3['p_indie']:.3f}\")\n"
            "print(f\"AAA   mean gap (log reviews): {h3['aaa_mean_gap']:+.4f}  p = {h3['p_aaa']:.3f}\")\n"
            "print(f\"Placebos used: {h3['n_placebos']}\")\n"
            "print()\n"
            "\n"
            "het = pd.DataFrame(h3['heterogeneity'])\n"
            "if not het.empty:\n"
            "    print(het.groupby('tier')[['pre_rmspe','mean_gap_post']].agg(['mean','std']).round(4))\n"
            "    print()\n"
            "    print(het[['tier','name','pre_rmspe','mean_gap_post']].to_string(index=False))"
        )
    },
    {
        "cell_type": "markdown",
        "id": "md_gaps",
        "metadata": {},
        "source": "## 2. Gap plots"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "cd_gaps",
        "metadata": {},
        "outputs": [],
        "source": (
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
            "for ax, fname, title in [\n"
            "    (axes[0], 'fig_sc_gap_plots.png', 'Gap plots'),\n"
            "]:\n"
            "    pass\n"
            "\n"
            "img = mpimg.imread(RESULTS / 'fig_sc_gap_plots.png')\n"
            "plt.figure(figsize=(14, 5))\n"
            "plt.imshow(img)\n"
            "plt.axis('off')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        )
    },
    {
        "cell_type": "markdown",
        "id": "md_placebo",
        "metadata": {},
        "source": "## 3. Placebo distribution"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "cd_placebo",
        "metadata": {},
        "outputs": [],
        "source": (
            "img2 = mpimg.imread(RESULTS / 'fig_sc_placebo.png')\n"
            "plt.figure(figsize=(9, 4))\n"
            "plt.imshow(img2)\n"
            "plt.axis('off')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        )
    },
    {
        "cell_type": "markdown",
        "id": "md_interp",
        "metadata": {},
        "source": (
            "## 4. Interpretation\n\n"
            "| | Indie (n=5) | AAA (n=5) |\n"
            "|---|---|---|\n"
            "| Mean gap (log reviews) | **+0.91** | +0.05 |\n"
            "| One-sided p-value | **0.000** | 0.500 |\n\n"
            "**Finding:** Indie games show a large, statistically significant positive gap versus their synthetic controls during the Winter 2024 sale window. AAA games show no detectable lift.\n\n"
            "**Caveat:** MiSide (gap = +4.37) is a high-leverage outlier; the remaining 4 indie games show modest gaps (+0.05 to +0.06). The pre-RMSPE for MiSide is high (0.68), suggesting poor pre-period fit — interpret with caution."
        )
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "cd_summary",
        "metadata": {},
        "outputs": [],
        "source": (
            "print('H3 complete.')\n"
            "print(json.dumps({k: v for k, v in h3.items() if k != 'heterogeneity'}, indent=2))"
        )
    },
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "cells": cells,
}

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {nb_path}")
