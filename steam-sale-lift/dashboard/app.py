"""
Steam Sale Lift — Phase 5 Streamlit Dashboard
Causal analysis of Winter 2024 Steam sale effect on game reviews.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

RESULTS = Path(__file__).parent.parent / "results"

st.set_page_config(
    page_title="Steam Sale Lift",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── helpers ──────────────────────────────────────────────────────────────────

@st.cache_data
def load_json(name):
    return json.loads((RESULTS / name).read_text())

@st.cache_data
def load_parquet(name):
    return pd.read_parquet(RESULTS / name)

def sig_badge(p, thresholds=((0.01, "🟢 p<0.01"), (0.05, "🟡 p<0.05"), (0.10, "🟠 p<0.10"))):
    for thresh, label in thresholds:
        if p < thresh:
            return label
    return "🔴 n.s."

def show_image(fname, caption=None, use_column_width=True):
    path = RESULTS / fname
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=use_column_width)
    else:
        st.warning(f"Figure not found: {fname}")

# ── header ────────────────────────────────────────────────────────────────────

st.title("🎮 Steam Winter 2024 Sale — Causal Impact on Reviews")
st.markdown(
    "**Does participating in Steam's Winter 2024 sale (Dec 19 2024 – Jan 2 2025) "
    "increase the number of reviews a game receives?**  \n"
    "Three independent causal methods, one dataset: 2,185 games, 205,163 daily review observations."
)

# ── top KPI row ───────────────────────────────────────────────────────────────

h1 = load_json("h1_did_results.json")
h2 = load_json("h2_rdd_results.json")
h3 = load_json("h3_sc_results.json")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Games analysed", "2,185", "232 treated / 1,953 control")
col2.metric("DiD ATT (CUPED)", f"{h1['cuped_att']:+.3f} log reviews", sig_badge(h1['permutation_p']))
col3.metric("RDD jump at 50%", f"{h2['effect_at_cutoff']:+.1f} log reviews", sig_badge(h2['p_value']))
col4.metric("SC lift — Indie", f"{h3['indie_mean_gap']:+.2f} log reviews", sig_badge(h3['p_indie']))
col5.metric("SC lift — AAA", f"{h3['aaa_mean_gap']:+.2f} log reviews", sig_badge(h3['p_aaa']))

st.divider()

# ── tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "H1 — Diff-in-Diff",
    "H2 — Regression Discontinuity",
    "H3 — Synthetic Control",
    "Data & Methods",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — DiD
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("H1 — Did sale participation increase reviews? (DiD + CUPED)")
    st.markdown(
        "**Design:** 2×2 Difference-in-Differences comparing treated games (SteamSpy discount > 0) "
        "to untreated games across pre-sale (Nov 19 – Dec 18) and sale windows (Dec 19 – Jan 2).  \n"
        "CUPED (Deng et al. 2013) adjusts for pre-period variance, reducing noise by "
        f"**{h1['variance_reduction']*100:.0f}%**."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Naive ATT", f"{h1['naive_att']:+.4f}", "log reviews")
    c2.metric("CUPED ATT", f"{h1['cuped_att']:+.4f}", "log reviews")
    c3.metric("95% CI", f"[{h1['cuped_ci_low']:+.3f}, {h1['cuped_ci_high']:+.3f}]")
    c4.metric("Permutation p", f"{h1['permutation_p']:.3f}", sig_badge(h1['permutation_p']))

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        show_image("fig_did_parallel_trends.png", "Parallel trends check")
    with col_b:
        show_image("fig_did_permutation.png", "Permutation null distribution")

    st.markdown("---")
    st.subheader("Interpretation")
    att = h1['cuped_att']
    pct_change = (np.exp(abs(att)) - 1) * 100
    direction = "fewer" if att < 0 else "more"
    st.info(
        f"**Finding:** The CUPED-adjusted ATT is **{att:+.4f}** log reviews "
        f"({pct_change:.1f}% {direction} reviews). "
        f"The 95% CI is [{h1['cuped_ci_low']:+.3f}, {h1['cuped_ci_high']:+.3f}] and "
        f"the permutation p-value is **{h1['permutation_p']:.3f}** — not significant at 5%.  \n\n"
        f"The CUPED adjustment reduced variance by {h1['variance_reduction']*100:.0f}%, "
        f"tightening the SE from {h1['naive_se']:.4f} to {h1['cuped_se']:.4f}. "
        f"Despite the efficiency gain, the effect is indistinguishable from zero, "
        f"suggesting that **participating in the sale does not generate a detectable average "
        f"review lift across the full game population**."
    )
    st.caption(
        f"n_treated={h1['n_treated']:,}  n_control={h1['n_control']:,}  "
        f"variance_reduction={h1['variance_reduction']:.2%}"
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — RDD
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("H2 — Does crossing the 50% discount threshold boost reviews? (RDD)")
    st.markdown(
        "**Design:** Sharp RDD on discount depth. Running variable: SteamSpy discount %. "
        "Cutoff: 50% — Steam's alleged front-page featured-placement visibility threshold. "
        "Outcome: log review count during the 14-day sale window. Bandwidth: ±15 pp."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jump at cutoff", f"{h2['effect_at_cutoff']:+.3f}", "log reviews")
    c2.metric("SE", f"{h2['se']:.3f}")
    c3.metric("95% CI", f"[{h2['ci_low']:+.2f}, {h2['ci_high']:+.2f}]")
    c4.metric("p-value", f"{h2['p_value']:.3f}", sig_badge(h2['p_value']))

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        show_image("fig_rdd_main.png", "RDD plot — fitted lines either side of cutoff")
    with col_b:
        show_image("fig_rdd_bandwidth.png", "Bandwidth sensitivity")

    col_c, col_d = st.columns(2)
    with col_c:
        show_image("fig_rdd_mccrary.png", "McCrary density test")
    with col_d:
        show_image("fig_rdd_placebo.png", "Placebo cutoffs")

    st.markdown("---")
    st.subheader("Interpretation")

    concern_text = (
        "⚠️ **Manipulation concern flagged** — the density ratio above/below the cutoff "
        "exceeds 1.5x, suggesting developers may be gaming the 50% threshold. "
        "This would invalidate the RDD assumption of no precise control over the running variable.  \n\n"
        if h2["manipulation_concern"] else ""
    )
    st.warning(
        f"**Finding:** The estimated jump at the 50% cutoff is **{h2['effect_at_cutoff']:+.3f}** "
        f"log reviews (SE={h2['se']:.3f}, p={h2['p_value']:.3f}) — **not significant**.  \n\n"
        f"{concern_text}"
        f"Only **{h2['n_obs']}** games fall within the ±15pp bandwidth, limiting power. "
        f"The bootstrap CI [{h2['bootstrap_ci_low']:+.2f}, {h2['bootstrap_ci_high']:+.2f}] is wide. "
        f"No evidence that crossing 50% discount generates additional reviews above what the "
        f"discount level itself would predict."
    )
    st.caption(f"n_obs={h2['n_obs']}  bandwidth=±15pp  local linear regression  HC1 robust SE")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Synthetic Control
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("H3 — Is the sale lift larger for indie games than AAA? (Synthetic Control)")
    st.markdown(
        "**Design:** Synthetic control (Abadie et al.) for top 5 indie and top 5 AAA treated games. "
        "Donor pool: 200 randomly-sampled non-treated games. Weights chosen by SLSQP to minimise "
        "pre-period MSPE. Placebo-based inference: p-value = fraction of 50 placebo units with "
        "post/pre RMSPE ratio ≥ treated."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Indie mean gap", f"{h3['indie_mean_gap']:+.3f}", "log reviews")
    c2.metric("Indie p-value", f"{h3['p_indie']:.3f}", sig_badge(h3['p_indie']))
    c3.metric("AAA mean gap", f"{h3['aaa_mean_gap']:+.3f}", "log reviews")
    c4.metric("AAA p-value", f"{h3['p_aaa']:.3f}", sig_badge(h3['p_aaa']))

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        show_image("fig_sc_gap_plots.png", "Gap plots: actual − synthetic")
    with col_b:
        show_image("fig_sc_placebo.png", "Placebo distribution vs treated gaps")

    st.markdown("---")
    st.subheader("Per-game breakdown")

    het = pd.DataFrame(h3["heterogeneity"])
    if not het.empty:
        het["mean_gap_post"] = het["mean_gap_post"].round(4)
        het["pre_rmspe"] = het["pre_rmspe"].round(4)
        het["fit_quality"] = het["pre_rmspe"].apply(
            lambda x: "Good" if x < 0.05 else ("Fair" if x < 0.2 else "Poor")
        )
        st.dataframe(
            het[["tier","name","mean_gap_post","pre_rmspe","fit_quality"]].rename(columns={
                "tier": "Tier", "name": "Game",
                "mean_gap_post": "Mean gap (log reviews)",
                "pre_rmspe": "Pre-RMSPE", "fit_quality": "Fit quality"
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.subheader("Interpretation")
    st.success(
        f"**Finding:** Indie games show a mean gap of **{h3['indie_mean_gap']:+.3f}** log reviews "
        f"(placebo p = **{h3['p_indie']:.3f}**) — highly significant.  \n"
        f"AAA games show **{h3['aaa_mean_gap']:+.3f}** log reviews (p = {h3['p_aaa']:.3f}) — no effect.  \n\n"
        f"**The heterogeneity finding is strong: indie games benefit from the Winter sale in terms "
        f"of review volume; AAA games do not.**  \n\n"
        f"⚠️ *Caveat:* MiSide (gap = +4.37) is a high-leverage outlier with poor pre-period fit "
        f"(pre-RMSPE = 0.68). Excluding it, the remaining 4 indie games show modest gaps "
        f"(+0.05 to +0.06), similar to AAA. The aggregate indie result is driven by one viral title."
    )
    st.caption(f"n_indie={h3['n_indie_treated']}  n_aaa={h3['n_aaa_treated']}  n_placebos={h3['n_placebos']}  donor_pool=200")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Data & Methods
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("Data & Methods")

    st.subheader("Data sources")
    st.markdown("""
| Source | What | Rows |
|---|---|---|
| SteamSpy API | Game metadata, discount snapshot, owner estimates | 2,996 games |
| Synthetic reviews | Daily review counts generated from SteamSpy lifetime totals + exponential decay + sale multipliers | 205,163 rows (76-day window) |
| Steam Charts | Monthly player counts | ~278K rows |

**Target event:** Steam Winter 2024 Sale — Dec 19 2024 to Jan 2 2025 (15 days).
**Analysis window:** Nov 19 2024 – Feb 2 2025 (76 days).
**Treatment proxy:** SteamSpy `discount > 0` snapshot → 232 treated (10.6%), 1,953 control.
    """)

    st.subheader("Methods")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
**H1 — DiD + CUPED**
- 2×2 DiD: (post−pre)_treated − (post−pre)_control
- CUPED adjustment uses pre-period reviews as covariate
- Permutation-based inference (1,000 shuffles)
- Variance reduction: 94%
        """)
    with col2:
        st.markdown("""
**H2 — RDD**
- Running variable: discount %
- Cutoff: 50% (visibility threshold)
- Local linear regression, HC1 robust SE
- Bandwidth sensitivity: 5–30pp
- McCrary density test for manipulation
        """)
    with col3:
        st.markdown("""
**H3 — Synthetic Control**
- Abadie et al. convex weights (SLSQP)
- Minimise pre-period MSPE
- Donor pool: 200 non-treated games
- Placebo inference: 50 fake treated units
- Pre-RMSPE as fit quality metric
        """)

    st.subheader("Limitations")
    st.markdown("""
- **Synthetic reviews:** All review data is generated from SteamSpy lifetime totals with exponential decay and sale boost multipliers — not real Steam review timestamps. This means causal estimates measure the signal we built into the data-generating process, not true user behaviour.
- **Treatment proxy:** `discount > 0` is a current snapshot, not a verified Winter 2024 participation record. Some discounts may be unrelated to the sale.
- **Small RDD sample:** Only 58 games fall within the ±15pp bandwidth around the 50% cutoff, limiting statistical power for H2.
- **MiSide outlier:** The indie SC result is heavily influenced by one viral game with poor synthetic fit.
    """)

    st.subheader("Reproducibility")
    st.code("""
# Clone and run
git clone <repo>
cd steam-sale-lift
uv sync
uv run python scripts/run_sc_analysis.py   # recompute SC results
uv run streamlit run dashboard/app.py       # launch dashboard
    """, language="bash")
