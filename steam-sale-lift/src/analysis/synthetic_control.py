"""
Synthetic Control method for heterogeneity analysis (H3).
Outcome: daily log review count. Donor pool: non-treated games in the same genre.
Uses scipy.optimize to find convex weights minimising pre-period MSPE.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _mspe(weights: np.ndarray, treated_pre: np.ndarray, donor_pre: np.ndarray) -> float:
    synthetic = donor_pre @ weights
    return float(np.mean((treated_pre - synthetic) ** 2))


def fit_synthetic_control(
    panel: pd.DataFrame,
    treated_appid: int,
    donor_appids: list[int],
    date_col: str = "review_date",
    outcome_col: str = "log_reviews",
    pre_end: str = "2024-12-18",
    post_start: str = "2024-12-19",
    post_end: str = "2025-01-02",
) -> dict:
    """
    Fit synthetic control for one treated unit.

    panel: long DataFrame with columns [date_col, appid, outcome_col].
    Returns dict with weights, gap series, pre-RMSPE, post gap mean.
    """
    pivot = panel.pivot(index=date_col, columns="appid", values=outcome_col).fillna(0)

    if treated_appid not in pivot.columns:
        return {"error": f"appid {treated_appid} not in panel"}

    available_donors = [d for d in donor_appids if d in pivot.columns]
    if len(available_donors) < 2:
        return {"error": "fewer than 2 donor units available"}

    pre_mask  = pivot.index <= pre_end
    post_mask = (pivot.index >= post_start) & (pivot.index <= post_end)

    treated_pre  = pivot.loc[pre_mask,  treated_appid].values
    donor_pre    = pivot.loc[pre_mask,  available_donors].values
    treated_post = pivot.loc[post_mask, treated_appid].values
    donor_post   = pivot.loc[post_mask, available_donors].values

    n_donors = len(available_donors)
    w0 = np.ones(n_donors) / n_donors

    result = minimize(
        _mspe,
        w0,
        args=(treated_pre, donor_pre),
        method="SLSQP",
        bounds=[(0, 1)] * n_donors,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        options={"ftol": 1e-9, "maxiter": 1000},
    )
    weights = result.x

    synthetic_pre  = donor_pre  @ weights
    synthetic_post = donor_post @ weights

    pre_rmspe  = float(np.sqrt(np.mean((treated_pre  - synthetic_pre)  ** 2)))
    gap_post   = treated_post - synthetic_post
    post_rmspe = float(np.sqrt(np.mean(gap_post ** 2)))

    gap_series = pd.Series(
        gap_post,
        index=pivot.index[post_mask],
        name=treated_appid,
    )

    top_donors = sorted(
        zip(available_donors, weights), key=lambda x: -x[1]
    )[:5]

    return {
        "treated_appid": treated_appid,
        "weights": dict(zip(available_donors, weights.tolist())),
        "top_donors": top_donors,
        "pre_rmspe": pre_rmspe,
        "post_rmspe": post_rmspe,
        "mean_gap_post": float(gap_post.mean()),
        "gap_series": gap_series,
        "n_donors": len(available_donors),
        "convergence": bool(result.success),
    }


def placebo_test(
    panel: pd.DataFrame,
    treated_appids: list[int],
    donor_appids: list[int],
    n_placebos: int = 50,
    seed: int = 42,
    **kwargs,
) -> dict:
    """
    Apply synthetic control to n_placebos randomly chosen non-treated units.
    Returns the distribution of post-period gaps for inference.
    p-value ≈ rank of treated gap in placebo distribution.
    """
    rng = np.random.default_rng(seed)
    pool = [a for a in donor_appids if a not in treated_appids]
    placebo_units = rng.choice(pool, size=min(n_placebos, len(pool)), replace=False)

    placebo_gaps = []
    for placebo_id in placebo_units:
        placebo_donors = [a for a in pool if a != placebo_id]
        result = fit_synthetic_control(
            panel, int(placebo_id), placebo_donors, **kwargs
        )
        if "error" not in result and result["pre_rmspe"] > 0:
            # Normalise by pre-RMSPE so units are comparable
            ratio = result["post_rmspe"] / result["pre_rmspe"]
            placebo_gaps.append({"appid": int(placebo_id), "ratio": ratio,
                                  "mean_gap": result["mean_gap_post"]})

    return {
        "placebo_results": placebo_gaps,
        "n_placebos": len(placebo_gaps),
    }


def compute_pvalue(treated_ratio: float, placebo_results: list[dict]) -> float:
    """
    p-value = fraction of placebos with post/pre RMSPE ratio ≥ treated ratio.
    """
    placebo_ratios = [r["ratio"] for r in placebo_results]
    return float(np.mean(np.array(placebo_ratios) >= treated_ratio))
