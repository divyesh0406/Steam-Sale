"""
CUPED (Controlled-experiment Using Pre-Experiment Data) variance reduction.
Reference: Deng et al. 2013 — "Improving the Sensitivity of Online Controlled Experiments
using Control Variates" (Microsoft).
"""
import numpy as np
import pandas as pd


def cuped_adjust(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Adjust outcome y using pre-experiment covariate x.
    y_cuped = y - theta * (x - mean(x))
    theta = Cov(y, x) / Var(x)
    Returns (y_cuped, theta).
    """
    theta = np.cov(y, x)[0, 1] / np.var(x)
    y_adj = y - theta * (x - x.mean())
    return y_adj, float(theta)


def variance_reduction(y_raw: np.ndarray, y_cuped: np.ndarray) -> float:
    """Fraction of variance removed by CUPED. Typical range: 0.20–0.45."""
    return float(1.0 - y_cuped.var() / y_raw.var())


def cuped_did_estimate(
    df: pd.DataFrame,
    outcome: str,
    pre_covariate: str,
    treatment_col: str = "treated",
) -> dict:
    """
    CUPED-adjusted ATT (average treatment effect on the treated).

    df has one row per unit with columns:
        outcome       — post-period outcome (e.g. log_post_reviews)
        pre_covariate — pre-period value of the same metric
        treatment_col — 1/True = treated, 0/False = control

    Returns dict with ATT, SE (delta method), variance reduction, theta.
    """
    treated = df[treatment_col].astype(bool)
    y = df[outcome].values.astype(float)
    x = df[pre_covariate].values.astype(float)

    y_cuped, theta = cuped_adjust(y, x)
    vr = variance_reduction(y, y_cuped)

    att = y_cuped[treated].mean() - y_cuped[~treated].mean()
    se = np.sqrt(
        y_cuped[treated].var() / treated.sum()
        + y_cuped[~treated].var() / (~treated).sum()
    )

    return {
        "att": float(att),
        "se": float(se),
        "ci_low": float(att - 1.96 * se),
        "ci_high": float(att + 1.96 * se),
        "theta": theta,
        "variance_reduction": vr,
        "n_treated": int(treated.sum()),
        "n_control": int((~treated).sum()),
    }
