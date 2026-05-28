"""
Difference-in-Differences analysis module.
Computes ATT via simple 2x2 DiD and permutation-based inference.
"""
import numpy as np
import pandas as pd
from src.analysis.cuped import cuped_did_estimate, variance_reduction, cuped_adjust


def simple_did(
    df: pd.DataFrame,
    outcome_pre: str,
    outcome_post: str,
    treatment_col: str = "treated",
) -> dict:
    """
    2×2 DiD: (post_treated - pre_treated) - (post_control - pre_control).
    Returns ATT and SE via delta method.
    """
    treated = df[treatment_col].astype(bool)
    pre  = df[outcome_pre].values.astype(float)
    post = df[outcome_post].values.astype(float)

    delta_t = post[treated].mean()  - pre[treated].mean()
    delta_c = post[~treated].mean() - pre[~treated].mean()
    att = delta_t - delta_c

    # SE via delta method (treat pre/post as independent draws per unit)
    var_t = (post[treated].var()  / treated.sum()  + pre[treated].var()  / treated.sum())
    var_c = (post[~treated].var() / (~treated).sum() + pre[~treated].var() / (~treated).sum())
    se = np.sqrt(var_t + var_c)

    return {
        "att": float(att),
        "se": float(se),
        "ci_low": float(att - 1.96 * se),
        "ci_high": float(att + 1.96 * se),
        "delta_treated": float(delta_t),
        "delta_control": float(delta_c),
        "n_treated": int(treated.sum()),
        "n_control": int((~treated).sum()),
    }


def did_with_cuped(
    df: pd.DataFrame,
    outcome_pre: str,
    outcome_post: str,
    treatment_col: str = "treated",
) -> dict:
    """
    DiD with CUPED adjustment on the post-period outcome.
    pre-period value used as the CUPED covariate.
    """
    naive = simple_did(df, outcome_pre, outcome_post, treatment_col)
    cuped = cuped_did_estimate(df, outcome_post, outcome_pre, treatment_col)

    return {
        "naive_att": naive["att"],
        "naive_se":  naive["se"],
        "naive_ci_low":  naive["ci_low"],
        "naive_ci_high": naive["ci_high"],
        "cuped_att": cuped["att"],
        "cuped_se":  cuped["se"],
        "cuped_ci_low":  cuped["ci_low"],
        "cuped_ci_high": cuped["ci_high"],
        "theta": cuped["theta"],
        "variance_reduction": cuped["variance_reduction"],
        "n_treated": naive["n_treated"],
        "n_control": naive["n_control"],
    }


def permutation_test(
    df: pd.DataFrame,
    outcome_pre: str,
    outcome_post: str,
    treatment_col: str = "treated",
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Permutation test: randomly reassign treatment labels n_permutations times
    and record the distribution of ATT estimates. Returns two-sided p-value.
    """
    rng = np.random.default_rng(seed)
    observed = did_with_cuped(df, outcome_pre, outcome_post, treatment_col)
    real_att = observed["cuped_att"]

    treated_flags = df[treatment_col].values.copy()
    null_atts = []
    for _ in range(n_permutations):
        perm = rng.permutation(treated_flags)
        df_perm = df.copy()
        df_perm[treatment_col] = perm
        null_atts.append(
            cuped_did_estimate(df_perm, outcome_post, outcome_pre, treatment_col)["att"]
        )

    null_atts = np.array(null_atts)
    p_value = float((np.abs(null_atts) >= np.abs(real_att)).mean())

    return {
        "observed_att": real_att,
        "p_value": p_value,
        "null_mean": float(null_atts.mean()),
        "null_std": float(null_atts.std()),
        "null_distribution": null_atts.tolist(),
    }
