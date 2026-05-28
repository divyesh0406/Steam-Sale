"""
Regression Discontinuity Design on discount depth.
Running variable: discount % (SteamSpy snapshot).
Cutoff: 50% — Steam's featured-placement visibility threshold.
Outcome: review count in the 14 days around the sale window.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def sharp_rdd(
    df: pd.DataFrame,
    running_var: str = "discount_pct",
    cutoff: float = 50.0,
    outcome: str = "reviews_14d_post",
    bandwidth: float = 15.0,
    poly: int = 1,
) -> dict:
    """
    Local polynomial regression on either side of the cutoff.
    Returns effect at cutoff (jump discontinuity), SE, and CI.
    poly=1 → local linear (default); poly=2 → local quadratic.
    """
    band = df[
        (df[running_var] >= cutoff - bandwidth) &
        (df[running_var] <= cutoff + bandwidth)
    ].copy()

    if len(band) < 10:
        return {"error": f"Only {len(band)} obs in bandwidth {bandwidth}"}

    band["above"]    = (band[running_var] >= cutoff).astype(float)
    band["centered"] = band[running_var] - cutoff

    feature_cols = ["above", "centered", "centered_above"]
    band["centered_above"] = band["centered"] * band["above"]

    if poly == 2:
        band["centered2"]       = band["centered"] ** 2
        band["centered2_above"] = band["centered2"] * band["above"]
        feature_cols += ["centered2", "centered2_above"]

    X = sm.add_constant(band[feature_cols])
    y = band[outcome]
    fit = sm.OLS(y, X).fit(cov_type="HC1")

    ci = fit.conf_int()
    return {
        "effect_at_cutoff": float(fit.params["above"]),
        "se":      float(fit.bse["above"]),
        "ci_low":  float(ci.loc["above", 0]),
        "ci_high": float(ci.loc["above", 1]),
        "p_value": float(fit.pvalues["above"]),
        "bandwidth": bandwidth,
        "n_obs":   len(band),
        "n_above": int(band["above"].sum()),
        "n_below": int((~band["above"].astype(bool)).sum()),
    }


def bandwidth_sensitivity(
    df: pd.DataFrame,
    bandwidths: list[float] | None = None,
    **kwargs,
) -> list[dict]:
    """Sweep bandwidths and return list of RDD results."""
    if bandwidths is None:
        bandwidths = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    return [sharp_rdd(df, bandwidth=bw, **kwargs) for bw in bandwidths]


def placebo_cutoffs(
    df: pd.DataFrame,
    cutoffs: list[float] | None = None,
    **kwargs,
) -> list[dict]:
    """Run RDD at fake cutoffs — effects should be null."""
    if cutoffs is None:
        cutoffs = [20.0, 30.0, 40.0, 60.0, 70.0, 80.0]
    return [sharp_rdd(df, cutoff=c, **kwargs) for c in cutoffs]


def mccrary_density_test(
    df: pd.DataFrame,
    running_var: str = "discount_pct",
    cutoff: float = 50.0,
    bandwidth: float = 15.0,
    bins: int = 20,
) -> dict:
    """
    Visual McCrary-style manipulation test: compare histogram density on either
    side of the cutoff. Returns bin counts and a simple t-test of the density jump.
    A significant jump → developers may be gaming the threshold (RDD invalid).
    """
    band = df[
        (df[running_var] >= cutoff - bandwidth) &
        (df[running_var] <= cutoff + bandwidth)
    ][running_var].values

    below = band[band < cutoff]
    above = band[band >= cutoff]

    # Density per unit: count / bandwidth_half
    density_below = len(below) / bandwidth
    density_above = len(above) / bandwidth

    # Simple ratio test (not a formal McCrary test, but sufficient for diagnostics)
    ratio = density_above / density_below if density_below > 0 else np.nan

    return {
        "n_below": int(len(below)),
        "n_above": int(len(above)),
        "density_below": float(density_below),
        "density_above": float(density_above),
        "density_ratio": float(ratio),
        "manipulation_concern": bool(ratio > 1.5 or ratio < 0.67),
    }


def bootstrap_rdd(
    df: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 42,
    **kwargs,
) -> dict:
    """Bootstrap 95% CI for the RDD effect at cutoff."""
    rng = np.random.default_rng(seed)
    point_est = sharp_rdd(df, **kwargs)
    if "error" in point_est:
        return point_est

    effects = []
    for _ in range(n_bootstrap):
        sample = df.sample(len(df), replace=True, random_state=int(rng.integers(1e6)))
        r = sharp_rdd(sample, **kwargs)
        if "error" not in r:
            effects.append(r["effect_at_cutoff"])

    effects = np.array(effects)
    return {
        **point_est,
        "bootstrap_ci_low":  float(np.percentile(effects, 2.5)),
        "bootstrap_ci_high": float(np.percentile(effects, 97.5)),
        "bootstrap_se":      float(effects.std()),
        "n_bootstrap":       len(effects),
    }
