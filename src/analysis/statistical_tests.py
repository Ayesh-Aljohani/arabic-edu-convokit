"""Statistical test utilities: bootstrap CIs, effect sizes, correlations."""

import logging
from typing import Callable

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# Reproducibility
RANDOM_SEED = 42


def bootstrap_ci(
    data: np.ndarray,
    func: Callable[[np.ndarray], float],
    n_boot: int = 10000,
    ci: float = 0.95,
) -> dict:
    """Compute a bootstrap confidence interval for any scalar statistic.

    Parameters
    ----------
    data : np.ndarray
        1-D array of observations.
    func : callable
        A function that takes a 1-D array and returns a scalar statistic.
    n_boot : int
        Number of bootstrap resamples.
    ci : float
        Confidence level (e.g. 0.95 for a 95 % interval).

    Returns
    -------
    dict
        Keys: ``observed``, ``ci_lower``, ``ci_upper``, ``boot_mean``,
        ``boot_std``, ``n_boot``, ``ci_level``.
    """
    rng = np.random.RandomState(RANDOM_SEED)
    data = np.asarray(data)
    observed = float(func(data))

    boot_stats = np.empty(n_boot)
    n = len(data)
    for i in range(n_boot):
        sample = data[rng.randint(0, n, size=n)]
        boot_stats[i] = func(sample)

    alpha = 1.0 - ci
    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1.0 - alpha / 2)))

    logger.debug(
        "bootstrap_ci: observed=%.4f  CI=[%.4f, %.4f]  n_boot=%d",
        observed, ci_lower, ci_upper, n_boot,
    )

    return {
        "observed": observed,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "boot_mean": float(np.mean(boot_stats)),
        "boot_std": float(np.std(boot_stats, ddof=1)),
        "n_boot": n_boot,
        "ci_level": ci,
    }


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size for two independent or paired groups.

    Uses the pooled standard deviation as the denominator.

    Parameters
    ----------
    group1, group2 : np.ndarray
        Arrays of observations (same or different lengths).

    Returns
    -------
    float
        Cohen's d value.  Positive when group1 mean > group2 mean.
    """
    group1 = np.asarray(group1, dtype=float)
    group2 = np.asarray(group2, dtype=float)

    n1, n2 = len(group1), len(group2)
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        logger.warning("cohens_d: pooled std is zero; returning 0.0")
        return 0.0

    d = float((np.mean(group1) - np.mean(group2)) / pooled_std)
    logger.debug("cohens_d: d=%.4f  n1=%d  n2=%d", d, n1, n2)
    return d


def compute_correlation_with_ci(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "pearson",
    n_boot: int = 10000,
    ci: float = 0.95,
) -> dict:
    """Compute a correlation coefficient with a bootstrap confidence interval.

    Parameters
    ----------
    x, y : np.ndarray
        Paired observations of equal length.
    method : str
        ``'pearson'`` or ``'spearman'``.
    n_boot : int
        Number of bootstrap resamples.
    ci : float
        Confidence level.

    Returns
    -------
    dict
        Keys: ``method``, ``r``, ``p_value``, ``ci_lower``, ``ci_upper``,
        ``n``, ``n_boot``, ``ci_level``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y):
        raise ValueError(
            f"x and y must have the same length, got {len(x)} and {len(y)}"
        )

    if method == "pearson":
        r, p = stats.pearsonr(x, y)
    elif method == "spearman":
        r, p = stats.spearmanr(x, y)
    else:
        raise ValueError(f"Unknown method '{method}'; expected 'pearson' or 'spearman'")

    # Bootstrap CI on the correlation coefficient
    rng = np.random.RandomState(RANDOM_SEED)
    n = len(x)
    boot_rs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if method == "pearson":
            boot_rs[i] = stats.pearsonr(x[idx], y[idx])[0]
        else:
            boot_rs[i] = stats.spearmanr(x[idx], y[idx])[0]

    alpha = 1.0 - ci
    ci_lower = float(np.percentile(boot_rs, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_rs, 100 * (1.0 - alpha / 2)))

    logger.info(
        "correlation (%s): r=%.4f  p=%.2e  CI=[%.4f, %.4f]  n=%d",
        method, r, p, ci_lower, ci_upper, n,
    )

    return {
        "method": method,
        "r": float(r),
        "p_value": float(p),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n": n,
        "n_boot": n_boot,
        "ci_level": ci,
    }
