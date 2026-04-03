"""Cross-linguistic comparison between Arabic and English features."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.statistical_tests import (
    bootstrap_ci,
    cohens_d,
    compute_correlation_with_ci,
)

logger = logging.getLogger(__name__)

# Reproducibility
np.random.seed(42)


def compute_talk_time_correlation(df: pd.DataFrame) -> dict:
    """Compute Pearson and Spearman correlations between EN and AR word counts.

    Both correlations are accompanied by 95 % bootstrap confidence intervals
    (10 000 resamples, seed 42).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``talktime_words`` (English) and
        ``talktime_words_arabic`` (Arabic).

    Returns
    -------
    dict
        Top-level keys: ``pearson``, ``spearman``, ``n_observations``,
        ``en_mean``, ``en_std``, ``ar_mean``, ``ar_std``.
    """
    en = df["talktime_words"].values.astype(float)
    ar = df["talktime_words_arabic"].values.astype(float)

    # Drop rows where either value is NaN
    mask = ~(np.isnan(en) | np.isnan(ar))
    en = en[mask]
    ar = ar[mask]

    logger.info(
        "Computing talk-time correlations on %d observations (dropped %d NaN rows)",
        len(en), (~mask).sum(),
    )

    pearson_result = compute_correlation_with_ci(en, ar, method="pearson")
    spearman_result = compute_correlation_with_ci(en, ar, method="spearman")

    return {
        "pearson": pearson_result,
        "spearman": spearman_result,
        "n_observations": int(len(en)),
        "en_mean": float(np.mean(en)),
        "en_std": float(np.std(en, ddof=1)),
        "ar_mean": float(np.mean(ar)),
        "ar_std": float(np.std(ar, ddof=1)),
    }


def compute_paired_tests(df: pd.DataFrame) -> dict:
    """Run paired t-test and Wilcoxon signed-rank test on EN vs AR word counts.

    Reports t-statistic, p-value, and Cohen's d effect size for the parametric
    test, plus the Wilcoxon statistic and p-value for the non-parametric test.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``talktime_words`` and ``talktime_words_arabic``.

    Returns
    -------
    dict
        Top-level keys: ``paired_ttest``, ``wilcoxon``, ``descriptive``.
    """
    en = df["talktime_words"].values.astype(float)
    ar = df["talktime_words_arabic"].values.astype(float)

    mask = ~(np.isnan(en) | np.isnan(ar))
    en = en[mask]
    ar = ar[mask]

    logger.info("Running paired tests on %d paired observations", len(en))

    # Paired t-test
    t_stat, t_pval = stats.ttest_rel(en, ar)
    d = cohens_d(en, ar)

    # Interpret effect size
    abs_d = abs(d)
    if abs_d < 0.2:
        d_interpretation = "negligible"
    elif abs_d < 0.5:
        d_interpretation = "small"
    elif abs_d < 0.8:
        d_interpretation = "medium"
    else:
        d_interpretation = "large"

    # Wilcoxon signed-rank test
    diff = en - ar
    # Remove zero differences (Wilcoxon cannot handle them)
    nonzero_diff = diff[diff != 0]
    if len(nonzero_diff) > 0:
        w_stat, w_pval = stats.wilcoxon(nonzero_diff)
    else:
        logger.warning("All differences are zero; Wilcoxon test is undefined")
        w_stat, w_pval = float("nan"), float("nan")

    # Descriptive statistics on the differences
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    median_diff = float(np.median(diff))

    # Bootstrap CI on the mean difference
    diff_ci = bootstrap_ci(diff, func=np.mean, n_boot=10000, ci=0.95)

    logger.info(
        "Paired t-test: t=%.4f  p=%.2e  d=%.4f (%s)",
        t_stat, t_pval, d, d_interpretation,
    )
    logger.info("Wilcoxon: W=%.1f  p=%.2e", w_stat, w_pval)

    return {
        "paired_ttest": {
            "t_statistic": float(t_stat),
            "p_value": float(t_pval),
            "cohens_d": d,
            "effect_size_interpretation": d_interpretation,
        },
        "wilcoxon": {
            "W_statistic": float(w_stat),
            "p_value": float(w_pval),
            "n_nonzero_differences": int(len(nonzero_diff)),
        },
        "descriptive": {
            "mean_difference_en_minus_ar": mean_diff,
            "std_difference": std_diff,
            "median_difference": median_diff,
            "mean_diff_ci_95_lower": diff_ci["ci_lower"],
            "mean_diff_ci_95_upper": diff_ci["ci_upper"],
        },
        "n_observations": int(len(en)),
    }


def run_cross_linguistic_analysis(root_dir: str) -> dict:
    """Master function: load data, run all cross-linguistic analyses, save.

    Loads ``data/processed/all_sessions.csv``, computes correlations and
    paired tests, and writes the combined results to
    ``results/analysis/cross_linguistic.json``.

    Parameters
    ----------
    root_dir : str
        Project root directory (e.g. ``'.'``).

    Returns
    -------
    dict
        The full results dictionary that was saved to disk.
    """
    root = Path(root_dir)
    data_path = root / "data" / "processed" / "all_sessions.csv"
    output_path = root / "results" / "analysis" / "cross_linguistic.json"

    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)

    # Require the necessary columns
    required_cols = {"talktime_words", "talktime_words_arabic"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info("Dataset shape: %s", df.shape)

    results = {
        "talk_time_correlation": compute_talk_time_correlation(df),
        "paired_tests": compute_paired_tests(df),
        "dataset_info": {
            "source_file": str(data_path),
            "n_rows": len(df),
            "n_columns": len(df.columns),
        },
    }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Cross-linguistic results saved to %s", output_path)

    return results
