"""Evaluation utilities: metrics, confidence intervals, result serialization."""

import json
import logging
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(y_true: list, y_pred: list) -> dict:
    """Compute classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_positive": float(f1_score(y_true, y_pred, pos_label=1)),
        "precision_positive": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_positive": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """Compute mean, std, and 95% CI across folds."""
    metric_names = [
        "accuracy", "f1_weighted", "f1_macro", "f1_positive",
        "precision_positive", "recall_positive",
    ]
    result = {"n_folds": len(fold_metrics), "per_fold": fold_metrics}

    for metric in metric_names:
        values = [fm[metric] for fm in fold_metrics]
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        n = len(values)
        se = std / np.sqrt(n)
        ci_low = mean - stats.t.ppf(0.975, df=n - 1) * se
        ci_high = mean + stats.t.ppf(0.975, df=n - 1) * se

        result[metric] = {
            "mean": mean,
            "std": std,
            "ci_95_low": float(ci_low),
            "ci_95_high": float(ci_high),
            "values": values,
        }

    return result


def save_results(results: dict, output_path: str | Path) -> None:
    """Save results as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_path)
