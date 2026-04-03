"""Error analysis: extract and characterize misclassified examples."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

# Reproducibility
np.random.seed(42)

# Task column names in all_sessions.csv
TASK_COLUMNS = {
    "focusing_questions": "focusing_questions",
    "student_reasoning": "student_reasoning",
    "uptake": "uptake",
}

# Best model per task (used by the master function)
BEST_MODELS = {
    "focusing_questions": "arabert",
    "student_reasoning": "mbert",
    "uptake": "xlmr",
}


def extract_error_examples(
    root_dir: str,
    task: str,
    model_key: str,
    n_examples: int = 10,
) -> list[dict]:
    """Extract misclassified examples for one task/model combination.

    Loads the model's per-fold predictions from
    ``results/classification/{task}_{model_key}_results.json``, reconstructs
    which rows were in each fold using the same ``StratifiedKFold(n_splits=5,
    shuffle=True, random_state=42)`` split, and returns up to *n_examples*
    misclassified examples with full context.

    Parameters
    ----------
    root_dir : str
        Project root directory.
    task : str
        Task name (e.g. ``'focusing_questions'``).
    model_key : str
        Short model identifier matching the results filename
        (e.g. ``'arabert'``, ``'mbert'``, ``'xlmr'``).
    n_examples : int
        Maximum number of error examples to return.

    Returns
    -------
    list[dict]
        Each dict contains: ``text_english``, ``text_arabic``, ``true_label``,
        ``predicted_label``, ``confidence``, ``fold``, ``row_index``.
    """
    root = Path(root_dir)

    # ------------------------------------------------------------------
    # 1. Load the results JSON
    # ------------------------------------------------------------------
    results_path = root / "results" / "classification" / f"{task}_{model_key}_results.json"
    if not results_path.exists():
        logger.warning("Results file not found: %s", results_path)
        return []

    with open(results_path) as f:
        results = json.load(f)

    per_fold = results["per_fold"]
    n_folds = results.get("n_folds", len(per_fold))

    logger.info(
        "Loaded results for %s / %s (%d folds)", task, model_key, n_folds,
    )

    # ------------------------------------------------------------------
    # 2. Load the dataset and reconstruct fold indices
    # ------------------------------------------------------------------
    data_path = root / "data" / "processed" / "all_sessions.csv"
    df = pd.read_csv(data_path)

    label_col = TASK_COLUMNS[task]
    mask = df[label_col].notna()
    df_task = df.loc[mask].reset_index(drop=True)
    labels = df_task[label_col].astype(int).values
    texts_en = df_task["text"].values
    texts_ar = df_task["text_arabic"].values

    logger.info(
        "Task '%s': %d labelled rows (positive rate %.1f%%)",
        task, len(labels), 100 * labels.mean(),
    )

    # Reproduce the exact same fold splits used during training
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_val_indices = [
        val_idx for _, val_idx in skf.split(np.zeros(len(labels)), labels)
    ]

    # ------------------------------------------------------------------
    # 3. Collect all misclassified examples across folds
    # ------------------------------------------------------------------
    errors: list[dict] = []

    for fold_idx, fold_data in enumerate(per_fold):
        y_true = np.array(fold_data["y_true"])
        y_pred = np.array(fold_data["y_pred"])
        y_proba = np.array(fold_data["y_proba"])

        val_idx = fold_val_indices[fold_idx]

        # Sanity check: fold size must match
        if len(val_idx) != len(y_true):
            logger.warning(
                "Fold %d: expected %d samples but got %d in results; skipping",
                fold_idx, len(val_idx), len(y_true),
            )
            continue

        # Find misclassified positions within this fold
        wrong = np.where(y_true != y_pred)[0]

        for pos in wrong:
            global_idx = int(val_idx[pos])
            # Confidence is P(predicted_class): if prediction is 1,
            # confidence = y_proba; if prediction is 0, confidence = 1 - y_proba
            pred_label = int(y_pred[pos])
            confidence = float(y_proba[pos]) if pred_label == 1 else 1.0 - float(y_proba[pos])

            errors.append({
                "text_english": str(texts_en[global_idx]),
                "text_arabic": str(texts_ar[global_idx]),
                "true_label": int(y_true[pos]),
                "predicted_label": pred_label,
                "confidence": round(confidence, 6),
                "fold": fold_idx,
                "row_index": global_idx,
            })

    logger.info(
        "Found %d total misclassified examples across %d folds",
        len(errors), n_folds,
    )

    # ------------------------------------------------------------------
    # 4. Return a diverse sample of errors
    # ------------------------------------------------------------------
    if len(errors) <= n_examples:
        selected = errors
    else:
        # Deterministic sampling: pick the highest-confidence errors
        # (the model was most wrong on these)
        errors.sort(key=lambda e: e["confidence"], reverse=True)
        selected = errors[:n_examples]

    logger.info("Returning %d error examples", len(selected))
    return selected


def run_error_analysis(root_dir: str) -> dict:
    """Run error analysis for every task using the best model.

    Best models per task:
    - focusing_questions: AraBERT
    - student_reasoning: mBERT
    - uptake: XLM-R

    Results are saved to ``results/analysis/error_analysis.json``.

    Parameters
    ----------
    root_dir : str
        Project root directory.

    Returns
    -------
    dict
        Keys are task names; values contain ``model``, ``n_errors``,
        ``n_total``, and ``examples`` (list of error dicts).
    """
    root = Path(root_dir)
    output_path = root / "results" / "analysis" / "error_analysis.json"

    all_results: dict = {}

    for task, model_key in BEST_MODELS.items():
        logger.info("--- Error analysis: %s (%s) ---", task, model_key)

        # Load results to get total sample count and overall error rate
        results_path = root / "results" / "classification" / f"{task}_{model_key}_results.json"
        if not results_path.exists():
            logger.warning("Skipping %s: results file not found", task)
            continue

        with open(results_path) as f:
            results_data = json.load(f)

        n_total = sum(len(f["y_true"]) for f in results_data["per_fold"])
        n_errors_total = sum(
            int(np.sum(np.array(f["y_true"]) != np.array(f["y_pred"])))
            for f in results_data["per_fold"]
        )

        examples = extract_error_examples(
            root_dir=root_dir,
            task=task,
            model_key=model_key,
            n_examples=10,
        )

        all_results[task] = {
            "model": model_key,
            "n_errors": n_errors_total,
            "n_total": n_total,
            "error_rate": round(n_errors_total / n_total, 6) if n_total > 0 else 0.0,
            "examples": examples,
        }

        logger.info(
            "%s: %d / %d misclassified (%.2f%%)",
            task, n_errors_total, n_total,
            100 * n_errors_total / n_total if n_total > 0 else 0.0,
        )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("Error analysis saved to %s", output_path)

    return all_results
