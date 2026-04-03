"""Phase 4: Run all classification experiments — baselines, transformers, cross-lingual.

Fully resumable: checks for existing result files before running each model-task combo.
"""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.classification.baselines import train_dummy, train_tfidf_lr, train_tfidf_svm
from src.classification.evaluate import aggregate_fold_metrics, compute_metrics, save_results
from src.classification.train import run_cv_transformer, set_seed
from src.classification.cross_lingual import train_english_test_arabic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def load_task_data(df: pd.DataFrame, task: str) -> tuple[list[str], np.ndarray]:
    """Extract texts and labels for a task, dropping NaN labels."""
    mask = df[task].notna()
    texts = df.loc[mask, "text_arabic_normalized"].tolist()
    labels = df.loc[mask, task].astype(int).values
    return texts, labels


def load_task_data_bilingual(
    df: pd.DataFrame, task: str
) -> tuple[list[str], list[str], np.ndarray]:
    """Extract English texts, Arabic texts, and labels for cross-lingual."""
    mask = df[task].notna()
    en_texts = df.loc[mask, "text"].tolist()
    ar_texts = df.loc[mask, "text_arabic_normalized"].tolist()
    labels = df.loc[mask, task].astype(int).values
    return en_texts, ar_texts, labels


def _result_path(results_dir: Path, task: str, model: str) -> Path:
    return results_dir / f"{task}_{model}_results.json"


def _load_existing(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def run_baseline_cv(
    texts: list[str],
    labels: np.ndarray,
    task: str,
    n_folds: int = 5,
    config: dict = None,
    results_dir: Path = None,
) -> dict:
    """Run CV for all baseline models on a task. Resumes from saved results."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {}

    for model_name in ["dummy_most_frequent", "dummy_stratified", "tfidf_lr", "tfidf_svm"]:
        rpath = _result_path(results_dir, task, model_name)
        existing = _load_existing(rpath)
        if existing:
            logger.info("%s / %s: loaded from cache", task, model_name)
            results[model_name] = existing
            continue

        fold_metrics = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
            X_train = [texts[i] for i in train_idx]
            y_train = labels[train_idx]
            X_val = [texts[i] for i in val_idx]
            y_val = labels[val_idx]

            if model_name == "dummy_most_frequent":
                pred_result = train_dummy(y_train, y_val, "most_frequent")
            elif model_name == "dummy_stratified":
                pred_result = train_dummy(y_train, y_val, "stratified")
            elif model_name == "tfidf_lr":
                pred_result = train_tfidf_lr(
                    X_train, y_train, X_val,
                    max_features=config["baselines"]["tfidf"]["max_features"],
                    ngram_range=tuple(config["baselines"]["tfidf"]["ngram_range"]),
                    C=config["baselines"]["logistic_regression"]["C"],
                    max_iter=config["baselines"]["logistic_regression"]["max_iter"],
                )
            elif model_name == "tfidf_svm":
                pred_result = train_tfidf_svm(
                    X_train, y_train, X_val,
                    max_features=config["baselines"]["tfidf"]["max_features"],
                    ngram_range=tuple(config["baselines"]["tfidf"]["ngram_range"]),
                    C=config["baselines"]["svm"]["C"],
                )

            metrics = compute_metrics(y_val.tolist(), pred_result["y_pred"])
            metrics["fold"] = fold_idx
            fold_metrics.append(metrics)

        aggregated = aggregate_fold_metrics(fold_metrics)
        results[model_name] = aggregated
        save_results(aggregated, rpath)

        logger.info(
            "%s / %s: acc=%.4f+/-%.4f  f1=%.4f+/-%.4f",
            task, model_name,
            aggregated["accuracy"]["mean"], aggregated["accuracy"]["std"],
            aggregated["f1_weighted"]["mean"], aggregated["f1_weighted"]["std"],
        )

    return results


def run_transformer_task(
    model_key: str,
    model_cfg: dict,
    texts: list[str],
    labels: np.ndarray,
    task: str,
    n_folds: int,
    results_dir: Path,
) -> dict:
    """Run 5-fold CV for one transformer on one task. Returns cached if exists."""
    rpath = _result_path(results_dir, task, model_key)
    existing = _load_existing(rpath)
    if existing:
        logger.info("%s / %s: loaded from cache", task, model_key)
        return existing

    logger.info("Running %s for %s", model_key, task)
    folds = run_cv_transformer(
        model_name=model_cfg["name"],
        texts=texts,
        labels=labels,
        task_name=task,
        n_folds=n_folds,
        output_base=str(results_dir),
        max_length=model_cfg["max_length"],
        learning_rate=model_cfg["learning_rate"],
        batch_size=model_cfg["batch_size"],
        num_epochs=model_cfg["num_epochs"],
        weight_decay=model_cfg["weight_decay"],
        warmup_ratio=model_cfg["warmup_ratio"],
        patience=model_cfg["early_stopping_patience"],
    )
    aggregated = aggregate_fold_metrics(folds)
    save_results(aggregated, rpath)

    logger.info(
        "%s / %s: acc=%.4f+/-%.4f  f1=%.4f+/-%.4f",
        task, model_key,
        aggregated["accuracy"]["mean"], aggregated["accuracy"]["std"],
        aggregated["f1_weighted"]["mean"], aggregated["f1_weighted"]["std"],
    )
    return aggregated


def main() -> None:
    set_seed(42)

    with open(ROOT / "config" / "config.yaml") as f:
        config = yaml.safe_load(f)

    processed_path = ROOT / "data" / "processed" / "all_sessions.csv"
    results_dir = ROOT / "results" / "classification"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(processed_path)
    logger.info("Loaded %d utterances", len(df))

    tasks = config["classification"]["tasks"]
    n_folds = config["classification"]["cv_folds"]
    all_results = {}

    for task in tasks:
        logger.info("=== Task: %s ===", task)
        texts, labels = load_task_data(df, task)
        logger.info(
            "Samples: %d  Positive: %d (%.1f%%)",
            len(labels), labels.sum(), 100 * labels.mean(),
        )

        task_results = {}
        start = time.time()

        # Baselines
        baseline_results = run_baseline_cv(
            texts, labels, task, n_folds, config, results_dir
        )
        task_results.update(baseline_results)

        # Transformers
        for model_key in ["arabert", "mbert", "xlmr"]:
            model_cfg = config["models"][model_key]
            result = run_transformer_task(
                model_key, model_cfg, texts, labels, task, n_folds, results_dir
            )
            task_results[model_key] = result

        elapsed = time.time() - start
        logger.info("Task %s completed in %.1f seconds", task, elapsed)
        all_results[task] = task_results

    # Cross-lingual zero-shot
    logger.info("=== Cross-lingual zero-shot transfer ===")
    cl_path = results_dir / "cross_lingual_results.json"
    existing_cl = _load_existing(cl_path)
    if existing_cl:
        logger.info("Cross-lingual results loaded from cache")
        cross_lingual_results = existing_cl
    else:
        cross_lingual_results = {}
        for task in tasks:
            en_texts, ar_texts, labels = load_task_data_bilingual(df, task)
            task_cl = {}

            for model_key in ["mbert", "xlmr"]:
                cl_rpath = _result_path(results_dir, task, f"cross_lingual_{model_key}")
                existing_cl_model = _load_existing(cl_rpath)
                if existing_cl_model:
                    logger.info("Cross-lingual %s on %s: loaded from cache", model_key, task)
                    task_cl[model_key] = existing_cl_model
                    continue

                model_cfg = config["models"][model_key]
                logger.info("Cross-lingual %s on %s", model_key, task)
                cl_result = train_english_test_arabic(
                    model_name=model_cfg["name"],
                    X_train_en=en_texts,
                    y_train=labels,
                    X_test_ar=ar_texts,
                    y_test=labels,
                    output_dir=str(results_dir / f"{task}/cross_lingual_{model_key}"),
                    max_length=model_cfg["max_length"],
                    learning_rate=model_cfg["learning_rate"],
                    batch_size=model_cfg["batch_size"],
                    num_epochs=model_cfg["num_epochs"],
                    weight_decay=model_cfg["weight_decay"],
                    warmup_ratio=model_cfg["warmup_ratio"],
                    patience=model_cfg["early_stopping_patience"],
                )
                task_cl[model_key] = cl_result
                save_results(cl_result, cl_rpath)

            cross_lingual_results[task] = task_cl

        save_results(cross_lingual_results, cl_path)

    # Final summary
    logger.info("=== FINAL SUMMARY ===")
    for task in tasks:
        logger.info("--- %s ---", task)
        for model_name, res in all_results[task].items():
            if "accuracy" in res:
                logger.info(
                    "  %s: acc=%.4f+/-%.4f  f1=%.4f+/-%.4f",
                    model_name,
                    res["accuracy"]["mean"], res["accuracy"]["std"],
                    res["f1_weighted"]["mean"], res["f1_weighted"]["std"],
                )

    save_results(all_results, results_dir / "all_results.json")
    logger.info("All results saved to %s", results_dir)


if __name__ == "__main__":
    main()
