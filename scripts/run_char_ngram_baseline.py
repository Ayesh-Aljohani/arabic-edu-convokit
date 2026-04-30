"""Char-n-gram TF-IDF baselines (3-5 chars) for Arabic classification.

Addresses reviewer R1-W5: word-level TF-IDF baselines are sub-optimal for
morphologically rich languages; char n-grams are the field-standard for Arabic.
"""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.classification.evaluate import (
    aggregate_fold_metrics,
    compute_metrics,
    save_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT / "results" / "classification"


def load_task_data(df: pd.DataFrame, task: str) -> tuple[list[str], np.ndarray]:
    mask = df[task].notna()
    texts = df.loc[mask, "text_arabic_normalized"].tolist()
    labels = df.loc[mask, task].astype(int).values
    return texts, labels


def train_char_tfidf_lr(X_tr, y_tr, X_te) -> dict:
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        max_features=20000, sublinear_tf=True,
    )
    Xt = vec.fit_transform(X_tr); Xe = vec.transform(X_te)
    clf = LogisticRegression(C=1.0, max_iter=1000,
                             class_weight="balanced", random_state=42)
    clf.fit(Xt, y_tr); return {"y_pred": clf.predict(Xe).tolist()}


def train_char_tfidf_svm(X_tr, y_tr, X_te) -> dict:
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        max_features=20000, sublinear_tf=True,
    )
    Xt = vec.fit_transform(X_tr); Xe = vec.transform(X_te)
    clf = LinearSVC(C=1.0, class_weight="balanced",
                    random_state=42, max_iter=10000)
    clf.fit(Xt, y_tr); return {"y_pred": clf.predict(Xe).tolist()}


def run_char_baseline(texts, labels, task, model_key, predictor) -> dict:
    rpath = RESULTS_DIR / f"{task}_{model_key}_results.json"
    if rpath.exists():
        logger.info("%s / %s: cached", task, model_key)
        import json
        with open(rpath) as f: return json.load(f)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    for fold_idx, (tr, vl) in enumerate(skf.split(texts, labels)):
        X_tr = [texts[i] for i in tr]; y_tr = labels[tr]
        X_vl = [texts[i] for i in vl]; y_vl = labels[vl]
        out = predictor(X_tr, y_tr, X_vl)
        m = compute_metrics(y_vl.tolist(), out["y_pred"])
        m["fold"] = fold_idx
        fold_metrics.append(m)
    agg = aggregate_fold_metrics(fold_metrics)
    save_results(agg, rpath)
    logger.info("%s / %s acc=%.4f f1w=%.4f", task, model_key,
                agg["accuracy"]["mean"], agg["f1_weighted"]["mean"])
    return agg


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "processed" / "all_sessions.csv")
    for task in ["focusing_questions", "student_reasoning", "uptake"]:
        texts, labels = load_task_data(df, task)
        logger.info("=== %s : %d samples ===", task, len(labels))
        run_char_baseline(texts, labels, task, "tfidf_char_lr", train_char_tfidf_lr)
        run_char_baseline(texts, labels, task, "tfidf_char_svm", train_char_tfidf_svm)


if __name__ == "__main__":
    main()
