"""Session-grouped 5-fold cross-validation for transformer models.

Re-runs transformers using GroupKFold(session_id) instead of StratifiedKFold
to estimate session-level leakage in the original instance-stratified splits.
Reports per-fold accuracy / F1 deltas (R5 / R1-W3 in the review).

Focused configuration: all three encoders on focusing-questions (most
leakage-prone, largest sample), plus the best-of-task model on SR and UP.
This keeps total compute under ~2 hours on M4 Max with MPS.
"""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.classification.evaluate import aggregate_fold_metrics, save_results
from src.classification.train import set_seed, train_transformer_fold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT / "results" / "classification"
DATA_PATH = ROOT / "data" / "processed" / "all_sessions_with_sid.csv"

# (task, list of (model_key, model_name)) — focused to keep within ~2h compute
RUN_CONFIG = [
    ("focusing_questions", [
        ("arabert_grouped", "aubmindlab/bert-base-arabertv2"),
        ("mbert_grouped", "bert-base-multilingual-cased"),
        ("xlmr_grouped", "xlm-roberta-base"),
    ]),
    ("student_reasoning", [
        ("mbert_grouped", "bert-base-multilingual-cased"),
    ]),
    ("uptake", [
        ("xlmr_grouped", "xlm-roberta-base"),
    ]),
]


def load_task(df, task):
    mask = df[task].notna()
    texts = df.loc[mask, "text_arabic_normalized"].tolist()
    labels = df.loc[mask, task].astype(int).values
    sids = df.loc[mask, "session_id"].values
    return texts, labels, sids


def run_grouped(model_name, model_key, texts, labels, sids, task):
    rpath = RESULTS_DIR / f"{task}_{model_key}_results.json"
    if rpath.exists():
        logger.info("%s / %s: cached, skipping", task, model_key)
        return
    gkf = GroupKFold(n_splits=5)
    fold_metrics = []
    short = model_name.split("/")[-1]
    for fold_idx, (tr, vl) in enumerate(gkf.split(texts, labels, groups=sids)):
        X_tr = [texts[i] for i in tr]; y_tr = labels[tr]
        X_vl = [texts[i] for i in vl]; y_vl = labels[vl]
        n_train_sessions = len(np.unique(sids[tr]))
        n_val_sessions = len(np.unique(sids[vl]))
        logger.info(
            "Fold %d/5: train=%d (%d sessions) val=%d (%d sessions)",
            fold_idx + 1, len(y_tr), n_train_sessions, len(y_vl), n_val_sessions,
        )
        out_dir = f"{RESULTS_DIR}/{task}_grouped/{short}/fold_{fold_idx}"
        m = train_transformer_fold(
            model_name=model_name,
            X_train=X_tr, y_train=y_tr,
            X_val=X_vl, y_val=y_vl,
            output_dir=out_dir,
            max_length=128, learning_rate=2e-5,
            batch_size=32, num_epochs=10,
            weight_decay=0.01, warmup_ratio=0.1, patience=3,
        )
        m["fold"] = fold_idx
        m["n_train_sessions"] = int(n_train_sessions)
        m["n_val_sessions"] = int(n_val_sessions)
        fold_metrics.append(m)
    agg = aggregate_fold_metrics(fold_metrics)
    save_results(agg, rpath)
    logger.info(
        "%s / %s done: acc=%.4f±%.4f  f1w=%.4f±%.4f  f1+=%.4f±%.4f",
        task, model_key,
        agg["accuracy"]["mean"], agg["accuracy"]["std"],
        agg["f1_weighted"]["mean"], agg["f1_weighted"]["std"],
        agg["f1_positive"]["mean"], agg["f1_positive"]["std"],
    )


def main():
    set_seed(42)
    df = pd.read_csv(DATA_PATH)
    logger.info("Loaded %d rows, %d sessions", len(df), df["session_id"].nunique())
    for task, model_list in RUN_CONFIG:
        logger.info("=== Task: %s ===", task)
        texts, labels, sids = load_task(df, task)
        logger.info(
            "%d samples (%.1f%% positive) across %d sessions",
            len(labels), 100 * labels.mean(), len(np.unique(sids)),
        )
        for model_key, model_name in model_list:
            t0 = time.time()
            run_grouped(model_name, model_key, texts, labels, sids, task)
            logger.info("%s / %s elapsed: %.1fs", task, model_key, time.time() - t0)


if __name__ == "__main__":
    main()
