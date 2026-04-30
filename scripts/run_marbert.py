"""Run MARBERTv2 baseline on all three classification tasks (5-fold stratified CV).

Adds MARBERT as a fourth transformer baseline alongside AraBERT, mBERT, XLM-R, in
response to reviewer R2-W1 / EIC-W3 / Devil's-Advocate-M1.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.classification.evaluate import aggregate_fold_metrics, save_results
from src.classification.train import run_cv_transformer, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


MODEL_NAME = "UBC-NLP/MARBERTv2"
MODEL_KEY = "marbert"
RESULTS_DIR = ROOT / "results" / "classification"


def load_task_data(df: pd.DataFrame, task: str) -> tuple[list[str], np.ndarray]:
    mask = df[task].notna()
    texts = df.loc[mask, "text_arabic_normalized"].tolist()
    labels = df.loc[mask, task].astype(int).values
    return texts, labels


def main() -> None:
    set_seed(42)

    df = pd.read_csv(ROOT / "data" / "processed" / "all_sessions.csv")
    logger.info("Loaded %d utterances", len(df))

    tasks = ["focusing_questions", "student_reasoning", "uptake"]

    for task in tasks:
        rpath = RESULTS_DIR / f"{task}_{MODEL_KEY}_results.json"
        if rpath.exists():
            logger.info("%s: cached, skipping", task)
            continue

        texts, labels = load_task_data(df, task)
        logger.info(
            "Task %s: %d samples, %.1f%% positive",
            task, len(labels), 100 * labels.mean(),
        )

        start = time.time()
        folds = run_cv_transformer(
            model_name=MODEL_NAME,
            texts=texts,
            labels=labels,
            task_name=f"{task}_marbert",
            n_folds=5,
            output_base=str(RESULTS_DIR),
            max_length=128,
            learning_rate=2e-5,
            batch_size=32,
            num_epochs=10,
            weight_decay=0.01,
            warmup_ratio=0.1,
            patience=3,
        )
        aggregated = aggregate_fold_metrics(folds)
        save_results(aggregated, rpath)

        elapsed = time.time() - start
        logger.info(
            "%s / %s done in %.1fs: acc=%.4f f1w=%.4f f1+=%.4f",
            task, MODEL_KEY, elapsed,
            aggregated["accuracy"]["mean"],
            aggregated["f1_weighted"]["mean"],
            aggregated["f1_positive"]["mean"],
        )


if __name__ == "__main__":
    main()
