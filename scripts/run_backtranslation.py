"""Phase 2: Back-translate Arabic to English and compute MT quality metrics."""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import logging
import sys
import time
from pathlib import Path

import yaml

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.translation.back_translate import back_translate_all_sessions
from src.translation.mt_metrics import compute_metrics_per_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main() -> None:
    with open(ROOT / "config" / "config.yaml") as f:
        config = yaml.safe_load(f)

    device = config["project"]["device"]
    translated_dir = ROOT / config["paths"]["translated_data"]
    back_translated_dir = ROOT / "data" / "back_translated"
    results_dir = ROOT / config["paths"]["results"] / "translation"

    # Step 1: Back-translate Arabic -> English
    logger.info("Step 1: Back-translating Arabic to English")
    start = time.time()

    back_translate_all_sessions(
        translated_dir=translated_dir,
        output_dir=back_translated_dir,
        model_name=config["models"]["translation"]["name"],
        batch_size=config["models"]["translation"]["batch_size"],
        device=device,
    )

    bt_time = time.time() - start
    logger.info("Back-translation completed in %.1f seconds", bt_time)

    # Step 2: Compute MT quality metrics
    logger.info("Step 2: Computing MT quality metrics")
    start = time.time()

    results = compute_metrics_per_session(
        back_translated_dir=back_translated_dir,
        results_dir=results_dir,
        bertscore_device=device,
    )

    metrics_time = time.time() - start
    logger.info("MT metrics computed in %.1f seconds", metrics_time)

    # Print summary
    agg = results["aggregate"]
    logger.info("--- Aggregate MT Metrics ---")
    logger.info("BLEU:       %.2f", agg["bleu"]["score"])
    logger.info("chrF++:     %.2f", agg["chrf_pp"]["score"])
    logger.info("METEOR:     %.4f", agg["meteor"]["mean"])
    logger.info("BERTScore:  P=%.4f  R=%.4f  F1=%.4f",
                agg["bertscore"]["precision_mean"],
                agg["bertscore"]["recall_mean"],
                agg["bertscore"]["f1_mean"])
    logger.info("Total utterances: %d", agg["num_samples"])
    logger.info("Total sessions:   %d", agg["num_sessions"])
    logger.info("Total time: %.1f seconds", bt_time + metrics_time)


if __name__ == "__main__":
    main()
