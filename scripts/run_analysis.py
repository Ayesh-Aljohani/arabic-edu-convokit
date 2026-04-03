"""Phase 5: Cross-linguistic analysis, error analysis, figures, and tables."""

import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.cross_linguistic import run_cross_linguistic_analysis
from src.analysis.error_analysis import run_error_analysis
from src.visualization.plots import plot_all
from src.visualization.tables import generate_all_tables

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    start = time.time()

    logger.info("Step 1: Cross-linguistic analysis")
    run_cross_linguistic_analysis(ROOT)

    logger.info("Step 2: Error analysis")
    run_error_analysis(ROOT)

    logger.info("Step 3: Generating figures")
    plot_all(ROOT)

    logger.info("Step 4: Generating LaTeX tables")
    generate_all_tables(ROOT)

    elapsed = time.time() - start
    logger.info("Phase 5 completed in %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
