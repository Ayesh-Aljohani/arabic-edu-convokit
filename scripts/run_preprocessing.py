"""Phase 3: Preprocess all Arabic data — normalization, talk time, math density."""

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.normalize import normalize_arabic
from src.preprocessing.tokenize_ar import count_arabic_words
from src.preprocessing.math_lexicon import get_lexicon_size, count_math_terms, has_math_content
from src.features.talk_time import compute_talk_time, talk_time_by_speaker
from src.features.math_density import compute_math_density

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    translated_dir = ROOT / "data" / "translated"
    processed_dir = ROOT / "data" / "processed"
    results_dir = ROOT / "results" / "analysis"
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Verify math lexicon size
    lexicon_size = get_lexicon_size()
    logger.info("Math lexicon size: %d terms", lexicon_size)

    csv_files = sorted(translated_dir.glob("*.csv"), key=lambda p: int(p.stem))
    all_dfs = []

    start = time.time()
    for csv_file in csv_files:
        session_id = csv_file.stem
        logger.info("Processing session %s", session_id)

        df = pd.read_csv(csv_file)

        # Normalize Arabic text
        df["text_arabic_normalized"] = df["text_arabic"].apply(normalize_arabic)

        # Compute Arabic talk time
        df = compute_talk_time(df)

        # Compute Arabic math density
        df = compute_math_density(df)

        # Save processed file
        output_path = processed_dir / f"{session_id}.csv"
        df.to_csv(output_path, index=False)
        all_dfs.append(df)

    # Merge all sessions
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(processed_dir / "all_sessions.csv", index=False)

    elapsed = time.time() - start
    logger.info("Preprocessing completed in %.1f seconds", elapsed)
    logger.info("Total utterances: %d", len(combined))

    # Talk time analysis
    tt = talk_time_by_speaker(combined)
    logger.info("Talk time by speaker:\n%s", tt.to_string(index=False))
    tt.to_csv(results_dir / "talk_time_by_speaker.csv", index=False)

    # Math density summary
    math_summary = {
        "lexicon_size": lexicon_size,
        "total_utterances": len(combined),
        "utterances_with_math_arabic": int(combined["has_math_arabic"].sum()),
        "pct_with_math_arabic": float(combined["has_math_arabic"].mean()),
        "utterances_with_math_english": int((combined["math_density"] > 0).sum()),
        "pct_with_math_english": float((combined["math_density"] > 0).mean()),
        "mean_math_density_arabic": float(combined["math_density_arabic"].mean()),
        "mean_math_density_english": float(combined["math_density"].mean()),
    }
    with open(results_dir / "math_density_summary.json", "w") as f:
        json.dump(math_summary, f, indent=2)
    logger.info("Math density summary: %s", math_summary)

    # Dataset statistics
    stats = {
        "total_utterances": len(combined),
        "total_sessions": len(csv_files),
        "total_words_english": int(combined["talktime_words"].sum()),
        "total_words_arabic": int(combined["talktime_words_arabic"].sum()),
        "mean_words_english": float(combined["talktime_words"].mean()),
        "mean_words_arabic": float(combined["talktime_words_arabic"].mean()),
        "word_ratio_ar_en": float(
            combined["talktime_words_arabic"].sum() / combined["talktime_words"].sum()
        ),
        "speaker_counts": combined["speaker"].value_counts().to_dict(),
        "label_distributions": {},
    }
    for label in ["focusing_questions", "student_reasoning", "uptake"]:
        valid = combined[label].dropna()
        stats["label_distributions"][label] = {
            "labeled_count": len(valid),
            "positive_count": int(valid.sum()),
            "positive_rate": float(valid.mean()),
        }
    with open(results_dir / "dataset_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Dataset stats saved")


if __name__ == "__main__":
    main()
