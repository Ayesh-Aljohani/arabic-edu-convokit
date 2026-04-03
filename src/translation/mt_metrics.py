"""Compute MT quality metrics: BLEU, METEOR, chrF++, BERTScore."""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import sacrebleu
from bert_score import score as bertscore_compute
from nltk.translate.meteor_score import meteor_score as nltk_meteor
import nltk

logger = logging.getLogger(__name__)

# Ensure NLTK wordnet is available
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def compute_bleu(references: list[str], hypotheses: list[str]) -> dict:
    """Compute corpus-level BLEU using sacrebleu."""
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return {
        "score": bleu.score,
        "bp": bleu.bp,
        "precisions": bleu.precisions,
    }


def compute_chrf(references: list[str], hypotheses: list[str]) -> dict:
    """Compute corpus-level chrF++ using sacrebleu."""
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)
    return {"score": chrf.score}


def compute_meteor(references: list[str], hypotheses: list[str]) -> dict:
    """Compute average sentence-level METEOR."""
    scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = nltk.word_tokenize(ref)
        hyp_tokens = nltk.word_tokenize(hyp)
        scores.append(nltk_meteor([ref_tokens], hyp_tokens))
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "scores": [float(s) for s in scores],
    }


def compute_bertscore(
    references: list[str],
    hypotheses: list[str],
    model_type: str = "bert-base-multilingual-cased",
    device: str = "mps",
) -> dict:
    """Compute BERTScore for semantic similarity."""
    P, R, F1 = bertscore_compute(
        hypotheses,
        references,
        model_type=model_type,
        device=device,
        verbose=False,
    )
    return {
        "precision_mean": float(P.mean()),
        "recall_mean": float(R.mean()),
        "f1_mean": float(F1.mean()),
        "precision_std": float(P.std()),
        "recall_std": float(R.std()),
        "f1_std": float(F1.std()),
        "f1_scores": [float(f) for f in F1.tolist()],
    }


def compute_all_metrics(
    references: list[str],
    hypotheses: list[str],
    bertscore_device: str = "mps",
) -> dict:
    """Compute all MT quality metrics."""
    logger.info("Computing BLEU...")
    bleu = compute_bleu(references, hypotheses)

    logger.info("Computing chrF++...")
    chrf = compute_chrf(references, hypotheses)

    logger.info("Computing METEOR...")
    meteor = compute_meteor(references, hypotheses)

    logger.info("Computing BERTScore...")
    bert = compute_bertscore(
        references, hypotheses, device=bertscore_device
    )

    return {
        "bleu": bleu,
        "chrf_pp": chrf,
        "meteor": meteor,
        "bertscore": bert,
        "num_samples": len(references),
    }


def compute_metrics_per_session(
    back_translated_dir: str | Path,
    results_dir: str | Path,
    bertscore_device: str = "mps",
) -> dict:
    """Compute MT metrics per session and aggregate."""
    back_translated_dir = Path(back_translated_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(
        back_translated_dir.glob("*.csv"), key=lambda p: int(p.stem)
    )

    all_refs = []
    all_hyps = []
    session_metrics = {}

    for csv_file in csv_files:
        session_id = csv_file.stem
        logger.info("Computing metrics for session %s", session_id)

        df = pd.read_csv(csv_file)
        refs = df["text"].tolist()
        hyps = df["text_back_translated"].tolist()

        # Filter out empty pairs
        valid = [
            (r, h)
            for r, h in zip(refs, hyps)
            if isinstance(r, str) and isinstance(h, str) and r.strip() and h.strip()
        ]
        if not valid:
            logger.warning("Session %s: no valid pairs", session_id)
            continue

        refs_clean, hyps_clean = zip(*valid)
        refs_clean, hyps_clean = list(refs_clean), list(hyps_clean)

        all_refs.extend(refs_clean)
        all_hyps.extend(hyps_clean)

        # Per-session: BLEU and chrF only (BERTScore per-session is expensive)
        bleu = compute_bleu(refs_clean, hyps_clean)
        chrf = compute_chrf(refs_clean, hyps_clean)

        session_metrics[session_id] = {
            "bleu": bleu["score"],
            "chrf_pp": chrf["score"],
            "num_utterances": len(refs_clean),
        }

    # Aggregate metrics with BERTScore
    logger.info("Computing aggregate metrics over %d utterances", len(all_refs))
    aggregate = compute_all_metrics(all_refs, all_hyps, bertscore_device)
    aggregate["num_sessions"] = len(session_metrics)

    # Save results
    results = {
        "aggregate": aggregate,
        "per_session": session_metrics,
    }

    output_path = results_dir / "mt_metrics.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("MT metrics saved to %s", output_path)

    return results
