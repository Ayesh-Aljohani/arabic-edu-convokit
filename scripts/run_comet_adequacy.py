"""Forward-direction reference-free MT quality estimation.

We use LaBSE (Language-agnostic BERT Sentence Embedding) cosine similarity
between the English source and the Arabic translation as a reference-free
adequacy proxy. LaBSE produces a single shared semantic space across 109
languages and its cross-lingual cosine has been shown to correlate with
direct-assessment scores comparably to BERTScore in published QE studies
(Feng et al. 2022, "Language-agnostic BERT Sentence Embedding"). It is fully
open and does not require gated-model access.

We compute on a stratified sample of N=500 utterances, drawn proportional to
session, with seed=42 for reproducibility.

This addresses reviewer R1-W1 / EIC-W1 (round-trip BLEU does not measure
forward-direction adequacy).
"""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

SAMPLE_SIZE = 500
SEED = 42
DATA_PATH = ROOT / "data" / "processed" / "all_sessions_with_sid.csv"
OUT_PATH = ROOT / "results" / "translation" / "labse_qe.json"


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    df = pd.read_csv(DATA_PATH)
    logger.info("Loaded %d utterances", len(df))

    # Drop empty translations
    df = df.dropna(subset=["text", "text_arabic"])
    df = df[df["text"].str.strip().astype(bool) & df["text_arabic"].str.strip().astype(bool)]
    logger.info("After dropping empties: %d", len(df))

    # Stratified-by-session sample of size N (build by sampling per session)
    parts = []
    for sid, g in df.groupby("session_id"):
        n_take = max(1, int(round(SAMPLE_SIZE * len(g) / len(df))))
        parts.append(g.sample(n=min(n_take, len(g)), random_state=SEED))
    sample = pd.concat(parts, ignore_index=True)
    if len(sample) > SAMPLE_SIZE:
        sample = sample.sample(n=SAMPLE_SIZE, random_state=SEED).reset_index(drop=True)
    logger.info("Sampled %d utterances across %d sessions",
                len(sample), sample["session_id"].nunique())

    # Load LaBSE
    from transformers import AutoTokenizer, AutoModel
    name = "sentence-transformers/LaBSE"
    logger.info("Loading %s", name)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device).eval()

    def embed(texts, batch_size=32):
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=256, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**enc)
            # LaBSE uses pooler_output
            emb = out.pooler_output
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            embs.append(emb.cpu())
        return torch.cat(embs, dim=0)

    en_texts = sample["text"].tolist()
    ar_texts = sample["text_arabic"].tolist()
    logger.info("Embedding English (%d) ...", len(en_texts))
    en_emb = embed(en_texts)
    logger.info("Embedding Arabic (%d) ...", len(ar_texts))
    ar_emb = embed(ar_texts)

    cos = (en_emb * ar_emb).sum(dim=1).numpy()
    arr = np.asarray(cos)

    # Quality bins per Feng et al. 2022 / Tatoeba conventions
    bins = {
        "high (>=0.80)": int((arr >= 0.80).sum()),
        "medium [0.60, 0.80)": int(((arr >= 0.60) & (arr < 0.80)).sum()),
        "low (<0.60)": int((arr < 0.60).sum()),
    }

    summary = {
        "model": "sentence-transformers/LaBSE",
        "metric": "LaBSE cosine similarity (reference-free QE proxy)",
        "rationale": (
            "LaBSE produces 109-language shared sentence embeddings; cross-"
            "lingual cosine has been shown to correlate with human direct-"
            "assessment ratings of MT adequacy (Feng et al., ACL 2022)."
        ),
        "n": int(len(arr)),
        "n_sessions": int(sample["session_id"].nunique()),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "ci_95_low": float(arr.mean() - 1.96 * arr.std(ddof=1) / np.sqrt(len(arr))),
        "ci_95_high": float(arr.mean() + 1.96 * arr.std(ddof=1) / np.sqrt(len(arr))),
        "quality_bins": bins,
        "scores": cos.tolist(),
        "sample_seed": SEED,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        "LaBSE cosine: mean=%.4f sd=%.4f median=%.4f  bins: %s",
        summary["mean"], summary["std"], summary["median"], bins,
    )
    logger.info("Saved %s", OUT_PATH)


if __name__ == "__main__":
    main()
