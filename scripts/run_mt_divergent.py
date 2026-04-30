"""Independent MT system (Helsinki-NLP/opus-mt-en-ar) translation +
cross-lingual classifier robustness re-evaluation.

This version retrains the cross-lingual classifiers in-process (the original
checkpoints from run_experiments.py were auto-cleaned by save_total_limit=1)
and then evaluates on both:
  (a) the NLLB-200 Arabic test set (sanity check vs Table 6),
  (b) the Helsinki MarianMT Arabic translations of the same labeled subset.

Addresses reviewer R1-W2 / EIC-W2 / Devil's-Advocate-C1.
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
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    MarianMTModel,
    MarianTokenizer,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.classification.dataset import UtteranceDataset
from src.classification.evaluate import compute_metrics, save_results
from src.classification.train import (
    WeightedTrainer,
    _hf_compute_metrics,
    compute_class_weights,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DATA_PATH = ROOT / "data" / "processed" / "all_sessions.csv"
PARTIAL = ROOT / "data" / "processed" / "marian_ar_partial.csv"
OUT_TRANSLATIONS = ROOT / "data" / "processed" / "marian_ar_labeled.csv"
OUT_RESULTS = ROOT / "results" / "classification" / "mt_divergent_results.json"


def translate_with_marian(texts, indices, batch_size=32):
    """Translate English texts to Arabic with Helsinki MarianMT, resumable."""
    name = "Helsinki-NLP/opus-mt-en-ar"
    logger.info("Loading %s", name)
    tok = MarianTokenizer.from_pretrained(name)
    model = MarianMTModel.from_pretrained(name)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device).eval()

    out_map = {}
    if PARTIAL.exists():
        prev = pd.read_csv(PARTIAL)
        if "orig_index" in prev.columns:
            for _, r in prev.iterrows():
                out_map[int(r["orig_index"])] = r["text_arabic_marian"]
            logger.info("Resuming: %d already translated", len(out_map))

    pending = [(idx, t) for idx, t in zip(indices, texts) if int(idx) not in out_map]
    n = len(pending)
    if n == 0:
        logger.info("All translations cached")
        return [out_map[int(idx)] for idx in indices]

    logger.info("To translate: %d / %d", n, len(texts))
    t0 = time.time()
    for i in range(0, n, batch_size):
        batch_pairs = pending[i:i+batch_size]
        batch_texts = [t if isinstance(t, str) and t.strip() else "." for _, t in batch_pairs]
        enc = tok(batch_texts, return_tensors="pt", padding=True,
                  truncation=True, max_length=256).to(device)
        with torch.no_grad():
            ids = model.generate(**enc, num_beams=4, max_length=256)
        decoded = tok.batch_decode(ids, skip_special_tokens=True)
        for (idx, _), d in zip(batch_pairs, decoded):
            out_map[int(idx)] = d
        if (i // batch_size) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + len(batch_pairs)) / max(elapsed, 1)
            eta = (n - (i + len(batch_pairs))) / max(rate, 0.001)
            logger.info(
                "Translated %d/%d (rate %.1f/s, ETA %.1fmin)",
                i + len(batch_pairs), n, rate, eta / 60,
            )
            partial_df = pd.DataFrame([
                {"orig_index": k, "text_arabic_marian": v}
                for k, v in out_map.items()
            ])
            partial_df.to_csv(PARTIAL, index=False)

    partial_df = pd.DataFrame([
        {"orig_index": k, "text_arabic_marian": v}
        for k, v in out_map.items()
    ])
    partial_df.to_csv(PARTIAL, index=False)
    return [out_map.get(int(idx), "") for idx in indices]


def train_cl_classifier(
    model_name, X_en_train, y_en_train, X_en_val, y_en_val,
    output_dir, max_length=128, learning_rate=2e-5, batch_size=32,
    num_epochs=10, weight_decay=0.01, warmup_ratio=0.1, patience=3,
):
    """Train a transformer on English, return the trainer (with model in memory)."""
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2,
    )

    train_enc = tokenizer(X_en_train, truncation=True, padding=True,
                          max_length=max_length, return_tensors="pt")
    val_enc = tokenizer(X_en_val, truncation=True, padding=True,
                        max_length=max_length, return_tensors="pt")
    train_ds = UtteranceDataset(train_enc, y_en_train.tolist())
    val_ds = UtteranceDataset(val_enc, y_en_val.tolist())

    cw = compute_class_weights(y_en_train)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        dataloader_num_workers=0,
        report_to="none",
        seed=42,
    )
    trainer = WeightedTrainer(
        class_weights=cw,
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=_hf_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )
    trainer.train()
    return trainer, tokenizer


def predict_metrics(trainer, tokenizer, X, y, max_length=128):
    # Sanitise inputs: ensure all elements are strings
    X = [s if isinstance(s, str) and s.strip() else "." for s in X]
    enc = tokenizer(X, truncation=True, padding=True,
                    max_length=max_length, return_tensors="pt")
    ds = UtteranceDataset(enc, y.tolist())
    pred = trainer.predict(ds)
    y_pred = np.argmax(pred.predictions, axis=-1)
    return compute_metrics(y.tolist(), y_pred.tolist())


def load_task_data_bilingual(df, task):
    mask = df[task].notna()
    en = df.loc[mask, "text"].tolist()
    ar_nllb = df.loc[mask, "text_arabic_normalized"].tolist()
    y = df.loc[mask, task].astype(int).values
    return en, ar_nllb, y, mask


def main():
    set_seed(42)
    df = pd.read_csv(DATA_PATH)
    logger.info("Loaded %d utterances", len(df))

    # Step 1: translate the labeled subset with MarianMT
    labeled_mask = (
        df["focusing_questions"].notna()
        | df["student_reasoning"].notna()
        | df["uptake"].notna()
    )
    labeled_df = df[labeled_mask].copy()
    labeled_df["orig_index"] = labeled_df.index
    logger.info("Labeled subset: %d utterances", len(labeled_df))

    if OUT_TRANSLATIONS.exists():
        logger.info("Loaded cached translations from %s", OUT_TRANSLATIONS)
        df_tr = pd.read_csv(OUT_TRANSLATIONS)
        marian_map = dict(zip(df_tr["orig_index"], df_tr["text_arabic_marian"]))
    else:
        translations = translate_with_marian(
            labeled_df["text"].fillna("").tolist(),
            labeled_df["orig_index"].tolist(),
            batch_size=32,
        )
        labeled_df["text_arabic_marian"] = translations
        labeled_df.to_csv(OUT_TRANSLATIONS, index=False)
        marian_map = dict(zip(labeled_df["orig_index"], translations))

    # Step 2: train cross-lingual classifier on English; evaluate on NLLB-AR and Marian-AR
    results = {}
    base_models = [
        ("mbert", "bert-base-multilingual-cased"),
        ("xlmr", "xlm-roberta-base"),
    ]

    for task in ["focusing_questions", "student_reasoning", "uptake"]:
        results[task] = {}
        en_texts, ar_nllb_texts, y_all, mask = load_task_data_bilingual(df, task)
        # Build Marian translations aligned to the same labeled mask
        idxs = df.index[mask].tolist()
        ar_marian_texts = [marian_map.get(int(i), "") for i in idxs]
        logger.info("Task %s: %d samples", task, len(y_all))

        # 80/20 English split
        n = len(y_all)
        rng = np.random.RandomState(42)
        perm = rng.permutation(n)
        split = int(0.8 * n)
        tr_idx, vl_idx = perm[:split], perm[split:]
        X_tr = [en_texts[i] for i in tr_idx]; y_tr = y_all[tr_idx]
        X_vl = [en_texts[i] for i in vl_idx]; y_vl = y_all[vl_idx]

        for model_key, model_name in base_models:
            t0 = time.time()
            output_dir = str(ROOT / "results" / "classification" / f"{task}_mt_div_{model_key}")
            logger.info("Training %s for cross-lingual %s", model_key, task)
            trainer, tokenizer = train_cl_classifier(
                model_name, X_tr, y_tr, X_vl, y_vl, output_dir,
                num_epochs=5,  # reduced for speed; original used 10 with early stopping
            )

            en_val_metrics = predict_metrics(trainer, tokenizer, X_vl, y_vl)
            nllb_metrics = predict_metrics(trainer, tokenizer, ar_nllb_texts, y_all)
            marian_metrics = predict_metrics(trainer, tokenizer, ar_marian_texts, y_all)

            results[task][model_key] = {
                "english_validation": en_val_metrics,
                "nllb_zero_shot": nllb_metrics,
                "marian_zero_shot": marian_metrics,
                "delta_acc_marian_minus_nllb": marian_metrics["accuracy"] - nllb_metrics["accuracy"],
                "delta_f1w_marian_minus_nllb": marian_metrics["f1_weighted"] - nllb_metrics["f1_weighted"],
            }
            logger.info(
                "%s/%s NLLB acc=%.4f Marian acc=%.4f delta=%+.4f (took %.1fs)",
                task, model_key,
                nllb_metrics["accuracy"], marian_metrics["accuracy"],
                results[task][model_key]["delta_acc_marian_minus_nllb"],
                time.time() - t0,
            )

            del trainer
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    save_results(results, OUT_RESULTS)
    logger.info("Saved %s", OUT_RESULTS)


if __name__ == "__main__":
    main()
