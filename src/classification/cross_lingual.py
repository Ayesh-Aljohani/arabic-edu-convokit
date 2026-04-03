"""Zero-shot cross-lingual transfer: train on English, test on Arabic."""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import logging

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.classification.dataset import UtteranceDataset
from src.classification.evaluate import compute_metrics
from src.classification.train import WeightedTrainer, compute_class_weights, set_seed, _hf_compute_metrics

logger = logging.getLogger(__name__)


def train_english_test_arabic(
    model_name: str,
    X_train_en: list[str],
    y_train: np.ndarray,
    X_test_ar: list[str],
    y_test: np.ndarray,
    output_dir: str,
    max_length: int = 128,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    num_epochs: int = 10,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    patience: int = 3,
) -> dict:
    """Train on English text, evaluate on Arabic text (zero-shot transfer)."""
    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    # Split English train into train/val (80/20)
    n = len(X_train_en)
    indices = np.random.RandomState(42).permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = indices[:split], indices[split:]

    X_tr = [X_train_en[i] for i in train_idx]
    y_tr = y_train[train_idx]
    X_vl = [X_train_en[i] for i in val_idx]
    y_vl = y_train[val_idx]

    train_enc = tokenizer(
        X_tr, truncation=True, padding=True, max_length=max_length,
        return_tensors="pt",
    )
    val_enc = tokenizer(
        X_vl, truncation=True, padding=True, max_length=max_length,
        return_tensors="pt",
    )
    test_enc = tokenizer(
        X_test_ar, truncation=True, padding=True, max_length=max_length,
        return_tensors="pt",
    )

    train_dataset = UtteranceDataset(train_enc, y_tr.tolist())
    val_dataset = UtteranceDataset(val_enc, y_vl.tolist())
    test_dataset = UtteranceDataset(test_enc, y_test.tolist())

    class_weights = compute_class_weights(y_tr)

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
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_hf_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    trainer.train()

    # English validation results
    en_preds = trainer.predict(val_dataset)
    en_y_pred = np.argmax(en_preds.predictions, axis=-1)
    en_metrics = compute_metrics(y_vl.tolist(), en_y_pred.tolist())

    # Arabic test results (zero-shot)
    ar_preds = trainer.predict(test_dataset)
    ar_y_pred = np.argmax(ar_preds.predictions, axis=-1)
    ar_metrics = compute_metrics(y_test.tolist(), ar_y_pred.tolist())
    ar_y_proba = torch.softmax(
        torch.tensor(ar_preds.predictions, dtype=torch.float32), dim=-1
    )[:, 1].numpy()

    ar_metrics["y_pred"] = ar_y_pred.tolist()
    ar_metrics["y_true"] = y_test.tolist()
    ar_metrics["y_proba"] = ar_y_proba.tolist()

    del model, trainer
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "english_validation": en_metrics,
        "arabic_zero_shot": ar_metrics,
    }
