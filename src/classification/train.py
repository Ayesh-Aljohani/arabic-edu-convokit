"""Training loop with 5-fold stratified cross-validation for transformer models."""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import logging
import random

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.classification.dataset import UtteranceDataset
from src.classification.evaluate import compute_metrics as eval_metrics

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Compute inverse-frequency class weights."""
    classes, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = total / (len(classes) * counts)
    return torch.tensor(weights, dtype=torch.float32)


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy loss."""

    def __init__(self, class_weights: torch.Tensor = None, **kwargs):
        super().__init__(**kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self._class_weights is not None:
            weight = self._class_weights.to(logits.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def _hf_compute_metrics(eval_pred):
    """Metric function for HuggingFace Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    from sklearn.metrics import f1_score
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": acc, "f1_weighted": f1}


def train_transformer_fold(
    model_name: str,
    X_train: list[str],
    y_train: np.ndarray,
    X_val: list[str],
    y_val: np.ndarray,
    output_dir: str,
    max_length: int = 128,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    num_epochs: int = 10,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    patience: int = 3,
) -> dict:
    """Train a transformer model on one fold and return predictions."""
    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    train_enc = tokenizer(
        X_train, truncation=True, padding=True, max_length=max_length,
        return_tensors="pt",
    )
    val_enc = tokenizer(
        X_val, truncation=True, padding=True, max_length=max_length,
        return_tensors="pt",
    )

    train_dataset = UtteranceDataset(train_enc, y_train.tolist())
    val_dataset = UtteranceDataset(val_enc, y_val.tolist())

    class_weights = compute_class_weights(y_train)

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

    # Get predictions
    preds_output = trainer.predict(val_dataset)
    y_pred = np.argmax(preds_output.predictions, axis=-1)
    y_proba = torch.softmax(
        torch.tensor(preds_output.predictions, dtype=torch.float32), dim=-1
    )[:, 1].numpy()

    metrics = eval_metrics(y_val.tolist(), y_pred.tolist())
    metrics["y_pred"] = y_pred.tolist()
    metrics["y_true"] = y_val.tolist()
    metrics["y_proba"] = y_proba.tolist()

    # Cleanup
    del model, trainer
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return metrics


def run_cv_transformer(
    model_name: str,
    texts: list[str],
    labels: np.ndarray,
    task_name: str,
    n_folds: int = 5,
    output_base: str = "results/classification",
    **train_kwargs,
) -> list[dict]:
    """Run stratified k-fold CV for a transformer model."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    short_name = model_name.split("/")[-1]
    logger.info("Running %d-fold CV for %s on %s", n_folds, short_name, task_name)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        logger.info("Fold %d/%d", fold_idx + 1, n_folds)

        X_train = [texts[i] for i in train_idx]
        y_train = labels[train_idx]
        X_val = [texts[i] for i in val_idx]
        y_val = labels[val_idx]

        fold_dir = f"{output_base}/{task_name}/{short_name}/fold_{fold_idx}"

        metrics = train_transformer_fold(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            output_dir=fold_dir,
            **train_kwargs,
        )
        metrics["fold"] = fold_idx

        fold_results.append(metrics)
        logger.info(
            "Fold %d: acc=%.4f  f1=%.4f",
            fold_idx, metrics["accuracy"], metrics["f1_weighted"],
        )

    return fold_results
