"""Baseline classifiers: TF-IDF + LR, TF-IDF + SVM."""

import logging

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)


def train_dummy(y_train: np.ndarray, y_test: np.ndarray, strategy: str) -> dict:
    """Train a dummy baseline classifier."""
    clf = DummyClassifier(strategy=strategy, random_state=42)
    clf.fit(np.zeros((len(y_train), 1)), y_train)
    y_pred = clf.predict(np.zeros((len(y_test), 1)))
    return {"y_pred": y_pred.tolist(), "model_name": f"dummy_{strategy}"}


def train_tfidf_lr(
    X_train: list[str],
    y_train: np.ndarray,
    X_test: list[str],
    max_features: int = 10000,
    ngram_range: tuple = (1, 2),
    C: float = 1.0,
    max_iter: int = 1000,
) -> dict:
    """Train TF-IDF + Logistic Regression."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Use class weights for imbalance
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)

    return {"y_pred": y_pred.tolist(), "model_name": "tfidf_lr"}


def train_tfidf_svm(
    X_train: list[str],
    y_train: np.ndarray,
    X_test: list[str],
    max_features: int = 10000,
    ngram_range: tuple = (1, 2),
    C: float = 1.0,
) -> dict:
    """Train TF-IDF + Linear SVM."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    clf = LinearSVC(
        C=C,
        class_weight="balanced",
        random_state=42,
        max_iter=10000,
    )
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)

    return {"y_pred": y_pred.tolist(), "model_name": "tfidf_svm"}
