"""
Model training, evaluation and prediction module for Tennis Match Prediction Model.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from joblib import dump, load

from .config import config, logger
from .exceptions import (
    ModelTrainingError,
    ModelEvaluationError,
    ModelSaveError,
    ModelLoadError,
    PredictionError,
    InvalidFeatureError,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def chronological_split(
    df_features: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data chronologically: Train up to TEST_START_YEAR-1, Test from TEST_START_YEAR."""
    logger.info("Splitting data chronologically")

    X = df_features[feature_cols].copy()
    y = df_features["target"].copy()
    years = df_features["year"].copy()

    train_mask = years < config.TEST_START_YEAR
    test_mask = years >= config.TEST_START_YEAR

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    logger.info(
        f"Train set: {len(X_train)} samples ({config.TRAIN_START_YEAR}-{config.TRAIN_END_YEAR})"
    )
    logger.info(
        f"Test set: {len(X_test)} samples ({config.TEST_START_YEAR}-{config.TEST_END_YEAR})"
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LGBMClassifier:
    """Train LightGBM model with configured hyperparameters."""
    logger.info("Training LightGBM model")

    try:
        model = LGBMClassifier(**config.MODEL_PARAMS)
        model.fit(X_train, y_train)
        logger.info("Model training complete")
        return model
    except Exception as e:
        raise ModelTrainingError(f"Failed to train model: {e}") from e


def evaluate_model(
    model: LGBMClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_cols: list[str],
) -> dict[str, float | pd.DataFrame]:
    """Comprehensive model evaluation."""
    logger.info("Evaluating model")

    try:
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        proba = model.predict_proba(X_test)
        y_test_proba = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]

        train_acc = accuracy_score(y_train, y_train_pred)
        train_prec = precision_score(y_train, y_train_pred, zero_division=0)
        train_rec = recall_score(y_train, y_train_pred, zero_division=0)
        train_f1 = f1_score(y_train, y_train_pred, zero_division=0)

        test_acc = accuracy_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred, zero_division=0)
        test_rec = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
        test_roc = roc_auc_score(y_test, y_test_proba)

        # Log training performance
        logger.info("TRAINING SET PERFORMANCE:")
        logger.info(f"  Accuracy:  {train_acc:.4f}")
        logger.info(f"  Precision: {train_prec:.4f}")
        logger.info(f"  Recall:    {train_rec:.4f}")
        logger.info(f"  F1-Score:  {train_f1:.4f}")

        # Log test performance
        logger.info(f"\nTEST SET PERFORMANCE ({config.TEST_START_YEAR}-{config.TEST_END_YEAR}):")
        logger.info(f"  Accuracy:  {test_acc:.4f}")
        logger.info(f"  Precision: {test_prec:.4f}")
        logger.info(f"  Recall:    {test_rec:.4f}")
        logger.info(f"  F1-Score:  {test_f1:.4f}")
        logger.info(f"  ROC-AUC:   {test_roc:.4f}")

        # Log confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        logger.info("CONFUSION MATRIX (Test Set):")
        logger.info(f"  TN: {cm[0, 0]}, FP: {cm[0, 1]}, FN: {cm[1, 0]}, TP: {cm[1, 1]}")

        # Cross-validation evaluation
        logger.info(f"\nCROSS-VALIDATION ({config.CV_FOLDS}-Fold Stratified):")
        skf = StratifiedKFold(
            n_splits=config.CV_FOLDS, shuffle=True, random_state=config.CV_RANDOM_STATE
        )
        cv_acc = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")
        cv_auc = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc")
        logger.info(f"  CV Accuracy: {cv_acc.mean():.4f} (+/- {cv_acc.std():.4f})")
        logger.info(f"  CV ROC-AUC:  {cv_auc.mean():.4f} (+/- {cv_auc.std():.4f})")

        # Feature importance
        importance = pd.DataFrame(
            {"feature": feature_cols, "importance": model.feature_importances_}
        )
        importance = importance.sort_values("importance", ascending=False)
        logger.info("TOP 10 FEATURES:")
        for _, row in importance.head(10).iterrows():
            logger.info(f"  {row['feature']:25s}: {row['importance']:.4f}")

        return {
            "test_accuracy": test_acc,
            "test_precision": test_prec,
            "test_recall": test_rec,
            "test_f1": test_f1,
            "test_roc_auc": test_roc,
            "feature_importance": importance,
        }

    except Exception as e:
        raise ModelEvaluationError(f"Failed to evaluate model: {e}") from e


def save_model(model: LGBMClassifier, path: str | None = None) -> None:
    """Save trained model to disk using joblib."""
    path = path or config.MODEL_PATH
    logger.info(f"Saving model to {path}")

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dump(model, path)
        logger.info("Model saved successfully")
    except Exception as e:
        raise ModelSaveError(f"Failed to save model to {path}: {e}") from e


def load_model(path: str | None = None) -> LGBMClassifier:
    """Load trained model from disk using joblib."""
    path = path or config.MODEL_PATH
    logger.info(f"Loading model from {path}")

    if not os.path.exists(path):
        raise ModelLoadError(f"Model not found at {path}")

    try:
        model = load(path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        raise ModelLoadError(f"Failed to load model from {path}: {e}") from e


def predict_match(
    model: LGBMClassifier,
    match_features: dict[str, float],
    feature_cols: list[str],
) -> tuple[int, float]:
    """Predict winner for a new match."""
    try:
        feature_array = np.array([match_features[f] for f in feature_cols]).reshape(
            1, -1
        )
        prediction = model.predict(feature_array)[0]
        probability = model.predict_proba(feature_array)[0][1]
        return int(prediction), float(probability)
    except KeyError as e:
        raise InvalidFeatureError(f"Missing feature: {e}") from e
    except Exception as e:
        raise PredictionError(f"Failed to make prediction: {e}") from e
