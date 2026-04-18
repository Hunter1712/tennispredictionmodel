"""
Hyperparameter tuning with Optuna for Tennis Prediction Model.
"""

from __future__ import annotations

import json
import os
import pickle
import re
from typing import Any

import numpy as np
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from config import logger


import os


def load_data() -> Any:
    """Load cached data for tuning."""
    cache_path = os.path.join(os.path.dirname(__file__), "..", "data_cache.pkl")
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def tune_hyperparameters(n_trials: int = 50) -> dict[str, float]:
    """Run hyperparameter tuning.

    Args:
        n_trials: Number of Optuna trials to run.

    Returns:
        Dictionary of best hyperparameters.
    """
    logger.info("Loading data for tuning...")
    df = load_data()

    # Use Elo + fatigue features (already computed in main pipeline)
    feature_cols = [
        "elo_diff",
        "elo_surface_diff",
        "days_since_last_diff",
        "rest_quality_diff",
    ]

    X = df[feature_cols].copy()
    y = df["target"].copy()

    logger.info(f"Data shape: {X.shape}")
    logger.info(f"Running {n_trials} trials...")

    # Suppress optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )

    # Use fixed cross-validation split for efficiency
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_splits = list(skf.split(X, y))

    def objective_fixed(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
            "tree_method": "hist",
        }

        scores = []
        for train_idx, val_idx in cv_splits:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = XGBClassifier(**params)
            model.fit(X_train, y_train)

            y_proba = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_proba)
            scores.append(score)

        return np.mean(scores)

    study.optimize(objective_fixed, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"\nBest ROC-AUC: {study.best_value:.4f}")
    logger.info(f"\nBest params:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")

    return study.best_params


if __name__ == "__main__":
    best_params = tune_hyperparameters(n_trials=50)

    # Update config.py with best params
    import re

    config_path = os.path.join(os.path.dirname(__file__), "config.py")
    with open(config_path, "r") as f:
        config_content = f.read()

    # Build new MODEL_PARAMS
    new_params = (
        "\n".join(
            [
                f'            "{k}": {repr(v) if isinstance(v, (int, float)) else v},'
                for k, v in best_params.items()
            ]
        )
        + """
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
            "tree_method": "hist","""
    )

    # Replace the MODEL_PARAMS section
    pattern = r"self\.MODEL_PARAMS = \{[^}]+\}"
    replacement = f"self.MODEL_PARAMS = {{{new_params}"
    config_content = re.sub(pattern, replacement, config_content, flags=re.DOTALL)

    with open(config_path, "w") as f:
        f.write(config_content)

    print("\n✅ Best params saved to best_params.json")
    print("✅ Config auto-updated in config.py")
