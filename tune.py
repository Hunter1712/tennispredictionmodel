"""
Hyperparameter tuning with Optuna for Tennis Prediction Model.
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Any

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from src.config import config, logger
from src.exceptions import DataLoadError


def load_data() -> pd.DataFrame:
    """Load cached processed data for hyperparameter tuning.
    
    Returns:
        DataFrame with engineered features and target variable.
        
    Raises:
        DataLoadError: If cache file not found or invalid.
    """
    try:
        cache_path = config.DATA_CACHE_PATH
        
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Cache file not found at {cache_path}\n"
                "Run 'python train.py' first to generate the cache."
            )
        
        logger.info(f"Loading cached data from {cache_path}")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        
        # Validate loaded data
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Expected DataFrame, got {type(data)}")
        
        if data.empty:
            raise ValueError("Loaded cache is empty")
        
        logger.info(f"Loaded {len(data):,} samples from cache")
        return data
        
    except FileNotFoundError as e:
        raise DataLoadError(str(e)) from e
    except Exception as e:
        raise DataLoadError(f"Failed to load tuning data: {e}") from e


def tune_hyperparameters(n_trials: int = 50) -> dict[str, float]:
    """Run hyperparameter tuning with Optuna.

    Args:
        n_trials: Number of optimization trials to run.

    Returns:
        Dictionary of best hyperparameters found.
    """
    # Load cached data
    df = load_data()

    # Use feature columns from centralized config
    feature_cols = config.FEATURE_COLS
    
    X = df[feature_cols].copy()
    y = df["target"].copy()

    logger.info(f"Data shape: {X.shape}")
    logger.info(f"Running {n_trials} trials for hyperparameter tuning...")

    # Suppress optuna verbosity
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )

    # Fixed cross-validation splits for reproducibility
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_splits = list(skf.split(X, y))

    def objective_fixed(trial):
        """Objective function for Optuna optimization."""
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, 20),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
            "force_col_wise": True,
        }

        scores = []
        for train_idx, val_idx in cv_splits:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = LGBMClassifier(**params)
            model.fit(X_train, y_train)

            y_proba = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_proba)
            scores.append(score)

        return np.mean(scores)

    study.optimize(objective_fixed, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"\nBest ROC-AUC: {study.best_value:.4f}")
    logger.info("Best hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")

    return study.best_params


if __name__ == "__main__":
    best_params = tune_hyperparameters(n_trials=50)

    # Update config.py with best parameters
    import re
    
    config_file = os.path.join(os.path.dirname(__file__), "src", "config.py")
    
    with open(config_file, "r") as f:
        config_content = f.read()

    # Build new MODEL_PARAMS section
    new_params = "\n".join(
        [
            f'            "{k}": {repr(v) if isinstance(v, (int, float)) else v},'
            for k, v in best_params.items()
        ]
    ) + """
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
            "force_col_wise": True,"""

    # Replace the MODEL_PARAMS section using regex
    pattern = r'self\.MODEL_PARAMS = \{[^}]+\}'
    replacement = f"self.MODEL_PARAMS = {{{new_params}\n        }}"
    config_content = re.sub(pattern, replacement, config_content, flags=re.DOTALL)

    with open(config_file, "w") as f:
        f.write(config_content)

    logger.info("\n✅ Best hyperparameters saved")
    logger.info("✅ Config auto-updated in src/config.py")
