"""
Configuration and constants for Tennis Match Prediction Model
"""

import logging
import sys
from dataclasses import dataclass


@dataclass
class Config:
    """Main configuration for the tennis prediction model"""

    DATA_FOLDER: str = "data"
    MODEL_PATH: str = "models/tennis_model.pkl"

    # Production: train on all data (1991-2026)
    # Use --test flag to enable test mode (train on 1991-2024, test on 2025-2026)
    TRAIN_WITH_TEST: bool = False  # Set via command line arg

    TRAIN_START_YEAR: int = 1991
    TRAIN_END_YEAR: int = 2026  # All data by default
    TEST_START_YEAR: int = 2025
    TEST_END_YEAR: int = 2026

    CV_FOLDS: int = 5
    CV_RANDOM_STATE: int = 42

    MODEL_PARAMS: dict = None

    def __post_init__(self):
        self.MODEL_PARAMS = {
            "n_estimators": 500,
            "max_depth": 7,
            "learning_rate": 0.05,
            "min_child_weight": 5,
            "colsample_bytree": 0.8,
            "subsample": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
            "tree_method": "hist",
        }


config = Config()


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the application logger."""
    logger = logging.getLogger("tennis_prediction")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logging()
