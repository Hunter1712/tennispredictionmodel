"""
Configuration and constants for Tennis Match Prediction Model
"""

import logging
import sys
from dataclasses import dataclass


@dataclass
class Config:
    """Main configuration for the tennis prediction model"""

    # File paths
    DATA_FOLDER: str = "data"
    MODEL_PATH: str = "models/tennis_model.pkl"
    DATA_CACHE_PATH: str = "models/data_cache.pkl"

    # Training configuration
    TRAIN_START_YEAR: int = 2000
    TRAIN_END_YEAR: int = 2023
    TEST_START_YEAR: int = 2024
    TEST_END_YEAR: int = 2026

    # Cross-validation
    CV_FOLDS: int = 5
    CV_RANDOM_STATE: int = 42

    # Export/prediction constants
    RECENT_YEARS_WINDOW: int = 3  # Years to consider for active players
    SURFACE_BLEND_WEIGHTS: tuple[float, float] = (0.7, 0.3)  # (overall_weight, surface_weight)
    DEFAULT_PLAYER_HEIGHT: int = 180  # cm
    DEFAULT_DRAW_SIZE: int = 32
    DEFAULT_TOURNAMENT_ROUND: int = 3

    # Feature columns (10 diff-based features)
    FEATURE_COLS: list[str] = None

    # Model hyperparameters
    MODEL_PARAMS: dict = None

    def __post_init__(self):
        # Initialize feature columns (single source of truth)
        if self.FEATURE_COLS is None:
            self.FEATURE_COLS = [
                "elo_diff",
                "elo_surface_diff",
                "rank_points_diff",
                "rank_diff",
                "age_diff",
                "seed_diff",
                "height_diff",
                "days_since_last_diff",
                "rest_quality_diff",
                "win_rate_diff",
            ]

        # Initialize model hyperparameters
        if self.MODEL_PARAMS is None:
            self.MODEL_PARAMS = {
                "n_estimators": 272,
                "max_depth": 9,
                "learning_rate": 0.05420686096244507,
                "min_child_samples": 5,
                "colsample_bytree": 0.9152072097559515,
                "subsample": 0.8338997111968103,
                "reg_alpha": 0.006693252268057703,
                "reg_lambda": 0.08081397412257926,
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
                "force_col_wise": True,
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
