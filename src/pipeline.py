"""
Pipeline orchestration for Tennis Match Prediction Model.
"""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import pandas as pd
from xgboost import XGBClassifier

from .config import config, logger
from .loader import load_all_csv
from .cleaner import clean_data
from .features import (
    build_player_stats,
    engineer_features,
    prepare_model_data,
    FEATURE_COLS,
)
from .model import (
    chronological_split,
    train_model,
    evaluate_model,
    save_model,
    load_model,
    predict_match,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

CACHE_PATH = "../data_cache.pkl"


class TennisPredictionPipeline:
    """Complete pipeline for tennis match prediction."""

    def __init__(self) -> None:
        self.model: XGBClassifier | None = None
        self.results: dict[str, float | pd.DataFrame] | None = None
        self.feature_cols: list[str] = FEATURE_COLS

    def run(self) -> tuple[XGBClassifier, dict[str, float | pd.DataFrame]]:
        """Execute full training pipeline.

        Returns:
            Tuple of (trained_model, evaluation_results).
        """
        logger.info("=" * 60)
        logger.info("TENNIS MATCH WINNER PREDICTION MODEL")
        logger.info("=" * 60)

        logger.info("1. LOADING DATA")
        df = load_all_csv()

        logger.info("2. CLEANING DATA")
        df = clean_data(df)

        logger.info("3. BUILDING PLAYER STATS")
        df = build_player_stats(df)

        logger.info("4. ENGINEERING FEATURES")
        df_features = engineer_features(df)

        # Save processed data for export_predictions.py to use
        logger.info("Caching processed data...")
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(df_features, f)
        logger.info(f"Data cache saved to {CACHE_PATH}")

        logger.info("5. PREPARING MODEL DATA")
        X, y, self.feature_cols = prepare_model_data(df_features)

        logger.info("6. SPLITTING DATA")
        X_train, X_test, y_train, y_test = chronological_split(
            df_features, self.feature_cols
        )

        logger.info("7. TRAINING MODEL")
        self.model = train_model(X_train, y_train)

        logger.info("8. EVALUATING MODEL")
        self.results = evaluate_model(
            self.model, X_train, X_test, y_train, y_test, self.feature_cols
        )

        logger.info("9. SAVING MODEL")
        save_model(self.model)

        logger.info("SUMMARY")
        logger.info(f"  Test Accuracy: {self.results['test_accuracy']:.2%}")
        logger.info(f"  Test ROC-AUC:  {self.results['test_roc_auc']:.4f}")
        logger.info(f"  Model saved to: {config.MODEL_PATH}")

        return self.model, self.results

    def load_existing(self) -> XGBClassifier:
        """Load an existing model."""
        self.model = load_model()
        return self.model

    def predict(self, match_features: dict[str, float]) -> tuple[int, float]:
        """Make a prediction for a match.

        Args:
            match_features: Dictionary of feature values.

        Returns:
            Tuple of (prediction, probability).
        """
        if self.model is None:
            self.load_existing()
        return predict_match(self.model, match_features, self.feature_cols)


def run_pipeline() -> tuple[XGBClassifier, dict[str, float | pd.DataFrame]]:
    """Convenience function to run the pipeline."""
    pipeline = TennisPredictionPipeline()
    return pipeline.run()


def load_cached_data() -> pd.DataFrame | None:
    """Load cached processed data for export_predictions.py."""
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return None
