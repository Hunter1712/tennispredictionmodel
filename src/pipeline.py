"""
Pipeline orchestration for Tennis Match Prediction Model
"""

import os
import pickle

from typing import Optional

from xgboost import XGBClassifier

from config import config, logger
from loader import load_all_csv
from cleaner import clean_data
from features import (
    build_player_stats,
    engineer_features,
    prepare_model_data,
    FEATURE_COLS,
)
from model import (
    chronological_split,
    train_model,
    evaluate_model,
    save_model,
    load_model,
    predict_match,
)

CACHE_PATH = "../data_cache.pkl"


class TennisPredictionPipeline:
    """Complete pipeline for tennis match prediction."""

    def __init__(self):
        self.model: Optional[XGBClassifier] = None
        self.results: Optional[dict] = None
        self.feature_cols = FEATURE_COLS

    def run(self) -> tuple[XGBClassifier, dict]:
        """Execute full training pipeline."""
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

    def predict(self, match_features: dict) -> tuple[int, float]:
        """Make a prediction for a match."""
        if self.model is None:
            self.load_existing()
        return predict_match(self.model, match_features, self.feature_cols)


def run_pipeline() -> tuple[XGBClassifier, dict]:
    """Convenience function to run the pipeline."""
    pipeline = TennisPredictionPipeline()
    return pipeline.run()


def load_existing_model() -> XGBClassifier:
    """Convenience function to load existing model."""
    return load_model()


def load_cached_data() -> pd.DataFrame:
    """Load cached processed data (for export_predictions.py)."""
    import pandas as pd

    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return None
