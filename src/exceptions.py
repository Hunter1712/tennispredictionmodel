"""
Custom exceptions for Tennis Match Prediction Model
"""


class TennisPredictionError(Exception):
    """Base exception for all tennis prediction errors"""

    pass


class DataLoadError(TennisPredictionError):
    """Failed to load data from CSV files"""

    pass


class DataCleanError(TennisPredictionError):
    """Failed to clean data"""

    pass


class FeatureEngineeringError(TennisPredictionError):
    """Failed during feature engineering"""

    pass


class ModelTrainingError(TennisPredictionError):
    """Failed during model training"""

    pass


class ModelEvaluationError(TennisPredictionError):
    """Failed during model evaluation"""

    pass


class ModelSaveError(TennisPredictionError):
    """Failed to save model to disk"""

    pass


class ModelLoadError(TennisPredictionError):
    """Failed to load model from disk"""

    pass


class PredictionError(TennisPredictionError):
    """Failed to make prediction"""

    pass


class InvalidFeatureError(TennisPredictionError):
    """Invalid feature provided for prediction"""

    pass
