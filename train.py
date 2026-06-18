"""
Tennis Match Winner Prediction Model
Main entry point
"""

import sys
from src.pipeline import run_pipeline
from src.config import config, logger


def main():
    # Parse command line args
    test_mode = "--test" in sys.argv

    if test_mode:
        # Enable test mode: train 1991-2024, test 2025-2026
        logger.info("MODE: Testing (--test flag detected)")
        config.TRAIN_END_YEAR = 2024
    else:
        logger.info("MODE: Production (train on all data)")

    model, results = run_pipeline()

    if test_mode:
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        print(f"Test Accuracy: {results['test_accuracy']:.2%}")
        print(f"Test ROC-AUC:  {results['test_roc_auc']:.4f}")
    else:
        print("\n" + "=" * 60)
        print("MODEL TRAINED ON ALL DATA")
        print("=" * 60)
        print(f"Samples used: {results.get('train_samples', 'N/A')}")


if __name__ == "__main__":
    main()
