"""
Tennis Match Winner Prediction Model
Main entry point
"""

from pipeline import run_pipeline


if __name__ == "__main__":
    model, results = run_pipeline()

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Test Accuracy: {results['test_accuracy']:.2%}")
    print(f"Test ROC-AUC:  {results['test_roc_auc']:.4f}")
