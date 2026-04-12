# Tennis Match Predictor (Backend)

XGBoost model predicting ATP tennis match winners.

**This is the private backend repository.** The web UI is in a separate public repo.

## Quick Start

```bash
# Train model (generates cache)
python src/main.py

# Export predictions to output/predictions.js
python src/export_predictions.py
```

## Project Structure

```
├── src/                    # Python source code
│   ├── main.py            # Entry point - trains model
│   ├── export_predictions.py # Generate predictions.js
│   ├── pipeline.py        # Training pipeline
│   ├── features.py       # Feature engineering
│   ├── cleaner.py         # Data cleaning
│   ├── loader.py          # CSV data loading
│   ├── model.py           # Model training/evaluation
│   └── config.py          # Configuration
├── data/                   # Raw CSV match data (1991-2026) - NOT committed
├── models/                 # Trained model (.pkl) - NOT committed
├── output/                 # predictions.js output - NOT committed
├── .github/workflows/     # GitHub Actions for auto-sync
└── pyproject.toml         # Dependencies
```

## How It Works

1. **Data**: Loads CSV files from `data/` folder
2. **Cleaning**: Removes incomplete matches, handles missing values
3. **Features**: Builds player stats (win rate, recent form, surface skill)
4. **Training**: XGBoost classifier trained on 1991-2024, tested on 2025-2026
5. **Export**: Generates `output/predictions.js` for web UI

## Syncing to Public Repo

### Option 1: Manual Copy
1. Run `python src/export_predictions.py`
2. Copy `output/predictions.js` to your public repo

### Option 2: GitHub Actions (see `.github/workflows/sync.yml`)
1. Go to **Actions** tab on GitHub
2. Run **Sync Predictions** workflow
3. Enter your public repo name and commit message

## Model Performance

- **Test Accuracy**: ~75%
- **ROC-AUC**: ~0.85
- **CV Accuracy**: ~81%

## Requirements

- Python 3.14+
- xgboost, pandas, scikit-learn, numpy

Install: `uv sync` or `pip install -r requirements.txt`

## Notes

- **Data files** (`data/*.csv`) are NOT committed (too large, keep local)
- **Trained model** (`models/*.pkl`) is NOT committed
- **Predictions** (`output/predictions.js`) - sync manually or via GitHub Actions to public repo