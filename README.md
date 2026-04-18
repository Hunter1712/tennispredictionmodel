# Tennis Match Predictor

ML-powered prediction of ATP tennis match winners using XGBoost.

## Features

- **4-feature model**: Elo ratings + fatigue metrics
- **Chronological split**: Train on 1991-2024, test on 2025-2026
- **Auto-tuning**: Optuna hyperparameter optimization
- **Exported predictions**: JavaScript for web deployment

## Quick Start

```bash
# Install dependencies
uv sync

# Train model (with test evaluation)
PYTHONPATH=. uv run python train.py --test

# Train on all data (production)
PYTHONPATH=. uv run python train.py

# Generate predictions.js for web
PYTHONPATH=. uv run python export.py

# Tune hyperparameters (optional)
PYTHONPATH=. uv run python tune.py
```

## Project Structure

```
├── train.py              # Train/evaluate model
├── export.py            # Generate predictions.js
├── tune.py              # Hyperparameter tuning
├── src/
│   ├── __init__.py
│   ├── config.py        # Configuration & hyperparameters
│   ├── exceptions.py   # Custom exceptions
│   ├── loader.py       # CSV data loading
│   ├── cleaner.py      # Data cleaning
│   ├── features.py     # Feature engineering (Elo + fatigue)
│   ├── model.py        # Model training/evaluation
│   └── pipeline.py     # Pipeline orchestration
├── data/                # Raw CSV match data (1991-2026)
├── models/              # Trained model (.pkl)
├── output/              # predictions.js for web UI
└── README.md
```

## How It Works

1. **Load**: ATP match data from CSV files (1991-2026)
2. **Clean**: Remove incomplete matches, handle missing values
3. **Features**: Build player stats (Elo ratings, rest quality, days since last match)
4. **Train**: XGBoost classifier with tuned hyperparameters
5. **Predict**: Generate matchup predictions for web UI

## Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | **73.55%** |
| Test ROC-AUC | **0.8227** |
| CV ROC-AUC | **0.8831** |

### Features Used

| Feature | Importance |
|---------|------------|
| `days_since_last_diff` | 57.0% |
| `rest_quality_diff` | 25.3% |
| `elo_diff` | 10.4% |
| `elo_surface_diff` | 7.4% |

## Requirements

- Python 3.14+
- [uv](https://github.com/astral-sh/uv) for dependency management

Install: `uv sync`

## Notes

- **Data files** (`data/*.csv`) - not committed (too large)
- **Trained model** (`models/*.pkl`) - not committed
- **Predictions** (`output/predictions.js`) - generated for web deployment