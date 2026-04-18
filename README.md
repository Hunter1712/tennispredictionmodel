# 🎾 Tennis Match Predictor

ML-powered prediction of ATP tennis match winners using XGBoost.

## Features

- **4-feature model**: Elo ratings + fatigue metrics
- **Chronological split**: Train on 1991-2024, test on 2025-2026
- **Auto-tuning**: Optuna hyperparameter optimization
- **Live predictions**: Exported to JavaScript for web deployment

## Quick Start

```bash
# Install dependencies
uv sync

# Train model
uv run python src/main.py

# Tune hyperparameters (optional)
uv run python src/tune.py

# Export predictions for web
uv run python src/export_predictions.py
```

## Project Structure

```
├── src/
│   ├── main.py              # Entry point - trains model
│   ├── tune.py              # Hyperparameter tuning with Optuna
│   ├── export_predictions.py # Generate predictions.js
│   ├── pipeline.py          # Training pipeline
│   ├── features.py          # Feature engineering (Elo + fatigue)
│   ├── cleaner.py           # Data cleaning
│   ├── loader.py            # CSV data loading
│   ├── model.py             # Model training/evaluation
│   └── config.py            # Configuration & hyperparameters
├── data/                    # Raw CSV match data (1991-2026)
├── models/                  # Trained model (.pkl)
├── output/                  # predictions.js for web UI
├── pyproject.toml           # Dependencies
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
| Test ROC-AUC | **0.8228** |
| CV ROC-AUC | **0.8831** |

### Features Used

| Feature | Importance |
|---------|------------|
| `days_since_last_diff` | 57.2% |
| `rest_quality_diff` | 25.1% |
| `elo_diff` | 10.3% |
| `elo_surface_diff` | 7.4% |

## Requirements

- Python 3.14+
- [uv](https://github.com/astral-sh/uv) for dependency management

Install: `uv sync`

## License

MIT License

## Notes

- **Data files** (`data/*.csv`) are not committed (too large)
- **Trained model** (`models/*.pkl`) is not committed
- **Predictions** (`output/predictions.js`) - generated for web deployment