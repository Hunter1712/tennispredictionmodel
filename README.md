# Tennis Match Predictor

ATP tennis match winner prediction using LightGBM.

## Quick Start

```bash
pip install -r requirements.txt
python train.py --test      # Train with test evaluation
python train.py             # Train on all data
python export.py            # Generate predictions.js
python tune.py              # Hyperparameter tuning
```

## Files

- `train.py` - Train/evaluate model
- `export.py` - Generate predictions.js for web
- `tune.py` - Hyperparameter tuning
- `src/` - Pipeline modules (config, loader, cleaner, features, model)
- `index.html` - Web UI (open after running export.py)

## Model

- **Algorithm**: LightGBM gradient boosting classifier
- **Features**: 10 diff-based metrics (Elo, ranking, player stats, fatigue)
- **Validation**: Chronological train/test split
- **Performance**: ~75% accuracy on test data