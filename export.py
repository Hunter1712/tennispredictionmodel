"""
Export predictions for web deployment.
Uses cached processed data from pipeline to generate player predictions.
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import config, logger
from src.model import load_model
from src.features import FEATURE_COLS
from src.pipeline import load_cached_data
from src.exceptions import PredictionError, DataLoadError


def get_player_stats(df: pd.DataFrame, name: str) -> dict | None:
    """Extract career statistics for a player from match history.
    
    Args:
        df: DataFrame with match data (winner/loser columns).
        name: Player name to look up.
        
    Returns:
        Dict with player stats including rank, Elo, surface win rates.
        None if player has no matches.
    """
    wins = df[df["winner_name"] == name].copy()
    losses = df[df["loser_name"] == name].copy()

    if wins.empty and losses.empty:
        return None

    # Combine all matches
    wins["is_winner"] = True
    losses["is_winner"] = False
    all_matches = pd.concat([wins, losses])

    # Use career peak rank points
    peak_rank_points = max(
        wins["winner_rank_points"].max() if len(wins) > 0 else 0,
        losses["loser_rank_points"].max() if len(losses) > 0 else 0,
    )

    # Get latest match for current stats
    latest = all_matches.sort_values("tourney_date", ascending=False).iloc[0]
    is_winner = latest["is_winner"]
    prefix = "winner" if is_winner else "loser"

    # Overall win rate
    total_wins = len(wins)
    total_losses = len(losses)
    win_rate = total_wins / max(total_wins + total_losses, 1)

    # Recent form (last 10 matches)
    recent = all_matches.sort_values("tourney_date", ascending=False).head(10)
    recent_wins = len(recent[recent["winner_name"] == name])
    recent_form = recent_wins / max(len(recent), 1)

    stats = {
        "rank": int(latest[f"{prefix}_rank"]),
        "rank_points": int(peak_rank_points),
        "age": float(latest[f"{prefix}_age"]),
        "height": int(latest.get(f"{prefix}_ht") or config.DEFAULT_PLAYER_HEIGHT),
        "seed": int(latest.get(f"{prefix}_seed") or 0),
        "win_rate": float(win_rate),
        "recent_form": float(recent_form),
        "matches": len(all_matches),
    }

    # Surface-specific win rates
    for surf in ["Hard", "Clay", "Grass", "Carpet"]:
        wins_surf = wins[wins["surface"] == surf]
        losses_surf = losses[losses["surface"] == surf]
        total = len(wins_surf) + len(losses_surf)
        stats[f"surface_{surf}"] = len(wins_surf) / max(total, 1)

    return stats


def get_features(player_a: dict, player_b: dict) -> dict:
    """Compute the 10 model features for a player matchup.
    
    Args:
        player_a: Stats dict for player A.
        player_b: Stats dict for player B.
        
    Returns:
        Dict with feature values required by the model (FEATURE_COLS).
    """
    # Compute diff-based features (only those used by model)
    features = {
        "elo_diff": 0,  # Placeholder (not computed for export predictions)
        "elo_surface_diff": 0,  # Placeholder
        "rank_points_diff": player_a["rank_points"] - player_b["rank_points"],
        "rank_diff": player_b["rank"] - player_a["rank"],  # Inverted: lower rank is better
        "age_diff": player_a["age"] - player_b["age"],
        "seed_diff": player_a["seed"] - player_b["seed"],
        "height_diff": player_a["height"] - player_b["height"],
        "days_since_last_diff": 0,  # Not computed for export predictions
        "rest_quality_diff": 0.5,  # Neutral default
        "win_rate_diff": player_a["win_rate"] - player_b["win_rate"],
    }
    
    # Validate all required features are present
    missing = set(FEATURE_COLS) - set(features.keys())
    if missing:
        raise PredictionError(f"Missing features: {missing}")
    
    return features


def main() -> None:
    """Generate predictions for all top players and export as JavaScript.
    
    Exports predictions as const PREDICTIONS to predictions.js for web UI.
    Loads cached data, computes player statistics, and batch predicts matchups.
    """
    logger.info("EXPORTING PREDICTIONS")
    logger.info("=" * 60)

    # [1] Load cached data
    logger.info("\n[1/5] Loading cached data...")
    try:
        df = load_cached_data()
        if df is None:
            raise DataLoadError(
                "No cache found. Run 'python train.py' first to generate cache."
            )
        logger.info(f"  Loaded {len(df):,} samples from cache")
    except Exception as e:
        logger.error(f"Failed to load cache: {e}")
        return

    # [2] Find active players
    logger.info("\n[2/5] Finding top players...")
    try:
        # Calculate active player window based on config
        current_year = datetime.now().year
        active_from_year = current_year - config.RECENT_YEARS_WINDOW
        recent_df = df[df["year"] >= active_from_year]
        
        # Players who competed in recent years
        recent_players = set(recent_df["winner_name"].dropna()) | set(
            recent_df["loser_name"].dropna()
        )
        logger.info(f"  {len(recent_players)} active players (year {active_from_year}+)")

        # Get career peak rank points for active players only
        all_players = set(df["winner_name"].dropna()) | set(df["loser_name"].dropna())
        player_pts = {}
        
        for p in all_players:
            if p not in recent_players:
                continue
            wins = df[df["winner_name"] == p]["winner_rank_points"].dropna()
            losses = df[df["loser_name"] == p]["loser_rank_points"].dropna()
            max_pts = max(
                (wins.max() if len(wins) > 0 else 0),
                (losses.max() if len(losses) > 0 else 0),
            )
            player_pts[p] = max_pts

        # Select top 250 by rank points
        top_players = sorted(player_pts.items(), key=lambda x: x[1], reverse=True)[:250]
        names = [p[0] for p in top_players]
        logger.info(f"  Selected top {len(names)} players by rank points")
        logger.info(f"  Top 5: {', '.join(names[:5])}")
    except Exception as e:
        logger.error(f"Failed to find players: {e}")
        return

    # [3] Compute player statistics
    logger.info("\n[3/5] Computing player statistics...")
    try:
        players = {n: get_player_stats(df, n) for n in names}
        players = {k: v for k, v in players.items() if v}  # Remove None entries
        logger.info(f"  Computed stats for {len(players)} players")
    except Exception as e:
        logger.error(f"Failed to compute player stats: {e}")
        return

    # [4] Generate predictions
    logger.info("\n[4/5] Generating predictions...")
    try:
        model = load_model()
        
        # Build matchup pairs
        pairs = []
        player_list = [n for n in names if n in players]
        for i, a in enumerate(player_list):
            for b in player_list[i + 1:]:
                pairs.append((a, b, players[a], players[b]))
        
        logger.info(f"  Predicting {len(pairs)} matchups...")
        
        predictions = {}
        if pairs:
            # Build feature matrix for batch prediction
            features_arr = np.array(
                [
                    [get_features(pa, pb)[c] for c in FEATURE_COLS]
                    for _, _, pa, pb in pairs
                ]
            )
            
            # Validate feature matrix shape
            if features_arr.shape[1] != len(FEATURE_COLS):
                raise PredictionError(
                    f"Feature mismatch: expected {len(FEATURE_COLS)}, got {features_arr.shape[1]}"
                )
            
            # Batch predict probabilities
            probs = model.predict_proba(features_arr)[:, 1]
            
            if len(probs) != len(pairs):
                raise PredictionError(
                    f"Prediction count mismatch: {len(probs)} vs {len(pairs)}"
                )
            
            # Build predictions with surface-specific blends
            overall_weight, surface_weight = config.SURFACE_BLEND_WEIGHTS
            
            for idx, (a, b, pa, pb) in enumerate(pairs):
                overall = float(probs[idx])
                pred = {"overall": round(overall, 4)}
                
                # Surface-specific predictions (blend overall with surface win rate)
                for surf in ["Hard", "Clay", "Grass", "Carpet"]:
                    surf_a = float(pa.get(f"surface_{surf}", 0.5))
                    surf_b = float(pb.get(f"surface_{surf}", 0.5))
                    
                    # Normalize surface win rates to probability
                    surface_blend = (
                        surf_a / (surf_a + surf_b + 0.001) if (surf_a + surf_b) > 0 else 0.5
                    )
                    
                    # Weighted combination: mostly overall prediction, some surface blend
                    pred[surf] = round(
                        overall * overall_weight + surface_blend * surface_weight, 4
                    )
                
                predictions[f"{a}|{b}"] = pred
        
        logger.info(f"  Generated {len(predictions)} predictions")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return

    # [5] Save to JavaScript
    logger.info("\n[5/5] Exporting to JavaScript...")
    try:
        # Build metadata
        output = {
            "meta": {
                "model": "LightGBM",
                "features": len(FEATURE_COLS),
                "feature_list": FEATURE_COLS,
                "trained_years": "1991-2026",
                "test_years": "2025-2026",
                "predictions_count": len(predictions),
                "players": len(players),
                "export_date": datetime.now().isoformat(),
            },
            "players": [{"name": n, **s} for n, s in players.items()],
            "predictions": predictions,
        }

        # Generate JavaScript constant
        js_content = "const PREDICTIONS = " + json.dumps(output) + ";"

        # Save to root (Cloudflare Pages will serve it)
        with open("predictions.js", "w") as f:
            f.write(js_content)
        
        logger.info(f"  Saved {len(predictions):,} predictions to predictions.js")
        logger.info("\nExport complete")
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return


if __name__ == "__main__":
    main()
