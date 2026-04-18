"""
Feature engineering module for Tennis Match Prediction Model.
Handles historical player statistics and feature creation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .config import logger
from .exceptions import FeatureEngineeringError

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Elo constants
ELO_DEFAULT = 1500.0
ELO_K_FACTOR = 32


def _compute_rest_quality(days: pd.Series) -> NDArray[np.float64]:
    """Vectorized rest quality computation.

    Args:
        days: Series of days since last match.

    Returns:
        Array of rest quality scores (0.0 to 1.0).
    """
    d = days.values
    result = np.zeros(len(d))
    result[(d == 0)] = 0.0
    result[(d >= 3) & (d <= 7)] = 1.0
    result[(d > 0) & (d < 3)] = 0.5 + (d[(d > 0) & (d < 3)] / 6)
    result[d > 7] = np.maximum(0.0, 1.0 - ((d[d > 7] - 7) / 30))
    return result


# Only Elo + fatigue features for simpler model
FEATURE_COLS: list[str] = [
    "elo_diff",
    "elo_surface_diff",
    "days_since_last_diff",
    "rest_quality_diff",
]


def build_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Build historical player stats with efficient batch processing.

    Args:
        df: DataFrame with match data sorted by date.

    Returns:
        DataFrame with computed player statistics.
    """
    logger.info("Building historical player statistics")

    df = df.sort_values("tourney_date").reset_index(drop=True)

    for col in ["winner_rank", "loser_rank", "winner_rank_points", "loser_rank_points"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Store last match date per player (for rest/fatigue calculation)
    player_last_date: dict[str, pd.Timestamp] = {}

    # Elo ratings: player_elo[player_id] = {"overall": elo, "hard": elo, "clay": elo, "grass": elo}
    player_elo = defaultdict(
        lambda: {
            "overall": ELO_DEFAULT,
            "hard": ELO_DEFAULT,
            "clay": ELO_DEFAULT,
            "grass": ELO_DEFAULT,
        }
    )

    # Pre-allocate arrays (only for used features)
    n = len(df)
    winner_elo, loser_elo = np.zeros(n), np.zeros(n)
    winner_elo_surface, loser_elo_surface = np.zeros(n), np.zeros(n)
    winner_days, loser_days = np.zeros(n), np.zeros(n)

    for idx in range(n):
        row = df.iloc[idx]
        w_id, l_id = row["winner_id"], row["loser_id"]
        surface, date = row["surface"], row["tourney_date"]
        surf_key = surface.lower() if surface in ["Hard", "Clay", "Grass"] else "hard"

        w_elo_stats, l_elo_stats = player_elo[w_id], player_elo[l_id]

        # Elo ratings (from before this match)
        winner_elo[idx] = w_elo_stats["overall"]
        loser_elo[idx] = l_elo_stats["overall"]
        winner_elo_surface[idx] = w_elo_stats[surf_key]
        loser_elo_surface[idx] = l_elo_stats[surf_key]

        # Rest days since last match
        if w_id in player_last_date:
            winner_days[idx] = (date - player_last_date[w_id]).days
        if l_id in player_last_date:
            loser_days[idx] = (date - player_last_date[l_id]).days

        # Update last match date for both players
        player_last_date[w_id] = date
        player_last_date[l_id] = date

        # Update Elo ratings
        w_elo, l_elo = w_elo_stats["overall"], l_elo_stats["overall"]
        exp_winner = 1 / (1 + 10 ** ((l_elo - w_elo) / 400))
        exp_loser = 1 - exp_winner
        w_elo_stats["overall"] = w_elo + ELO_K_FACTOR * (1 - exp_winner)
        l_elo_stats["overall"] = l_elo + ELO_K_FACTOR * (0 - exp_loser)

        # Surface-specific Elo
        w_elo_surf, l_elo_surf = w_elo_stats[surf_key], l_elo_stats[surf_key]
        exp_winner_surf = 1 / (1 + 10 ** ((l_elo_surf - w_elo_surf) / 400))
        exp_loser_surf = 1 - exp_winner_surf
        w_elo_stats[surf_key] = w_elo_surf + ELO_K_FACTOR * (1 - exp_winner_surf)
        l_elo_stats[surf_key] = l_elo_surf + ELO_K_FACTOR * (0 - exp_loser_surf)

    # Assign only the columns we actually use
    df["winner_elo"] = winner_elo
    df["loser_elo"] = loser_elo
    df["winner_elo_surface"] = winner_elo_surface
    df["loser_elo_surface"] = loser_elo_surface
    df["days_since_last_match_winner"] = winner_days
    df["days_since_last_match_loser"] = loser_days
    df["winner_rest_quality"] = _compute_rest_quality(df["days_since_last_match_winner"])
    df["loser_rest_quality"] = _compute_rest_quality(df["days_since_last_match_loser"])

    logger.info(f"Features built for {len(df)} matches")
    return df


# Player feature columns we actually use
_PLAYER_COLS = [
    "rank",
    "rank_points",
    "age",
    "ht",
    "seed",
    "elo",
    "elo_surface",
    "rest_quality",
]


def _map_player_features(df: pd.DataFrame, prefix: str, target: int) -> pd.DataFrame:
    """Map winner/loser columns to player/opponent format for binary classification.

    Args:
        df: DataFrame with winner/loser columns.
        prefix: 'winner' or 'loser'.
        target: Target value (1 for player wins, 0 for loses).

    Returns:
        DataFrame with player/opponent format columns.
    """
    df = df.copy()
    df["target"] = target

    # Map player (winner or loser based on prefix)
    for col in _PLAYER_COLS:
        src = f"{prefix}_{col}"
        dst = f"player_{col}"
        if src in df.columns:
            df[dst] = df[src]

    # Map opponent (the other player)
    opp_prefix = "loser" if prefix == "winner" else "winner"
    for col in _PLAYER_COLS:
        src = f"{opp_prefix}_{col}"
        dst = f"opponent_{col}"
        if src in df.columns:
            df[dst] = df[src]

    # Handle special columns
    df["player_seed"] = pd.to_numeric(
        df.get(f"{prefix}_seed", 0), errors="coerce"
    ).fillna(0)
    df["opponent_seed"] = pd.to_numeric(
        df.get(f"{opp_prefix}_seed", 0), errors="coerce"
    ).fillna(0)
    df["days_since_last"] = df.get(f"days_since_last_match_{prefix}", 0)
    df["opponent_days_since_last"] = df.get(f"days_since_last_match_{opp_prefix}", 0)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary classification features from match data.

    Args:
        df: DataFrame with player statistics.

    Returns:
        DataFrame with engineered features for binary classification.
    """
    logger.info("Engineering features")

    try:
        # Create two rows per match: one where winner is "player", one where loser is "player"
        df_winner = _map_player_features(df.copy(), "winner", 1)
        df_loser = _map_player_features(df.copy(), "loser", 0)

        df_features = pd.concat([df_winner, df_loser], ignore_index=True)

        # Compute only the diff features we use (4 features)
        df_features["elo_diff"] = (
            df_features["player_elo"] - df_features["opponent_elo"]
        )
        df_features["elo_surface_diff"] = (
            df_features["player_elo_surface"] - df_features["opponent_elo_surface"]
        )
        df_features["days_since_last_diff"] = (
            df_features["days_since_last"] - df_features["opponent_days_since_last"]
        )
        df_features["rest_quality_diff"] = (
            df_features["player_rest_quality"] - df_features["opponent_rest_quality"]
        )

        # Cap fatigue features to reasonable bounds
        MAX_REST_DAYS = 21
        df_features["days_since_last_diff"] = df_features["days_since_last_diff"].clip(
            -MAX_REST_DAYS, MAX_REST_DAYS
        )
        df_features["rest_quality_diff"] = df_features["rest_quality_diff"].clip(
            -1.0, 1.0
        )

        df_features = df_features.replace([np.inf, -np.inf], 0).fillna(0)

        logger.info(f"Features engineered: {len(df_features)} samples")
        return df_features

    except Exception as e:
        raise FeatureEngineeringError(f"Failed to engineer features: {e}") from e


def prepare_model_data(
    df_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Select final features and prepare X, y for modeling.

    Args:
        df_features: DataFrame with engineered features.

    Returns:
        Tuple of (X, y, feature_columns).
    """
    X = df_features[FEATURE_COLS].copy()
    y = df_features["target"].copy()
    return X, y, FEATURE_COLS
