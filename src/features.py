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


# All features for model (10 diff features - low overfit risk)
FEATURE_COLS: list[str] = [
    "elo_diff",
    "elo_surface_diff",
    "rank_points_diff",
    "rank_diff",
    "age_diff",
    "seed_diff",
    "height_diff",
    "days_since_last_diff",
    "rest_quality_diff",
    "win_rate_diff",
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

    for col in ["winner_rank", "loser_rank", "winner_rank_points", "loser_rank_points", "winner_age", "loser_age"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    player_last_date: dict[str, pd.Timestamp] = {}

    player_elo = defaultdict(
        lambda: {
            "overall": ELO_DEFAULT,
            "hard": ELO_DEFAULT,
            "clay": ELO_DEFAULT,
            "grass": ELO_DEFAULT,
        }
    )

    player_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "last_rank": 0, "last_rank_points": 0, "last_age": 0, "last_ht": 0})

    n = len(df)
    winner_elo, loser_elo = np.zeros(n), np.zeros(n)
    winner_elo_surface, loser_elo_surface = np.zeros(n), np.zeros(n)
    winner_days, loser_days = np.zeros(n), np.zeros(n)
    winner_rank, loser_rank = np.zeros(n), np.zeros(n)
    winner_rank_points, loser_rank_points = np.zeros(n), np.zeros(n)
    winner_age, loser_age = np.zeros(n), np.zeros(n)
    winner_ht, loser_ht = np.zeros(n), np.zeros(n)

    for idx in range(n):
        row = df.iloc[idx]
        w_id, l_id = row["winner_id"], row["loser_id"]
        surface, date = row["surface"], row["tourney_date"]
        surf_key = surface.lower() if surface in ["Hard", "Clay", "Grass"] else "hard"

        w_elo_stats, l_elo_stats = player_elo[w_id], player_elo[l_id]
        w_pstats, l_pstats = player_stats[w_id], player_stats[l_id]

        winner_elo[idx] = w_elo_stats["overall"]
        loser_elo[idx] = l_elo_stats["overall"]
        winner_elo_surface[idx] = w_elo_stats[surf_key]
        loser_elo_surface[idx] = l_elo_stats[surf_key]

        if w_id in player_last_date:
            winner_days[idx] = (date - player_last_date[w_id]).days
        if l_id in player_last_date:
            loser_days[idx] = (date - player_last_date[l_id]).days

        winner_rank[idx] = w_pstats["last_rank"] or row.get("winner_rank", 500)
        loser_rank[idx] = l_pstats["last_rank"] or row.get("loser_rank", 500)
        winner_rank_points[idx] = w_pstats["last_rank_points"] or row.get("winner_rank_points", 0)
        loser_rank_points[idx] = l_pstats["last_rank_points"] or row.get("loser_rank_points", 0)
        winner_age[idx] = w_pstats["last_age"] or row.get("winner_age", 25)
        loser_age[idx] = l_pstats["last_age"] or row.get("loser_age", 25)
        winner_ht[idx] = w_pstats["last_ht"] or row.get("winner_ht", 180)
        loser_ht[idx] = l_pstats["last_ht"] or row.get("loser_ht", 180)

        player_last_date[w_id] = date
        player_last_date[l_id] = date

        w_elo, l_elo = w_elo_stats["overall"], l_elo_stats["overall"]
        exp_winner = 1 / (1 + 10 ** ((l_elo - w_elo) / 400))
        exp_loser = 1 - exp_winner
        w_elo_stats["overall"] = w_elo + ELO_K_FACTOR * (1 - exp_winner)
        l_elo_stats["overall"] = l_elo + ELO_K_FACTOR * (0 - exp_loser)

        w_elo_surf, l_elo_surf = w_elo_stats[surf_key], l_elo_stats[surf_key]
        exp_winner_surf = 1 / (1 + 10 ** ((l_elo_surf - w_elo_surf) / 400))
        exp_loser_surf = 1 - exp_winner_surf
        w_elo_stats[surf_key] = w_elo_surf + ELO_K_FACTOR * (1 - exp_winner_surf)
        l_elo_stats[surf_key] = l_elo_surf + ELO_K_FACTOR * (0 - exp_loser_surf)

        w_pstats["wins"] += 1
        l_pstats["losses"] += 1
        w_pstats["last_rank"] = row.get("winner_rank", w_pstats["last_rank"])
        w_pstats["last_rank_points"] = row.get("winner_rank_points", w_pstats["last_rank_points"])
        w_pstats["last_age"] = row.get("winner_age", w_pstats["last_age"])
        w_pstats["last_ht"] = row.get("winner_ht", w_pstats["last_ht"])
        l_pstats["last_rank"] = row.get("loser_rank", l_pstats["last_rank"])
        l_pstats["last_rank_points"] = row.get("loser_rank_points", l_pstats["last_rank_points"])
        l_pstats["last_age"] = row.get("loser_age", l_pstats["last_age"])
        l_pstats["last_ht"] = row.get("loser_ht", l_pstats["last_ht"])

    df["winner_elo"] = winner_elo
    df["loser_elo"] = loser_elo
    df["winner_elo_surface"] = winner_elo_surface
    df["loser_elo_surface"] = loser_elo_surface
    df["days_since_last_match_winner"] = winner_days
    df["days_since_last_match_loser"] = loser_days
    df["winner_rest_quality"] = _compute_rest_quality(df["days_since_last_match_winner"])
    df["loser_rest_quality"] = _compute_rest_quality(df["days_since_last_match_loser"])
    df["winner_rank"] = winner_rank
    df["loser_rank"] = loser_rank
    df["winner_rank_points"] = winner_rank_points
    df["loser_rank_points"] = loser_rank_points
    df["winner_age"] = winner_age
    df["loser_age"] = loser_age
    df["winner_ht"] = winner_ht
    df["loser_ht"] = loser_ht

    for p_id, ps in player_stats.items():
        total = ps["wins"] + ps["losses"]
        ps["win_rate"] = ps["wins"] / max(total, 1)

    df["winner_win_rate"] = df["winner_id"].map(lambda x: player_stats[x]["win_rate"])
    df["loser_win_rate"] = df["loser_id"].map(lambda x: player_stats[x]["win_rate"])

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
    "win_rate",
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
        df_winner = _map_player_features(df.copy(), "winner", 1)
        df_loser = _map_player_features(df.copy(), "loser", 0)

        df_features = pd.concat([df_winner, df_loser], ignore_index=True)

        df_features["elo_diff"] = df_features["player_elo"] - df_features["opponent_elo"]
        df_features["elo_surface_diff"] = df_features["player_elo_surface"] - df_features["opponent_elo_surface"]
        df_features["rank_points_diff"] = df_features["player_rank_points"] - df_features["opponent_rank_points"]
        df_features["rank_diff"] = df_features["opponent_rank"] - df_features["player_rank"]
        df_features["age_diff"] = df_features["player_age"] - df_features["opponent_age"]
        df_features["seed_diff"] = df_features["player_seed"] - df_features["opponent_seed"]
        df_features["height_diff"] = df_features["player_ht"] - df_features["opponent_ht"]
        df_features["days_since_last_diff"] = df_features["days_since_last"] - df_features["opponent_days_since_last"]
        df_features["rest_quality_diff"] = df_features["player_rest_quality"] - df_features["opponent_rest_quality"]
        df_features["win_rate_diff"] = df_features["player_win_rate"].fillna(0.5) - df_features["opponent_win_rate"].fillna(0.5)

        MAX_REST_DAYS = 21
        df_features["days_since_last_diff"] = df_features["days_since_last_diff"].clip(-MAX_REST_DAYS, MAX_REST_DAYS)
        df_features["rest_quality_diff"] = df_features["rest_quality_diff"].clip(-1.0, 1.0)
        df_features["age_diff"] = df_features["age_diff"].clip(-15, 15)
        df_features["height_diff"] = df_features["height_diff"].clip(-30, 30)
        df_features["win_rate_diff"] = df_features["win_rate_diff"].clip(-0.5, 0.5)

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
