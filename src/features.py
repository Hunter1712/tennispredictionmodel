"""
Feature engineering module for Tennis Match Prediction Model.
Handles historical player statistics and feature creation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from config import logger
from exceptions import FeatureEngineeringError

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

    player_wins = defaultdict(
        lambda: {
            "wins": 0,
            "matches": 0,
            "last_date": None,
            "streak": 0,
            "recent_wins": [],
            "vs_top10_wins": 0,
            "vs_top10_matches": 0,
            "tournament_wins": 0,
            "tournament_matches": 0,
        }
    )
    player_surface = defaultdict(
        lambda: {
            "hard": {"w": 0, "m": 0},
            "clay": {"w": 0, "m": 0},
            "grass": {"w": 0, "m": 0},
        }
    )
    # Head-to-head: player_h2h[player_id][opponent_id] = {"w": wins, "m": matches}
    player_h2h = defaultdict(lambda: defaultdict(lambda: {"w": 0, "m": 0}))

    # Elo ratings: player_elo[player_id] = {"overall": elo, "hard": elo, "clay": elo, "grass": elo}
    player_elo = defaultdict(
        lambda: {
            "overall": ELO_DEFAULT,
            "hard": ELO_DEFAULT,
            "clay": ELO_DEFAULT,
            "grass": ELO_DEFAULT,
        }
    )

    # Pre-allocate arrays
    n = len(df)
    winner_wr, loser_wr = np.zeros(n), np.zeros(n)
    winner_recent, loser_recent = np.zeros(n), np.zeros(n)
    winner_surface_skill, loser_surface_skill = np.zeros(n), np.zeros(n)
    winner_streak, loser_streak = np.zeros(n), np.zeros(n)
    winner_days, loser_days = np.zeros(n), np.zeros(n)
    winner_vs_top10, loser_vs_top10 = np.zeros(n), np.zeros(n)
    winner_h2h, loser_h2h = np.zeros(n), np.zeros(n)
    winner_tournament_wr, loser_tournament_wr = np.zeros(n), np.zeros(n)
    winner_elo, loser_elo = np.zeros(n), np.zeros(n)
    winner_elo_surface, loser_elo_surface = np.zeros(n), np.zeros(n)

    for idx in range(n):
        row = df.iloc[idx]
        w_id, l_id = row["winner_id"], row["loser_id"]
        surface, date = row["surface"], row["tourney_date"]
        w_rank = row["winner_rank"]
        l_rank = row["loser_rank"]
        round_val = row.get("round", "R32")
        is_final = round_val in ["SF", "F"] if pd.notna(round_val) else False
        surf_key = surface.lower() if surface in ["Hard", "Clay", "Grass"] else "hard"

        w_stats, l_stats = player_wins[w_id], player_wins[l_id]
        w_elo_stats, l_elo_stats = player_elo[w_id], player_elo[l_id]

        # Elo ratings (from before this match)
        winner_elo[idx] = w_elo_stats["overall"]
        loser_elo[idx] = l_elo_stats["overall"]
        winner_elo_surface[idx] = w_elo_stats[surf_key]
        loser_elo_surface[idx] = l_elo_stats[surf_key]

        # Win rates
        winner_wr[idx] = w_stats["wins"] / max(w_stats["matches"], 1)
        loser_wr[idx] = l_stats["wins"] / max(l_stats["matches"], 1)

        # Recent form (weighted average of recent wins and overall win rate)
        winner_recent[idx] = (
            sum(w_stats["recent_wins"]) / max(len(w_stats["recent_wins"]), 1)
        ) * 0.6 + winner_wr[idx] * 0.4
        loser_recent[idx] = (
            sum(l_stats["recent_wins"]) / max(len(l_stats["recent_wins"]), 1)
        ) * 0.6 + loser_wr[idx] * 0.4

        # Surface skill
        winner_surface_skill[idx] = player_surface[w_id][surf_key]["w"] / max(
            player_surface[w_id][surf_key]["m"], 1
        )
        loser_surface_skill[idx] = player_surface[l_id][surf_key]["w"] / max(
            player_surface[l_id][surf_key]["m"], 1
        )

        # Head-to-head
        winner_h2h[idx] = player_h2h[w_id][l_id]["w"] / max(
            player_h2h[w_id][l_id]["m"], 1
        )
        loser_h2h[idx] = player_h2h[l_id][w_id]["w"] / max(
            player_h2h[l_id][w_id]["m"], 1
        )

        # vs Top 10 record (rank <= 10)
        winner_vs_top10[idx] = w_stats["vs_top10_wins"] / max(
            w_stats["vs_top10_matches"], 1
        )
        loser_vs_top10[idx] = l_stats["vs_top10_wins"] / max(
            l_stats["vs_top10_matches"], 1
        )

        # Tournament win rate (wins in finals/semifinals)
        winner_tournament_wr[idx] = w_stats["tournament_wins"] / max(
            w_stats["tournament_matches"], 1
        )
        loser_tournament_wr[idx] = l_stats["tournament_wins"] / max(
            l_stats["tournament_matches"], 1
        )

        # Streak and rest days
        winner_streak[idx], loser_streak[idx] = w_stats["streak"], l_stats["streak"]
        if w_stats["last_date"]:
            winner_days[idx] = (date - w_stats["last_date"]).days
        if l_stats["last_date"]:
            loser_days[idx] = (date - l_stats["last_date"]).days

        # Update stats for winner
        w_stats["matches"] += 1
        w_stats["wins"] += 1
        w_stats["streak"] = max(0, w_stats["streak"]) + 1
        w_stats["recent_wins"].append(1)
        if len(w_stats["recent_wins"]) > 5:
            w_stats["recent_wins"].pop(0)
        w_stats["last_date"] = date

        # Update vs_top10 (winner)
        if l_rank <= 10:
            w_stats["vs_top10_wins"] += 1
            w_stats["vs_top10_matches"] += 1

        # Update tournament stats for winner
        if is_final:
            w_stats["tournament_wins"] += 1
            w_stats["tournament_matches"] += 1

        # Update stats for loser
        l_stats["matches"] += 1
        l_stats["streak"] = min(0, l_stats["streak"]) - 1
        l_stats["recent_wins"].append(0)
        if len(l_stats["recent_wins"]) > 5:
            l_stats["recent_wins"].pop(0)
        l_stats["last_date"] = date

        # Update vs_top10 (loser)
        if w_rank <= 10:
            l_stats["vs_top10_matches"] += 1

        # Update tournament stats for loser
        if is_final:
            l_stats["tournament_matches"] += 1

        # Update surface stats
        player_surface[w_id][surf_key]["m"] += 1
        player_surface[w_id][surf_key]["w"] += 1
        player_surface[l_id][surf_key]["m"] += 1

        # Update head-to-head
        player_h2h[w_id][l_id]["m"] += 1
        player_h2h[w_id][l_id]["w"] += 1
        player_h2h[l_id][w_id]["m"] += 1

        # Update Elo ratings (after all stats are read, before moving to next match)
        w_elo, l_elo = w_elo_stats["overall"], l_elo_stats["overall"]

        # Expected scores (winner expected to win, loser expected to lose)
        exp_winner = 1 / (1 + 10 ** ((l_elo - w_elo) / 400))
        exp_loser = 1 - exp_winner

        # Update overall Elo
        w_elo_stats["overall"] = w_elo + ELO_K_FACTOR * (1 - exp_winner)
        l_elo_stats["overall"] = l_elo + ELO_K_FACTOR * (0 - exp_loser)

        # Update surface-specific Elo
        w_elo_surf, l_elo_surf = w_elo_stats[surf_key], l_elo_stats[surf_key]
        exp_winner_surf = 1 / (1 + 10 ** ((l_elo_surf - w_elo_surf) / 400))
        exp_loser_surf = 1 - exp_winner_surf

        w_elo_stats[surf_key] = w_elo_surf + ELO_K_FACTOR * (1 - exp_winner_surf)
        l_elo_stats[surf_key] = l_elo_surf + ELO_K_FACTOR * (0 - exp_loser_surf)

    # Assign computed columns
    df["winner_win_rate"] = winner_wr
    df["loser_win_rate"] = loser_wr
    df["winner_recent_form"] = winner_recent
    df["loser_recent_form"] = loser_recent
    df["winner_surface_skill"] = winner_surface_skill
    df["loser_surface_skill"] = loser_surface_skill
    df["winner_streak"] = winner_streak.astype(int)
    df["loser_streak"] = loser_streak.astype(int)
    df["days_since_last_match_winner"] = winner_days
    df["days_since_last_match_loser"] = loser_days
    df["winner_h2h"] = winner_h2h
    df["loser_h2h"] = loser_h2h
    df["winner_vs_top10_record"] = winner_vs_top10
    df["loser_vs_top10_record"] = loser_vs_top10
    df["winner_tournament_win_rate"] = winner_tournament_wr
    df["loser_tournament_win_rate"] = loser_tournament_wr
    df["winner_elo"] = winner_elo
    df["loser_elo"] = loser_elo
    df["winner_elo_surface"] = winner_elo_surface
    df["loser_elo_surface"] = loser_elo_surface

    # Vectorized rest quality (much faster than apply)
    df["winner_rest_quality"] = _compute_rest_quality(
        df["days_since_last_match_winner"]
    )
    df["loser_rest_quality"] = _compute_rest_quality(df["days_since_last_match_loser"])

    logger.info(f"Features built for {len(df)} matches")
    return df


# Player feature columns to map
_PLAYER_COLS = [
    "rank",
    "rank_points",
    "age",
    "ht",
    "seed",
    "hand",
    "ioc",
    "entry",
    "win_rate",
    "recent_form",
    "surface_skill",
    "h2h",
    "streak",
    "tournament_win_rate",
    "vs_top10_record",
    "rest_quality",
    "elo",
    "elo_surface",
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

        # Compute diff features
        df_features["rank_diff"] = (
            df_features["player_rank"] - df_features["opponent_rank"]
        )
        df_features["rank_points_diff"] = (
            df_features["player_rank_points"] - df_features["opponent_rank_points"]
        )
        df_features["age_diff"] = (
            df_features["player_age"] - df_features["opponent_age"]
        )
        df_features["height_diff"] = (
            df_features["player_ht"] - df_features["opponent_ht"]
        )
        df_features["seed_diff"] = (
            df_features["player_seed"] - df_features["opponent_seed"]
        )

        df_features["win_rate_diff"] = (
            df_features["player_win_rate"] - df_features["opponent_win_rate"]
        )
        df_features["recent_form_diff"] = (
            df_features["player_recent_form"] - df_features["opponent_recent_form"]
        )
        df_features["surface_skill_diff"] = (
            df_features["player_surface_skill"] - df_features["opponent_surface_skill"]
        )
        df_features["streak_diff"] = (
            df_features["player_streak"] - df_features["opponent_streak"]
        )
        df_features["days_since_last_diff"] = (
            df_features["days_since_last"] - df_features["opponent_days_since_last"]
        )
        df_features["tournament_win_rate_diff"] = (
            df_features["player_tournament_win_rate"]
            - df_features["opponent_tournament_win_rate"]
        )
        df_features["vs_top10_record_diff"] = (
            df_features["player_vs_top10_record"]
            - df_features["opponent_vs_top10_record"]
        )
        df_features["h2h_record_diff"] = (
            df_features["player_h2h"] - df_features["opponent_h2h"]
        )
        df_features["rest_quality_diff"] = (
            df_features["player_rest_quality"] - df_features["opponent_rest_quality"]
        )

        # Elo difference features (most important new features!)
        df_features["elo_diff"] = (
            df_features["player_elo"] - df_features["opponent_elo"]
        )
        df_features["elo_surface_diff"] = (
            df_features["player_elo_surface"] - df_features["opponent_elo_surface"]
        )

        # Cap fatigue features to reasonable bounds (reduce overfitting to extreme rest days)
        # Cap to [-21, 21] days (~3 weeks) - beyond that, rest quality is fully recovered/drained
        MAX_REST_DAYS = 21
        df_features["days_since_last_diff"] = df_features["days_since_last_diff"].clip(
            -MAX_REST_DAYS, MAX_REST_DAYS
        )
        df_features["rest_quality_diff"] = df_features["rest_quality_diff"].clip(
            -1.0, 1.0
        )

        # Use win_rate as proxy for level_strength and recent_5_form
        df_features["level_strength_diff"] = (
            df_features["player_win_rate"] - df_features["opponent_win_rate"]
        )
        df_features["recent_5_form_diff"] = (
            df_features["player_recent_form"] - df_features["opponent_recent_form"]
        )

        round_map = {"R128": 1, "R64": 1, "R32": 1, "R16": 2, "QF": 3, "SF": 4, "F": 5}
        df_features["round_enc"] = (
            df_features["round"].fillna("R32").map(round_map).fillna(1).astype(int)
        )
        df_features["draw_size"] = pd.to_numeric(
            df_features["draw_size"], errors="coerce"
        ).fillna(32)

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
