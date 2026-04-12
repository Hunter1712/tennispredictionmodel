"""
Feature engineering module for Tennis Match Prediction Model
Handles historical player statistics and feature creation
"""

import numpy as np
import pandas as pd
from collections import defaultdict

from config import logger
from exceptions import FeatureEngineeringError


FEATURE_COLS = [
    "rank_diff",
    "rank_points_diff",
    "age_diff",
    "height_diff",
    "seed_diff",
    "win_rate_diff",
    "recent_form_diff",
    "surface_skill_diff",
    "level_strength_diff",
    "streak_diff",
    "days_since_last_diff",
    "recent_5_form_diff",
    "tournament_win_rate_diff",
    "vs_top10_record_diff",
    "h2h_record_diff",
    "rest_quality_diff",
    "round_enc",
    "draw_size",
]


def build_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Build historical player stats with efficient batch processing."""
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

    # Rest quality based on days since last match
    def rest_score(days):
        if days == 0:
            return 0.0
        if 3 <= days <= 7:
            return 1.0
        if days < 3:
            return 0.5 + (days / 6)
        return max(0.0, 1.0 - ((days - 7) / 30))

    df["winner_rest_quality"] = df["days_since_last_match_winner"].apply(rest_score)
    df["loser_rest_quality"] = df["days_since_last_match_loser"].apply(rest_score)

    logger.info(f"Features built for {len(df)} matches")
    return df


def _map_player_features(df: pd.DataFrame, prefix: str, target: int) -> pd.DataFrame:
    """Map winner/loser columns to player/opponent format for binary classification."""
    df = df.copy()
    df["target"] = target

    # Map player (winner or loser based on prefix)
    for col in [
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
    ]:
        src = f"{prefix}_{col}"
        dst = f"player_{col}"
        if src in df.columns:
            df[dst] = df[src]

    # Map opponent (the other player)
    opp_prefix = "loser" if prefix == "winner" else "winner"
    for col in [
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
    ]:
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
    """Create binary classification features from match data."""
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
    """Select final features and prepare X, y for modeling."""
    X = df_features[FEATURE_COLS].copy()
    y = df_features["target"].copy()
    return X, y, FEATURE_COLS
