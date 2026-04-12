"""
Export predictions for GitHub Pages deployment
Uses cached data from main pipeline to generate predictions
"""

import json
import os

import numpy as np
import pandas as pd

from config import config
from model import load_model
from features import FEATURE_COLS
from pipeline import load_cached_data


def get_player_stats(df: pd.DataFrame, name: str) -> dict | None:
    """Get career peak stats for a player."""
    wins = df[df["winner_name"] == name].copy()
    losses = df[df["loser_name"] == name].copy()

    if wins.empty and losses.empty:
        return None

    wins["is_winner"] = True
    losses["is_winner"] = False
    all_matches = pd.concat([wins, losses])

    # Use CAREER PEAK rank points, not latest
    peak_rank_points = max(
        wins["winner_rank_points"].max() if len(wins) > 0 else 0,
        losses["loser_rank_points"].max() if len(losses) > 0 else 0,
    )

    # Get latest match for other stats
    latest = all_matches.sort_values("tourney_date", ascending=False).iloc[0]
    is_winner = latest["is_winner"]
    prefix = "winner" if is_winner else "loser"

    # Calculate overall win rate
    total_wins = len(wins)
    total_losses = len(losses)
    win_rate = total_wins / max(total_wins + total_losses, 1)

    # Recent form (last 10 matches)
    recent = all_matches.sort_values("tourney_date", ascending=False).head(10)
    recent_wins = len(recent[recent["winner_name"] == name])
    recent_form = recent_wins / max(len(recent), 1)

    stats = {
        "rank": int(latest[f"{prefix}_rank"]),
        "rank_points": int(peak_rank_points),  # Use peak, not latest
        "age": float(latest[f"{prefix}_age"]),
        "height": int(latest.get(f"{prefix}_ht", 180) or 180),
        "seed": int(latest.get(f"{prefix}_seed", 0) or 0),
        "win_rate": float(win_rate),
        "recent_form": float(recent_form),
        "surface_skill": 0.5,  # Will be computed from surface stats below
        "streak": 0,
        "matches": len(all_matches),
    }

    # Add surface win rates
    for surf in ["Hard", "Clay", "Grass", "Carpet"]:
        wins_surf = wins[wins["surface"] == surf]
        losses_surf = losses[losses["surface"] == surf]
        total = len(wins_surf) + len(losses_surf)
        stats[f"surface_{surf}"] = len(wins_surf) / max(total, 1)

    # Surface skill = best surface win rate
    stats["surface_skill"] = max(
        stats.get("surface_Hard", 0.5),
        stats.get("surface_Clay", 0.5),
        stats.get("surface_Grass", 0.5),
        stats.get("surface_Carpet", 0.5),
    )

    return stats


def get_features(player_a: dict, player_b: dict) -> dict:
    """Compute features for a matchup."""
    return {
        "rank_diff": player_a["rank"] - player_b["rank"],
        "rank_points_diff": player_a["rank_points"] - player_b["rank_points"],
        "age_diff": player_a["age"] - player_b["age"],
        "height_diff": player_a["height"] - player_b["height"],
        "seed_diff": player_a["seed"] - player_b["seed"],
        "win_rate_diff": player_a["win_rate"] - player_b["win_rate"],
        "recent_form_diff": player_a["recent_form"] - player_b["recent_form"],
        "surface_skill_diff": player_a["surface_skill"] - player_b["surface_skill"],
        "level_strength_diff": player_a["win_rate"] - player_b["win_rate"],
        "streak_diff": player_a["streak"] - player_b["streak"],
        "days_since_last_diff": 0,
        "recent_5_form_diff": player_a["win_rate"] - player_b["win_rate"],
        "tournament_win_rate_diff": player_a["win_rate"] - player_b["win_rate"],
        "vs_top10_record_diff": 0,
        "h2h_record_diff": 0,
        "rest_quality_diff": 0.5,
        "round_enc": 3,
        "draw_size": 32,
    }


def main():
    print("=" * 50)
    print("EXPORTING PREDICTIONS")
    print("=" * 50)

    # Try to load cached data first
    print("\n[1/5] Loading cached data...")
    df = load_cached_data()

    if df is None:
        print("    No cache found! Run main.py first to generate cache.")
        print("    Alternative: run full pipeline manually")
        return

    print(f"    {len(df):,} cached samples loaded")

    # Get top 250 players by rank points (active players only)
    print("\n[2/5] Finding top players (active)...")

    # Filter to recent years only (last 3 years) to get active players
    recent_years = [2024, 2025, 2026]
    recent_df = df[df["year"].isin(recent_years)]

    # Get players active in recent years
    recent_players = set(recent_df["winner_name"].dropna()) | set(
        recent_df["loser_name"].dropna()
    )
    print(f"    {len(recent_players)} active players (2024-2026)")

    # Now get ALL players' career peak rank points (from full dataset)
    all_players = set(df["winner_name"].dropna()) | set(df["loser_name"].dropna())

    player_pts = {}
    for p in all_players:
        # Only consider active players
        if p not in recent_players:
            continue

        # Get max rank points as winner OR loser (career peak)
        wins = df[df["winner_name"] == p]["winner_rank_points"].dropna()
        losses = df[df["loser_name"] == p]["loser_rank_points"].dropna()
        max_pts = 0
        if len(wins) > 0:
            max_pts = max(max_pts, wins.max())
        if len(losses) > 0:
            max_pts = max(max_pts, losses.max())
        player_pts[p] = max_pts

    top_players = sorted(player_pts.items(), key=lambda x: x[1], reverse=True)[:250]
    names = [p[0] for p in top_players]
    print(f"    {len(names)} players selected (active only)")
    print(f"    Top 5: {names[:5]}")

    # Get player stats
    print("\n[3/5] Computing player stats...")
    players = {n: get_player_stats(df, n) for n in names}
    players = {k: v for k, v in players.items() if v}
    print(f"    {len(players)} players with stats")

    # Generate predictions
    print("\n[4/5] Generating predictions...")
    model = load_model()

    # Build feature matrix for all pairs
    pairs = []
    player_list = [n for n in names if n in players]
    for i, a in enumerate(player_list):
        for b in player_list[i + 1 :]:
            pairs.append((a, b, players[a], players[b]))

    print(f"    {len(pairs)} matchups to predict...")

    predictions = {}

    # Batch predict
    if pairs:
        features_arr = np.array(
            [
                [get_features(pa, pb).get(c, 0) for c in FEATURE_COLS]
                for _, _, pa, pb in pairs
            ]
        )
        probs = model.predict_proba(features_arr)[:, 1]

        for idx, (a, b, pa, pb) in enumerate(pairs):
            overall = float(probs[idx])
            pred = {"overall": round(overall, 4)}
            for surf in ["Hard", "Clay", "Grass", "Carpet"]:
                surf_a = float(pa.get(f"surface_{surf}", 0.5))
                surf_b = float(pb.get(f"surface_{surf}", 0.5))
                blend = (
                    surf_a / (surf_a + surf_b + 0.001) if (surf_a + surf_b) > 0 else 0.5
                )
                pred[surf] = round(float(overall) * 0.7 + blend * 0.3, 4)
            predictions[f"{a}|{b}"] = pred

    print(f"    {len(predictions)} predictions generated")

    # Save
    print("\n[5/5] Saving...")
    accuracy = 0.7523  # From model training

    output = {
        "meta": {
            "model": "XGBoost",
            "accuracy": float(accuracy),
            "features": len(FEATURE_COLS),
            "feature_list": FEATURE_COLS,
            "trained_years": "1991-2024",
            "test_years": "2025-2026",
            "predictions_count": len(predictions),
            "players": len(players),
        },
        "players": [{"name": n, **s} for n, s in players.items()],
        "predictions": predictions,
    }

    # Save to output folder (for public repo)
    os.makedirs("../output", exist_ok=True)
    with open("../output/predictions.js", "w") as f:
        f.write("const PREDICTIONS = " + json.dumps(output) + ";")

    print(
        f"\n✓ Done! {len(predictions):,} predictions saved to ../output/predictions.js"
    )


if __name__ == "__main__":
    main()
