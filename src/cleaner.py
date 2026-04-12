"""
Data cleaning module for Tennis Match Prediction Model
"""

import pandas as pd

from config import logger
from exceptions import DataCleanError


# Critical columns that must exist for a valid match
CRITICAL_COLS = [
    "winner_rank",
    "loser_rank",
    "winner_age",
    "loser_age",
    "winner_rank_points",
    "loser_rank_points",
    "surface",
]

# Match statistics columns (missing indicates incomplete match)
STAT_COLS = [
    "w_ace",
    "w_svpt",
    "w_1stIn",
    "w_bpSaved",
    "w_bpFaced",
    "l_ace",
    "l_svpt",
    "l_1stIn",
    "l_bpSaved",
    "l_bpFaced",
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data:
    - Remove matches with incomplete outcomes
    - Remove matches with missing critical stats
    - Handle data type conversions

    Args:
        df: Raw DataFrame from loader

    Returns:
        pd.DataFrame: Cleaned DataFrame

    Raises:
        DataCleanError: If cleaning fails critically
    """
    logger.info("Cleaning data")

    try:
        initial_rows = len(df)

        # Convert tourney_date to datetime
        df["tourney_date"] = pd.to_datetime(
            df["tourney_date"], format="%Y%m%d", errors="coerce"
        )
        df["year"] = df["tourney_date"].dt.year

        # Check for valid dates
        if df["year"].isna().sum() > len(df) * 0.5:
            raise DataCleanError("More than 50% of dates are invalid")

        # Drop rows with missing critical columns
        df = df.dropna(subset=CRITICAL_COLS)

        # Drop rows with missing match stats (incomplete match)
        df = df.dropna(subset=STAT_COLS)

        # Drop matches with invalid scores
        df = df[df["score"].notna()]

        # Ensure numeric columns are proper types
        numeric_cols = [
            "winner_rank",
            "loser_rank",
            "winner_age",
            "loser_age",
            "winner_rank_points",
            "loser_rank_points",
            "winner_ht",
            "loser_ht",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Remove rows with invalid required numeric values
        required_numeric = [
            "winner_rank",
            "loser_rank",
            "winner_age",
            "loser_age",
            "winner_rank_points",
            "loser_rank_points",
        ]
        df = df.dropna(subset=required_numeric)

        rows_removed = initial_rows - len(df)
        removed_pct = (rows_removed / initial_rows * 100) if initial_rows > 0 else 0

        logger.info(f"Rows removed: {rows_removed} ({removed_pct:.1f}%)")
        logger.info(f"Remaining rows: {len(df)}")

        if len(df) == 0:
            raise DataCleanError("All rows were removed during cleaning")

        return df

    except DataCleanError:
        raise
    except Exception as e:
        raise DataCleanError(f"Unexpected error during data cleaning: {e}") from e
