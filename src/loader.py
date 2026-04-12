"""
Data loading module for Tennis Match Prediction Model
"""

import os
from glob import glob

import pandas as pd

from config import config, logger
from exceptions import DataLoadError


def load_all_csv() -> pd.DataFrame:
    """
    Load all ATP CSV files from data folder and combine into single DataFrame

    Returns:
        pd.DataFrame: Combined dataframe with all matches

    Raises:
        DataLoadError: If no CSV files found or all fail to load
    """
    logger.info("Loading ATP data from CSV files")

    csv_files = sorted(glob(os.path.join(config.DATA_FOLDER, "*.csv")))

    if not csv_files:
        raise DataLoadError(f"No CSV files found in {config.DATA_FOLDER}")

    # Filter to main ATP files only (exclude challenger, amateur, etc.)
    main_files = []
    for f in csv_files:
        basename = os.path.basename(f)
        # Skip non-ATP files
        if any(
            pattern in basename.lower()
            for pattern in ["challenger", "amateur", "ongoing", "atp_database"]
        ):
            continue
        # Only include year-based files (e.g., 2024.csv)
        if basename.replace(".csv", "").isdigit():
            main_files.append(f)

    logger.info(f"Found {len(main_files)} ATP files")

    if not main_files:
        raise DataLoadError(f"No ATP CSV files found in {config.DATA_FOLDER}")

    dfs = []

    for csv_file in main_files:
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            dfs.append(df)
            logger.debug(f"Loaded {os.path.basename(csv_file)}: {len(df)} matches")
        except Exception as e:
            logger.warning(f"Failed to load {csv_file}: {e}")

    if not dfs:
        raise DataLoadError(f"Failed to load any CSV files from {config.DATA_FOLDER}")

    try:
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total ATP matches loaded: {len(combined_df):,}")
        return combined_df
    except Exception as e:
        raise DataLoadError(f"Failed to combine dataframes: {e}") from e


def load_csv_for_year(year: int) -> pd.DataFrame | None:
    """
    Load CSV file for a specific year

    Args:
        year: Year to load (e.g., 2024)

    Returns:
        pd.DataFrame: DataFrame for that year, or None if not found
    """
    csv_path = os.path.join(config.DATA_FOLDER, f"{year}.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, low_memory=False)
    return None


def get_available_years() -> list[int]:
    """Get list of available years in data folder"""
    csv_files = glob(os.path.join(config.DATA_FOLDER, "*.csv"))
    years = set()
    for f in csv_files:
        filename = os.path.basename(f)
        try:
            # Only include numeric year files (not challenger, amateur, etc.)
            if filename.replace(".csv", "").isdigit():
                year = int(filename.replace(".csv", ""))
                if 1960 <= year <= 2030:
                    years.add(year)
        except ValueError, IndexError:
            continue
    return sorted(years)
