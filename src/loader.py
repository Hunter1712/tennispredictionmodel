"""
Data loading module for Tennis Match Prediction Model.
"""

import os
from glob import glob

import pandas as pd

from .config import config, logger
from .exceptions import DataLoadError


def load_all_csv() -> pd.DataFrame:
    """Load all ATP CSV files from data folder and combine into single DataFrame.

    Returns:
        Combined DataFrame with all matches.

    Raises:
        DataLoadError: If no CSV files found or all fail to load.
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
