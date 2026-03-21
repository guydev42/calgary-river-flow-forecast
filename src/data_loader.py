"""
Data loader for Bow River Flow Forecasting project.

Fetches river level and flow data from Calgary Open Data portal,
caches locally, and engineers features for time series modeling.
"""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sodapy import Socrata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_ID = "5fdg-ifgr"
DOMAIN = "data.calgary.ca"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_FILE = DATA_DIR / "river_flow_raw.csv"
PROCESSED_FILE = DATA_DIR / "river_flow_processed.csv"
DEFAULT_LIMIT = 50_000  # 9.5M records total; cap for practicality


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
def fetch_river_data(limit: int = DEFAULT_LIMIT, force: bool = False) -> pd.DataFrame:
    """Fetch river level and flow data from Calgary Open Data.

    The full dataset contains approximately 9.5 million records at 5-minute
    intervals.  To keep things practical during development the fetch is
    capped at *limit* rows (default 50 000).

    Parameters
    ----------
    limit : int
        Maximum number of records to retrieve from the API.
    force : bool
        When *True* the local cache is ignored and data is re-downloaded.

    Returns
    -------
    pd.DataFrame
        Raw data as returned by the Socrata API.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_FILE.exists() and not force:
        logger.info("Loading cached raw data from %s", CACHE_FILE)
        return pd.read_csv(CACHE_FILE)

    logger.info(
        "Fetching up to %s records from Calgary Open Data (dataset %s) ...",
        f"{limit:,}",
        DATASET_ID,
    )

    try:
        client = Socrata(DOMAIN, None, timeout=60)
        results = client.get(DATASET_ID, limit=limit, order="timestamp DESC")
        client.close()

        df = pd.DataFrame.from_records(results)
        df.to_csv(CACHE_FILE, index=False)
        logger.info("Fetched and cached %s rows to %s", f"{len(df):,}", CACHE_FILE)
    except Exception as exc:
        logger.error("Failed to fetch data from Socrata API: %s", exc)
        if CACHE_FILE.exists():
            logger.warning("Falling back to cached data.")
            return pd.read_csv(CACHE_FILE)
        raise
    return df


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw river data and engineer base temporal features.

    Steps
    -----
    1. Parse the *timestamp* column to ``datetime``.
    2. Extract hour, day-of-week, month, and year.
    3. Convert *level* and *flow_rate* to numeric types.
    4. Sort by timestamp and drop duplicates.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data (as returned by :func:`fetch_river_data`).

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with additional temporal columns.
    """
    df = df.copy()

    # --- Timestamp -----------------------------------------------------------
    timestamp_col = None
    for candidate in ("timestamp", "date", "datetime"):
        if candidate in df.columns:
            timestamp_col = candidate
            break

    if timestamp_col is None:
        raise KeyError(
            "Could not find a timestamp column. "
            f"Available columns: {list(df.columns)}"
        )

    df["timestamp"] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)

    # --- Temporal features ---------------------------------------------------
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year

    # --- Numeric conversions -------------------------------------------------
    for col in ("level", "flow_rate"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Housekeeping --------------------------------------------------------
    df.sort_values("timestamp", inplace=True)
    df.drop_duplicates(subset=["timestamp"], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ---------------------------------------------------------------------------
# Daily resampling & feature engineering
# ---------------------------------------------------------------------------
def resample_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Resample preprocessed data to daily frequency.

    Aggregates 5-minute observations into daily mean values for
    *level* and *flow_rate*, then creates rolling-average and lag
    features suitable for time-series modelling.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed data (output of :func:`preprocess`).

    Returns
    -------
    pd.DataFrame
        Daily-frequency dataframe with engineered features.
    """
    df = df.copy()
    df.set_index("timestamp", inplace=True)

    numeric_cols = [c for c in ("level", "flow_rate") if c in df.columns]
    if not numeric_cols:
        raise ValueError("Neither 'level' nor 'flow_rate' found in dataframe.")

    daily = df[numeric_cols].resample("D").mean()
    daily.dropna(how="all", inplace=True)

    # --- Rolling averages ----------------------------------------------------
    for col in numeric_cols:
        daily[f"{col}_rolling_7d"] = daily[col].rolling(window=7, min_periods=1).mean()
        daily[f"{col}_rolling_30d"] = daily[col].rolling(window=30, min_periods=1).mean()

    # --- Lag features --------------------------------------------------------
    lag_days = [1, 7, 14, 30]
    for col in numeric_cols:
        for lag in lag_days:
            daily[f"{col}_lag_{lag}d"] = daily[col].shift(lag)

    # --- Re-derive calendar features from the daily index --------------------
    daily["day_of_week"] = daily.index.dayofweek
    daily["month"] = daily.index.month
    daily["year"] = daily.index.year

    daily.dropna(inplace=True)
    daily.reset_index(inplace=True)
    daily.rename(columns={"timestamp": "date"}, inplace=True)

    return daily


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------
def load_and_prepare(
    limit: int = DEFAULT_LIMIT,
    force_download: bool = False,
    save_processed: bool = True,
) -> pd.DataFrame:
    """End-to-end convenience loader.

    Fetches (or loads from cache), preprocesses, resamples to daily
    frequency, and engineers features ready for modelling.

    Parameters
    ----------
    limit : int
        Maximum records to fetch from the API.
    force_download : bool
        If *True* bypass the local CSV cache.
    save_processed : bool
        When *True* persist the processed daily dataframe as a CSV.

    Returns
    -------
    pd.DataFrame
        Processed daily dataframe with all engineered features.
    """
    raw_df = fetch_river_data(limit=limit, force=force_download)
    clean_df = preprocess(raw_df)
    daily_df = resample_daily(clean_df)

    if save_processed:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        daily_df.to_csv(PROCESSED_FILE, index=False)
        logger.info("Saved processed data to %s", PROCESSED_FILE)

    return daily_df


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_and_prepare()
    print(f"Processed data shape: {df.shape}")
    print(df.head())
