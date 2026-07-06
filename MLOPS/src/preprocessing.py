"""Preprocessing & feature engineering.

- Log-transform of transaction amounts (compresses heavy-tailed volumes so
  small structured patterns like smurfing are not drowned out).
- Missing-value handling for the model's feature columns.
- Building the behavioral feature matrix used by the Isolation Forest.
"""

import numpy as np
import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns (currently: amount_log)."""
    df = df.copy()
    df["amount_log"] = np.log1p(df["amount"].clip(lower=0))
    return df


def fill_missing(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Fill NAs in the given columns: median for numeric, mode for categorical."""
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            raise KeyError(
                f"Expected column '{col}' not found. Available: {sorted(df.columns)}"
            )
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_value = df[col].median()
            else:
                fill_value = df[col].mode()[0]
            n = df[col].isna().sum()
            df[col] = df[col].fillna(fill_value)
            logger.info("Filled %d missing values in '%s' with %s", n, col, fill_value)
    return df


def build_behavioral_matrix(df: pd.DataFrame, behavioral_cols: list[str]) -> pd.DataFrame:
    """Return the (unscaled) behavioral feature matrix in a fixed column order."""
    df = engineer_features(df)
    df = fill_missing(df, behavioral_cols)
    X = df[behavioral_cols].astype(float)
    logger.info("Behavioral matrix shape: %s", X.shape)
    return X


def compute_static_risk(df: pd.DataFrame, static_cols: list[str]) -> pd.Series:
    """Static risk score = mean of binary compliance flags (0..1)."""
    df = fill_missing(df, static_cols)
    return df[static_cols].astype(float).sum(axis=1) / len(static_cols)
