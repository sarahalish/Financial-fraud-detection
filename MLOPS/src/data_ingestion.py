"""Data ingestion: load raw CSVs and merge clients onto transactions."""

import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


def load_raw_data(clients_path: str, transactions_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw clients and transactions CSVs."""
    clients_df = pd.read_csv(clients_path)
    transactions_df = pd.read_csv(transactions_path)
    logger.info(
        "Loaded %d clients and %d transactions", len(clients_df), len(transactions_df)
    )
    return clients_df, transactions_df


def merge_data(
    transactions_df: pd.DataFrame,
    clients_df: pd.DataFrame,
    merge_key: str = "client_id",
    suffixes: tuple[str, str] = ("_tx", "_client"),
) -> pd.DataFrame:
    """Left-join client profiles onto transactions.

    Columns present in both files (e.g. fatf_country_flag) get the
    suffixes _tx / _client after the merge.
    """
    merged = pd.merge(
        transactions_df, clients_df, on=merge_key, how="left", suffixes=tuple(suffixes)
    )

    orphans = merged["client_name"].isna().sum() if "client_name" in merged else 0
    if orphans:
        logger.warning("%d transactions have no matching client profile", orphans)

    logger.info("Merged dataset shape: %s", merged.shape)
    return merged
