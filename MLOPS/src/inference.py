"""Inference: score new raw transactions with a trained model bundle.

Everything needed for consistent scoring is read from the bundle — the
fitted scaler, the fitted Isolation Forest, the training-time raw-score
min/max (so new scores land on the same 0-1 scale), the alert threshold,
and the per-sector peer baselines. Nothing is re-fitted here.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_ingestion import merge_data
from src.logger import get_logger
from src.preprocessing import build_behavioral_matrix, compute_static_risk

logger = get_logger(__name__)


def load_bundle(bundle_path: str | Path) -> dict:
    bundle = joblib.load(bundle_path)
    logger.info("Loaded model bundle trained at %s", bundle.get("trained_at_utc"))
    return bundle


def score_transactions(
    transactions_df: pd.DataFrame,
    clients_df: pd.DataFrame,
    bundle: dict,
) -> pd.DataFrame:
    """Score new raw transactions. Returns the merged frame with risk columns.

    Output columns added:
        behavioral_risk_score, static_risk_score, final_fraud_risk,
        is_fraud_alert, peer_deviation, refined_priority
    """
    df = merge_data(transactions_df, clients_df)

    # --- Behavioral score (using the FITTED scaler & model) ---------------
    X = build_behavioral_matrix(df, bundle["behavioral_cols"])
    X_scaled = bundle["scaler"].transform(X)
    raw = bundle["model"].score_samples(X_scaled)

    span = bundle["score_max"] - bundle["score_min"]
    behavioral = 1.0 - (raw - bundle["score_min"]) / span
    behavioral = np.clip(behavioral, 0.0, 1.0)  # new data may exceed train range

    # --- Static + ensemble --------------------------------------------------
    static = compute_static_risk(df, bundle["static_cols"])
    w_b = bundle["scoring"]["behavioral_weight"]
    w_s = bundle["scoring"]["static_weight"]

    df["behavioral_risk_score"] = behavioral
    df["static_risk_score"] = static
    df["final_fraud_risk"] = w_b * behavioral + w_s * static.values
    df["is_fraud_alert"] = (df["final_fraud_risk"] >= bundle["alert_threshold"]).astype(int)

    # --- Peer group analysis (training-time sector baselines) ---------------
    peer_col = bundle["scoring"]["peer_group_column"]
    baselines = df[peer_col].map(bundle["sector_baselines"])
    baselines = baselines.fillna(bundle["global_mean_amount"])  # unseen sectors
    df["peer_deviation"] = df["amount"] / baselines
    df["refined_priority"] = df["final_fraud_risk"] * df["peer_deviation"]

    logger.info(
        "Scored %d transactions — %d alerts (%.2f%%)",
        len(df), df["is_fraud_alert"].sum(),
        100 * df["is_fraud_alert"].mean(),
    )
    return df


def refined_alert_queue(scored_df: pd.DataFrame, peer_deviation_threshold: float) -> pd.DataFrame:
    """Alerts whose amount also deviates from sector norms, ranked by priority."""
    alerts = scored_df[
        (scored_df["is_fraud_alert"] == 1)
        & (scored_df["peer_deviation"] > peer_deviation_threshold)
    ]
    return alerts.sort_values("refined_priority", ascending=False)
