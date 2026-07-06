"""Model training.

Fits the behavioral Isolation Forest, computes the ensemble risk score
(0.7 * behavioral + 0.3 * static by default), derives the dynamic alert
threshold and the sector peer baselines, and serializes EVERYTHING needed
for consistent inference into a single joblib bundle:

    - fitted StandardScaler
    - fitted IsolationForest
    - raw-score min/max (so new scores are normalized on the same scale)
    - alert threshold
    - per-sector mean amounts (peer group baselines)
    - the feature lists and scoring weights used at training time
"""

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.logger import get_logger
from src.preprocessing import build_behavioral_matrix, compute_static_risk

logger = get_logger(__name__)


def train_model(merged_df: pd.DataFrame, cfg: dict) -> tuple[dict, pd.DataFrame]:
    """Train the fraud engine. Returns (bundle, scored training dataframe)."""
    behavioral_cols = cfg["features"]["behavioral"]
    static_cols = cfg["features"]["static"]
    if_params = dict(cfg["model"]["isolation_forest"])
    scoring = cfg["scoring"]

    # --- 1. Behavioral engine -------------------------------------------
    X = build_behavioral_matrix(merged_df, behavioral_cols)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info("Fitting IsolationForest with params: %s", if_params)
    model = IsolationForest(**if_params)
    model.fit(X_scaled)

    raw_scores = model.score_samples(X_scaled)
    score_min, score_max = float(raw_scores.min()), float(raw_scores.max())
    behavioral_risk = 1.0 - (raw_scores - score_min) / (score_max - score_min)

    # --- 2. Static risk + ensemble --------------------------------------
    static_risk = compute_static_risk(merged_df, static_cols)
    w_b, w_s = scoring["behavioral_weight"], scoring["static_weight"]
    final_risk = w_b * behavioral_risk + w_s * static_risk.values

    # --- 3. Dynamic alert threshold --------------------------------------
    threshold = float(np.percentile(final_risk, scoring["alert_percentile"]))
    logger.info(
        "Dynamic alert threshold (p%s): %.4f", scoring["alert_percentile"], threshold
    )

    # --- 4. Peer group baselines (sector mean amounts) -------------------
    peer_col = scoring["peer_group_column"]
    sector_baselines = merged_df.groupby(peer_col)["amount"].mean().to_dict()
    global_mean_amount = float(merged_df["amount"].mean())

    # --- 5. Attach scores to the training frame ---------------------------
    scored = merged_df.copy()
    scored["behavioral_risk_score"] = behavioral_risk
    scored["static_risk_score"] = static_risk
    scored["final_fraud_risk"] = final_risk
    scored["is_fraud_alert"] = (scored["final_fraud_risk"] >= threshold).astype(int)
    logger.info("Alerts flagged in training data: %d", scored["is_fraud_alert"].sum())

    # --- 6. Serialize the bundle ------------------------------------------
    bundle = {
        "scaler": scaler,
        "model": model,
        "score_min": score_min,
        "score_max": score_max,
        "alert_threshold": threshold,
        "sector_baselines": sector_baselines,
        "global_mean_amount": global_mean_amount,
        "behavioral_cols": behavioral_cols,
        "static_cols": static_cols,
        "scoring": scoring,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return bundle, scored


def save_bundle(bundle: dict, models_dir: str, filename: str) -> Path:
    """Persist the model bundle with joblib."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / filename
    joblib.dump(bundle, path)
    logger.info("Model bundle saved to %s", path)
    return path
