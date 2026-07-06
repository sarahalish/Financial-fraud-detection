"""Evaluation against a proxy ground truth.

No verified compliance labels exist for this synthetic dataset, so we use
the same proxy rule as the original notebook:

    fraud := (structuring OR rapid movement)
             AND (PEP OR sanctions OR FATF-listed country)

Replace this with verified labels from a compliance review backend when
they become available.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — safe on any machine, no GUI needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from src.logger import get_logger

logger = get_logger(__name__)


def build_proxy_labels(df: pd.DataFrame) -> pd.Series:
    """Proxy ground truth (see module docstring)."""
    return pd.Series(
        np.where(
            ((df["structuring_pattern_flag"] == 1) | (df["rapid_movement_flag"] == 1))
            & (
                (df["pep_flag"] == 1)
                | (df["sanctions_flag"] == 1)
                | (df["fatf_country_flag_tx"] == 1)
            ),
            1,
            0,
        ),
        index=df.index,
        name="true_fraud_label",
    )


def evaluate(scored_df: pd.DataFrame, reports_dir: str, threshold: float) -> dict:
    """Compute metrics, save a JSON report and a diagnostics figure."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    y_true = build_proxy_labels(scored_df)
    y_pred = scored_df["is_fraud_alert"]
    y_scores = scored_df["final_fraud_risk"]

    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "pr_auc": round(auc(recall_vals, precision_vals), 4),
        "n_alerts": int(y_pred.sum()),
        "n_proxy_fraud": int(y_true.sum()),
        "alert_threshold": round(threshold, 4),
    }
    logger.info("Evaluation metrics: %s", metrics)

    with open(reports_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    _plot_diagnostics(y_true, y_pred, y_scores, scored_df, threshold, reports_dir)
    return metrics


def _plot_diagnostics(y_true, y_pred, y_scores, df, threshold, reports_dir: Path):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False,
        xticklabels=["Normal", "Fraud Alert"],
        yticklabels=["True Normal", "True Fraud"],
    )
    axes[0].set_title("Confusion Matrix")
    axes[0].set_ylabel("Proxy Compliance Label")
    axes[0].set_xlabel("Model Prediction")

    # PR curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall_vals, precision_vals)
    axes[1].plot(recall_vals, precision_vals, color="darkred", lw=2,
                 label=f"PR AUC = {pr_auc:.3f}")
    axes[1].fill_between(recall_vals, precision_vals, alpha=0.1, color="darkred")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")

    # Score distribution
    sns.kdeplot(data=df[y_true == 0], x="final_fraud_risk", fill=True,
                color="green", label="Normal", ax=axes[2], alpha=0.4)
    sns.kdeplot(data=df[y_true == 1], x="final_fraud_risk", fill=True,
                color="red", label="Proxy fraud", ax=axes[2], alpha=0.6)
    axes[2].axvline(threshold, color="black", linestyle="--", label="Alert threshold")
    axes[2].set_title("Fraud Risk Score Distribution")
    axes[2].legend()

    plt.tight_layout()
    out = reports_dir / "evaluation_plots.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("Diagnostics figure saved to %s", out)
