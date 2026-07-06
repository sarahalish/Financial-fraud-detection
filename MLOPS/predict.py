"""Score new raw data with the trained model bundle.

Takes raw transactions + clients CSVs (same schema as the training files)
and writes a scored CSV with final_fraud_risk, alert flags, and the
peer-deviation-refined priority queue.

Usage (from the project root, with the venv active):
    python predict.py --transactions data/raw/transactions_with_fatf_ofac.csv \
                      --clients data/raw/clients_with_fatf_ofac.csv \
                      --output reports/scored_new_data.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.inference import load_bundle, refined_alert_queue, score_transactions
from src.logger import get_logger


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    logger = get_logger("inference", cfg["paths"]["logs_dir"])

    bundle_path = Path(cfg["paths"]["models_dir"]) / cfg["artifacts"]["model_bundle"]
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"No trained model at {bundle_path}. Run `python run_pipeline.py` first."
        )
    bundle = load_bundle(bundle_path)

    transactions_df = pd.read_csv(args.transactions)
    clients_df = pd.read_csv(args.clients)

    scored = score_transactions(transactions_df, clients_df, bundle)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output, index=False)
    logger.info("Scored data written to %s", output)

    # Show the refined priority queue for compliance review
    queue = refined_alert_queue(
        scored, cfg["scoring"]["peer_deviation_threshold"]
    )
    logger.info("Refined alert queue: %d high-priority cases", len(queue))
    cols = ["transaction_id", "client_id", "amount", "sector",
            "final_fraud_risk", "peer_deviation", "refined_priority"]
    cols = [c for c in cols if c in queue.columns]
    print("\n--- Top 10 highest-priority alerts ---")
    print(queue[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score new raw AML data.")
    parser.add_argument("--transactions", required=True, help="Path to transactions CSV.")
    parser.add_argument("--clients", required=True, help="Path to clients CSV.")
    parser.add_argument("--output", default="reports/scored_new_data.csv",
                        help="Where to write the scored CSV.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args)
