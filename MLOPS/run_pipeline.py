"""Full training pipeline: ingest -> preprocess -> train -> evaluate -> save.

Usage (from the project root, with the venv active):
    python run_pipeline.py
    python run_pipeline.py --config config/config.yaml
"""

import argparse
from pathlib import Path

from src.config import load_config
from src.data_ingestion import load_raw_data, merge_data
from src.evaluate import evaluate
from src.logger import get_logger
from src.train import save_bundle, train_model


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    logger = get_logger("pipeline", cfg["paths"]["logs_dir"])
    logger.info("=== AML fraud detection pipeline started ===")

    # 1. Ingestion
    clients_df, transactions_df = load_raw_data(
        cfg["paths"]["clients_csv"], cfg["paths"]["transactions_csv"]
    )
    merged_df = merge_data(
        transactions_df, clients_df,
        merge_key=cfg["data"]["merge_key"],
        suffixes=cfg["data"]["merge_suffixes"],
    )

    # 2. Train (preprocessing happens inside)
    bundle, scored_df = train_model(merged_df, cfg)

    # 3. Save model bundle
    save_bundle(bundle, cfg["paths"]["models_dir"], cfg["artifacts"]["model_bundle"])

    # 4. Save scored training data for audit / dashboards
    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    scored_path = processed_dir / "scored_training_data.csv"
    scored_df.to_csv(scored_path, index=False)
    logger.info("Scored training data saved to %s", scored_path)

    # 5. Evaluate against the proxy ground truth
    if cfg["evaluation"]["enabled"]:
        evaluate(scored_df, cfg["paths"]["reports_dir"], bundle["alert_threshold"])

    logger.info("=== Pipeline finished successfully ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the AML training pipeline.")
    parser.add_argument("--config", default="config/config.yaml",
                        help="Path to the YAML config file.")
    args = parser.parse_args()
    main(args.config)
