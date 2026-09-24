"""End-to-end training pipeline: extraction -> labeling -> split -> train -> save.

Usage: python -m scripts.train_pipeline [--artifacts-dir saved_artifacts]
Requires DB_USER, DB_PASS, DB_NAME (and optionally DB_HOST, DB_PORT) in the environment.
If MLFLOW_TRACKING_URI is set, also logs and registers the run (see src/registry.py).
"""

import argparse
import os
from pathlib import Path

from src.data_extraction import build_engine_from_env, build_ml_dataset, read_source_tables
from src.labeling import add_delay_label, parse_dates
from src.splitting import time_based_split
from src.training import run_training_pipeline, save_training_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full Olist delay-prediction pipeline")
    parser.add_argument("--artifacts-dir", default="saved_artifacts")
    arguments = parser.parse_args()

    engine = build_engine_from_env()
    tables = read_source_tables(engine)
    ml_df = build_ml_dataset(tables)

    labeled_df = add_delay_label(parse_dates(ml_df))
    train_df, val_df, test_df = time_based_split(labeled_df)

    result = run_training_pipeline(train_df, val_df, test_df)
    artifacts_dir = Path(arguments.artifacts_dir)
    save_training_artifacts(result, artifacts_dir)

    print(f"Selected model: {result.best_model_name} (threshold={result.threshold:.2f})")
    print("Test metrics:", result.test_metrics)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        from src.registry import log_training_run

        registration = log_training_run(result, artifacts_dir, tracking_uri)
        print(f"Registered in MLflow: {registration}")


if __name__ == "__main__":
    main()
