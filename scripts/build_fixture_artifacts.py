"""Build a fully-fitted but synthetic saved_artifacts/ directory for tests, local dev, and CI.

NOT the real trained model - see src.fixtures.build_synthetic_labeled_dataset. The real
model comes from scripts/train_pipeline.py against the actual Olist database; never
overwrite a real saved_artifacts/ directory with this script's output.
"""

import argparse
from pathlib import Path

from src.fixtures import build_synthetic_labeled_dataset
from src.splitting import time_based_split
from src.training import run_training_pipeline, save_training_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", default="saved_artifacts")
    parser.add_argument("--rows", type=int, default=600)
    arguments = parser.parse_args()

    labeled = build_synthetic_labeled_dataset(rows=arguments.rows)
    train_df, val_df, test_df = time_based_split(labeled)
    result = run_training_pipeline(train_df, val_df, test_df)
    save_training_artifacts(result, Path(arguments.artifacts_dir))

    print(f"Synthetic fixture artifacts written to {arguments.artifacts_dir}/")
    print("This is NOT a real trained model - see this script's docstring.")


if __name__ == "__main__":
    main()
