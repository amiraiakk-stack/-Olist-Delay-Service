"""Exercises real MLflow tracking/registry code against a local file-store URI, no mocks."""

from pathlib import Path

import pytest

from src.features import extract_features
from src.fixtures import build_synthetic_labeled_dataset
from src.preprocessing import apply_transformers
from src.registry import load_artifacts_from_registry, log_training_run
from src.splitting import time_based_split
from src.training import run_training_pipeline, save_training_artifacts

FILES = {
    "model_file": "final_best_model.joblib",
    "feature_names_file": "feature_names.joblib",
    "num_imputer_file": "num_imputer.joblib",
    "cat_imputer_file": "cat_imputer.joblib",
    "encoder_file": "ohe_encoder.joblib",
    "scaler_file": "scaler.joblib",
}


@pytest.fixture
def trained_result():
    labeled = build_synthetic_labeled_dataset(rows=600)
    train_df, val_df, test_df = time_based_split(labeled)
    return run_training_pipeline(train_df, val_df, test_df)


def test_log_training_run_registers_a_model_with_a_version_and_alias(tmp_path, trained_result):
    artifacts_dir = tmp_path / "artifacts"
    save_training_artifacts(trained_result, artifacts_dir)
    tracking_uri = Path(tmp_path / "mlruns").as_uri()

    registration = log_training_run(
        trained_result,
        artifacts_dir,
        tracking_uri,
        experiment_name="test-experiment",
        registered_model_name="test-olist-model",
        alias="champion",
    )

    assert registration["version"] == "1"
    assert registration["alias"] == "champion"
    assert registration["registered_model_name"] == "test-olist-model"


def test_load_artifacts_from_registry_matches_locally_saved_artifacts(tmp_path, trained_result):
    """Loading from the registry must match loading the same artifacts from disk."""
    artifacts_dir = tmp_path / "artifacts"
    save_training_artifacts(trained_result, artifacts_dir)
    tracking_uri = Path(tmp_path / "mlruns").as_uri()

    log_training_run(
        trained_result,
        artifacts_dir,
        tracking_uri,
        experiment_name="test-experiment",
        registered_model_name="test-olist-model",
        alias="champion",
    )

    from_registry = load_artifacts_from_registry(
        tracking_uri, "test-olist-model", FILES, version_or_alias="champion"
    )

    assert from_registry.threshold == pytest.approx(trained_result.threshold)
    assert from_registry.feature_names == trained_result.transformers.feature_names

    sample = build_synthetic_labeled_dataset(rows=3, seed=123)
    processed = apply_transformers(extract_features(sample), trained_result.transformers)
    registry_probs = from_registry.model.predict_proba(processed)
    local_probs = trained_result.best_model.predict_proba(processed)
    assert registry_probs.tolist() == local_probs.tolist()


def test_load_artifacts_from_registry_accepts_explicit_version(tmp_path, trained_result):
    artifacts_dir = tmp_path / "artifacts"
    save_training_artifacts(trained_result, artifacts_dir)
    tracking_uri = Path(tmp_path / "mlruns").as_uri()

    registration = log_training_run(
        trained_result,
        artifacts_dir,
        tracking_uri,
        experiment_name="test-experiment",
        registered_model_name="test-olist-model",
        alias="champion",
    )

    from_registry = load_artifacts_from_registry(
        tracking_uri,
        "test-olist-model",
        FILES,
        version_or_alias=registration["version"],
    )
    assert from_registry.model_name.startswith(trained_result.best_model_name)
