"""MLflow experiment tracking and model registry integration."""

from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient

from .preprocessing import ArtifactBundle, load_artifacts
from .training import TrainingResult

DEFAULT_EXPERIMENT_NAME = "olist-delay-inference"
DEFAULT_REGISTERED_MODEL_NAME = "olist-delay-classifier"
DEFAULT_ALIAS = "champion"


def log_training_run(
    result: TrainingResult,
    artifacts_dir: Path,
    tracking_uri: str,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    registered_model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
    alias: str = DEFAULT_ALIAS,
) -> dict[str, Any]:
    """Log params/metrics/artifacts for this run and point `alias` at the new version."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_param("best_model_name", result.best_model_name)
        model_params = getattr(result.best_model, "get_params", dict)()
        mlflow.log_params({f"model__{key}": value for key, value in model_params.items()})
        mlflow.log_metric("threshold", result.threshold)
        for metric_name, value in result.test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)
        for model_name, metrics in result.val_metrics.items():
            safe_name = model_name.replace(" ", "_")
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"val_{safe_name}_{metric_name}", value)
        mlflow.log_artifacts(str(artifacts_dir), artifact_path="artifacts")
        run_id = run.info.run_id

    model_version = mlflow.register_model(f"runs:/{run_id}/artifacts", registered_model_name)
    version = str(model_version.version)
    MlflowClient().set_registered_model_alias(registered_model_name, alias, version)

    return {
        "run_id": run_id,
        "registered_model_name": registered_model_name,
        "version": version,
        "alias": alias,
    }


def load_artifacts_from_registry(
    tracking_uri: str,
    registered_model_name: str,
    files: dict[str, str],
    version_or_alias: str = DEFAULT_ALIAS,
) -> ArtifactBundle:
    """Load a registered model version's artifacts from MLflow's artifact store.

    `version_or_alias` is either a version number or an alias name (e.g. "champion").
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    version_or_alias = str(version_or_alias)
    if version_or_alias.isdigit():
        model_version = client.get_model_version(registered_model_name, version_or_alias)
    else:
        model_version = client.get_model_version_by_alias(registered_model_name, version_or_alias)

    local_dir = mlflow.artifacts.download_artifacts(
        run_id=model_version.run_id, artifact_path="artifacts"
    )
    return load_artifacts(Path(local_dir), files)
