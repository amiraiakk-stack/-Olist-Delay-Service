import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    request_to_payload,
)
from src.config import Settings
from src.inference import InferenceService, PredictionError
from src.logging_utils import configure_logging
from src.preprocessing import load_artifacts


def create_service() -> InferenceService:
    root = Path(__file__).resolve().parents[1]
    config_path = Path(os.getenv("INFERENCE_CONFIG", root / "config" / "config.yaml"))
    settings = Settings.from_file(config_path)
    service_config = settings.section("service")
    configure_logging(service_config["log_level"], settings.root_dir / service_config["log_file"])
    model_config = settings.section("model")
    if model_config.get("source", "local") == "mlflow":
        from src.registry import load_artifacts_from_registry

        artifacts = load_artifacts_from_registry(
            tracking_uri=model_config["mlflow_tracking_uri"],
            registered_model_name=model_config["mlflow_registered_model_name"],
            files=model_config,
            version_or_alias=model_config.get("mlflow_model_alias", "champion"),
        )
    else:
        artifacts = load_artifacts(settings.path("model", "artifacts_dir"), model_config)
    return InferenceService(settings, artifacts)


service = create_service()
app = FastAPI(title="Olist Delay Inference API", version=service.model_info["version"])


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness/readiness check: whether the service is up and a model is loaded."""
    return HealthResponse(status="ok", model_loaded=service.artifacts.model is not None)


@app.get("/model", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Name, service version, decision threshold, and feature count of the loaded model."""
    return ModelInfoResponse(**service.model_info)


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus metrics for request counts, latency, errors, and predictions by class."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Score one order and return its delay prediction, probability, and model version."""
    try:
        return PredictionResponse(**service.predict(request_to_payload(request)))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except PredictionError as error:
        raise HTTPException(status_code=500, detail="internal prediction error") from error


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Score a bounded batch of orders (see config's max_batch_size) in one call."""
    try:
        predictions = service.predict_batch([request_to_payload(item) for item in request.requests])
        return BatchPredictionResponse(predictions=predictions)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except PredictionError as error:
        raise HTTPException(status_code=500, detail="internal prediction error") from error
