import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .expectations import GreatExpectationsValidator
from .logging_utils import append_jsonl
from .monitoring import record_error, record_prediction
from .preprocessing import ArtifactBundle, transform_for_model
from .validation import validate_input


class PredictionError(RuntimeError):
    """Raised for internal failures unrelated to invalid client input."""


class InferenceService:
    def __init__(self, settings: Any, artifacts: ArtifactBundle):
        self.settings = settings
        self.artifacts = artifacts
        service_config = settings.section("service")
        self.logger = logging.getLogger(__name__)
        self.prediction_log = Path(settings.root_dir) / service_config["prediction_log_file"]
        self.gx_validator = GreatExpectationsValidator(settings.section("data"))

    @property
    def model_info(self) -> dict[str, Any]:
        return {
            "name": self.artifacts.model_name,
            "version": self.settings.section("service")["model_version"],
            "threshold": self.artifacts.threshold,
            "features": len(self.artifacts.feature_names),
        }

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()

        def elapsed() -> float:
            return time.perf_counter() - started

        try:
            frame = pd.DataFrame([payload])
            result = validate_input(frame, self.settings.section("data"))
            if not result.valid:
                record_error("validation", elapsed())
                raise ValueError("Input validation failed: " + "; ".join(result.errors))
            gx_result = self.gx_validator.validate(frame)
            if not gx_result.valid:
                record_error("gx_validation", elapsed())
                raise ValueError(
                    "Great Expectations validation failed: " + "; ".join(gx_result.errors)
                )
            try:
                processed = transform_for_model(frame, self.artifacts)
                probability = float(self.artifacts.model.predict_proba(processed)[:, 1][0])
            except (KeyError, TypeError, ValueError) as error:
                record_error("processing", elapsed())
                raise ValueError(f"Prediction processing failed: {error}") from error
        except ValueError:
            raise
        except Exception as error:
            record_error("unexpected", elapsed())
            self.logger.error("unexpected prediction failure: %s", error, exc_info=True)
            raise PredictionError(f"Prediction failed unexpectedly: {error}") from error
        prediction = int(probability >= self.artifacts.threshold)
        duration_ms = elapsed() * 1000
        output = {
            "prediction": prediction,
            "label": "delayed" if prediction else "on_time",
            "probability": probability,
            "model_version": self.settings.section("service")["model_version"],
            "latency_ms": round(duration_ms, 3),
        }
        append_jsonl(
            self.prediction_log,
            {
                "predicted_at": datetime.now(timezone.utc).isoformat(),
                "input": payload,
                "output": output,
                "model": self.model_info,
            },
        )
        record_prediction(output["label"], duration_ms / 1000)
        self.logger.info(
            "prediction completed latency_ms=%.3f prediction=%s", duration_ms, prediction
        )
        return output

    def predict_batch(self, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        maximum = int(self.settings.section("service")["max_batch_size"])
        if not payloads or len(payloads) > maximum:
            raise ValueError(f"batch size must be between 1 and {maximum}")
        return [self.predict(payload) for payload in payloads]
