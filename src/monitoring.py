from prometheus_client import Counter, Histogram

REQUESTS = Counter("inference_requests_total", "Inference requests", ["endpoint", "status"])
LATENCY = Histogram("inference_latency_seconds", "Inference latency", ["endpoint"])
ERRORS = Counter("inference_errors_total", "Inference errors", ["endpoint", "kind"])
PREDICTIONS = Counter("inference_predictions_total", "Predictions by class", ["label"])


def record_prediction(label: str, latency_seconds: float, endpoint: str = "predict") -> None:
    REQUESTS.labels(endpoint, "success").inc()
    LATENCY.labels(endpoint).observe(latency_seconds)
    PREDICTIONS.labels(label).inc()


def record_error(kind: str, latency_seconds: float, endpoint: str = "predict") -> None:
    REQUESTS.labels(endpoint, "error").inc()
    ERRORS.labels(endpoint, kind).inc()
    LATENCY.labels(endpoint).observe(latency_seconds)
