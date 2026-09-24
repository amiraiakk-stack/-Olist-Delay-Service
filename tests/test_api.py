from fastapi.testclient import TestClient

from app.main import app, service

client = TestClient(app)


def request_payload() -> dict:
    return {
        "order_id": "ord-test-0001",
        "total_price": 100.0,
        "total_freight": 10.0,
        "items_count": 1,
        "unique_products": 1,
        "unique_sellers": 1,
        "total_payment_value": 110.0,
        "max_payment_installments": 1,
        "payment_types_count": 1,
        "customer_state": "SP",
        "primary_payment_type": "credit_card",
        "order_purchase_timestamp": "2018-01-01T10:00:00",
        "order_estimated_delivery_date": "2018-01-10T10:00:00",
    }


def test_health_and_model_routes():
    assert client.get("/health").status_code == 200
    response = client.get("/model")
    assert response.status_code == 200
    assert "version" in response.json()


def test_predict_rejects_invalid_request():
    response = client.post("/predict", json={"total_price": -1})
    assert response.status_code == 422


def test_metrics_route_is_available():
    response = client.get("/metrics")
    assert response.status_code == 200


def test_predict_returns_500_without_leaking_details_on_unexpected_error(monkeypatch):
    def explode(_values):
        raise RuntimeError("boom: internal secret path /etc/whatever")

    monkeypatch.setattr(service.artifacts.model, "predict_proba", explode)

    response = client.post("/predict", json=request_payload())
    assert response.status_code == 500
    assert "boom" not in response.text


def test_openapi_documented_example_is_a_working_request():
    """The example Swagger UI's 'Try it out' shows must actually pass validation."""
    schema = client.get("/openapi.json").json()["components"]["schemas"]["PredictionRequest"]
    example = schema["example"]

    response = client.post("/predict", json=example)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["label"] in ("on_time", "delayed")
    assert body["model_version"]


def test_predict_rejects_unknown_fields():
    response = client.post("/predict", json={**request_payload(), "totall_price": 999})
    assert response.status_code == 422
    assert any(item["type"] == "extra_forbidden" for item in response.json()["detail"])


def test_predict_requires_order_id():
    """order_id is required, not optional."""
    payload = request_payload()
    del payload["order_id"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert any(item["loc"] == ["body", "order_id"] for item in response.json()["detail"])


def test_predict_batch_happy_path():
    first, second = request_payload(), {**request_payload(), "order_id": "ord-test-0002"}
    response = client.post("/predict/batch", json={"requests": [first, second]})
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    for prediction in predictions:
        assert prediction["prediction"] in (0, 1)
        assert 0.0 <= prediction["probability"] <= 1.0


def test_predict_batch_rejects_empty_list():
    response = client.post("/predict/batch", json={"requests": []})
    assert response.status_code == 422


def test_predict_batch_rejects_batch_larger_than_max_batch_size():
    oversized = [request_payload() for _ in range(1001)]  # config max_batch_size is 1000
    response = client.post("/predict/batch", json={"requests": oversized})
    assert response.status_code == 422
    assert "batch size" in response.json()["detail"]
