import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import Settings
from src.features import extract_features
from src.inference import InferenceService, PredictionError
from src.preprocessing import ArtifactBundle
from src.validation import validate_input


class FakeTransformer:
    def transform(self, values):
        return np.asarray(values, dtype=float)


class FakeCategoricalImputer:
    def transform(self, values):
        return np.asarray(values, dtype=object)


class FakeEncoder:
    def transform(self, values):
        return np.zeros((len(values), 1), dtype=float)


class FakeModel:
    def predict_proba(self, values):
        return np.tile([0.2, 0.8], (len(values), 1))


class ExplodingModel:
    def predict_proba(self, values):
        raise RuntimeError("boom")


def settings(tmp_path: Path) -> Settings:
    return Settings(
        root_dir=tmp_path,
        values={
            "service": {
                "model_version": "test",
                "prediction_log_file": "predictions.jsonl",
                "max_batch_size": 2,
            },
            "data": {
                "required_columns": [
                    "total_price",
                    "total_freight",
                    "items_count",
                    "unique_products",
                    "unique_sellers",
                    "total_payment_value",
                    "max_payment_installments",
                    "payment_types_count",
                    "customer_state",
                    "primary_payment_type",
                    "order_purchase_timestamp",
                    "order_estimated_delivery_date",
                ],
                "numeric_ranges": {},
                "categorical_values": {},
                "max_missing_ratio": 0,
            },
        },
    )


def payload() -> dict:
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


FEATURE_NAMES = [
    "total_price",
    "total_freight",
    "items_count",
    "unique_products",
    "unique_sellers",
    "total_payment_value",
    "max_payment_installments",
    "payment_types_count",
    "purchase_hour",
    "purchase_dayofweek",
    "purchase_is_weekend",
    "purchase_month",
    "estimated_shipping_days",
    "approval_delay_hours",
    "freight_ratio",
    "cat",
]


def service(tmp_path: Path, model=None) -> InferenceService:
    artifacts = ArtifactBundle(
        model=model or FakeModel(),
        feature_names=FEATURE_NAMES,
        num_imputer=FakeTransformer(),
        cat_imputer=FakeCategoricalImputer(),
        encoder=FakeEncoder(),
        scaler=FakeTransformer(),
        threshold=0.63,
        model_name="fake",
    )
    return InferenceService(settings(tmp_path), artifacts)


def test_extract_features_matches_notebook_logic():
    result = extract_features(pd.DataFrame([payload()]))
    assert result.loc[0, "purchase_hour"] == 10
    assert result.loc[0, "purchase_is_weekend"] == 0
    assert result.loc[0, "estimated_shipping_days"] == 9
    assert result.loc[0, "freight_ratio"] == pytest.approx(0.1, rel=1e-3)


def test_invalid_category_is_reported():
    bad_payload = payload()
    bad_payload["customer_state"] = "XX"
    result = validate_input(
        pd.DataFrame([bad_payload]),
        {**settings(Path(".")).section("data"), "categorical_values": {"customer_state": ["SP"]}},
    )
    assert not result.valid


def test_prediction_returns_probability_and_saved_log(tmp_path):
    result = service(tmp_path).predict(payload())
    assert result["prediction"] == 1
    assert result["probability"] == 0.8
    log_path = tmp_path / "predictions.jsonl"
    assert log_path.exists()

    logged = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert logged["input"]["order_id"] == "ord-test-0001"
    assert "predicted_at" in logged
    datetime.fromisoformat(logged["predicted_at"])


def test_unexpected_model_failure_becomes_prediction_error_not_crash(tmp_path):
    """An unhandled exception type must be caught and reported as PredictionError, not crash."""
    with pytest.raises(PredictionError):
        service(tmp_path, model=ExplodingModel()).predict(payload())


def test_gx_layer_catches_type_violation_the_pandas_check_misses(tmp_path):
    """A non-numeric string in a numeric field slips past the pandas guard but not GE."""
    data_config = {**settings(tmp_path).section("data"), "column_types": {"total_price": "float64"}}
    bad_payload = payload()
    bad_payload["total_price"] = "not-a-number"

    pandas_result = validate_input(pd.DataFrame([bad_payload]), data_config)
    assert pandas_result.valid

    custom_settings = Settings(
        root_dir=tmp_path,
        values={"service": settings(tmp_path).section("service"), "data": data_config},
    )
    svc = InferenceService(custom_settings, service(tmp_path).artifacts)
    with pytest.raises(ValueError, match="Great Expectations"):
        svc.predict(bad_payload)
