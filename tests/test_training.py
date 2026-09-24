import numpy as np
import pandas as pd
import pytest

from src.config import Settings
from src.features import extract_features
from src.fixtures import build_synthetic_labeled_dataset
from src.inference import InferenceService
from src.labeling import add_delay_label, parse_dates
from src.preprocessing import ArtifactBundle, apply_transformers, fit_transformers, load_artifacts
from src.splitting import time_based_split
from src.training import (
    evaluate_model,
    run_training_pipeline,
    save_training_artifacts,
    tune_threshold,
)


def test_add_delay_label_matches_notebook_logic():
    raw = pd.DataFrame(
        {
            "order_status": ["delivered", "delivered", "canceled"],
            "order_delivered_customer_date": ["2018-01-05", "2018-01-05", "2018-01-05"],
            "order_estimated_delivery_date": ["2018-01-01", "2018-01-10", "2018-01-10"],
        }
    )
    labeled = add_delay_label(parse_dates(raw))
    assert len(labeled) == 2  # non-delivered orders are dropped
    assert labeled["is_delayed"].tolist() == [1, 0]
    assert labeled["delay_days"].iloc[0] == pytest.approx(4.0)


def test_time_based_split_boundaries():
    df = pd.DataFrame(
        {
            "order_purchase_timestamp": pd.to_datetime(
                [
                    "2018-05-31 23:59:59",
                    "2018-06-01 00:00:00",
                    "2018-07-15 23:59:59",
                    "2018-07-16 00:00:00",
                ]
            ),
            "value": [1, 2, 3, 4],
        }
    )
    train_df, val_df, test_df = time_based_split(df)
    assert train_df["value"].tolist() == [1]
    assert val_df["value"].tolist() == [2, 3]
    assert test_df["value"].tolist() == [4]


def test_evaluate_model_reports_notebook_metrics():
    class FakeModel:
        def predict_proba(self, x):
            probs = np.array([0.1, 0.4, 0.6, 0.9])
            return np.column_stack([1 - probs, probs])

    y = np.array([0, 0, 1, 1])
    metrics, probs, preds = evaluate_model(FakeModel(), x=np.zeros((4, 1)), y=y, threshold=0.5)
    assert set(metrics) == {"PR-AUC", "ROC-AUC", "F1-Score", "Precision", "Recall"}
    assert preds.tolist() == [0, 0, 1, 1]
    assert metrics["F1-Score"] == pytest.approx(1.0)


def test_fit_apply_transformers_round_trip():
    labeled = build_synthetic_labeled_dataset(rows=50)
    features = extract_features(labeled)
    transformers = fit_transformers(features)
    processed = apply_transformers(features, transformers)
    assert list(processed.columns) == transformers.feature_names
    assert not processed.isna().any().any()


def test_tune_threshold_picks_best_f1_on_grid():
    class FakeModel:
        def predict_proba(self, x):
            probs = np.linspace(0.05, 0.95, len(x))
            return np.column_stack([1 - probs, probs])

    y = (np.linspace(0.05, 0.95, 20) >= 0.5).astype(int)
    best_threshold, best_f1 = tune_threshold(FakeModel(), x_val=np.zeros((20, 1)), y_val=y)
    assert 0.1 <= best_threshold <= 0.9
    assert best_f1 == pytest.approx(1.0)


def test_training_pipeline_artifacts_reproduce_saved_test_metrics(tmp_path):
    """Saving/reloading artifacts must not change predictions vs. what training measured."""
    labeled = build_synthetic_labeled_dataset(rows=600)
    train_df, val_df, test_df = time_based_split(labeled)
    assert len(train_df) and len(val_df) and len(test_df)

    result = run_training_pipeline(train_df, val_df, test_df)
    save_training_artifacts(result, tmp_path)

    files = {
        "model_file": "final_best_model.joblib",
        "feature_names_file": "feature_names.joblib",
        "num_imputer_file": "num_imputer.joblib",
        "cat_imputer_file": "cat_imputer.joblib",
        "encoder_file": "ohe_encoder.joblib",
        "scaler_file": "scaler.joblib",
    }
    artifacts: ArtifactBundle = load_artifacts(tmp_path, files)
    assert artifacts.threshold == pytest.approx(result.threshold)

    settings = Settings(
        root_dir=tmp_path,
        values={
            "service": {
                "model_version": "test",
                "prediction_log_file": "predictions.jsonl",
                "max_batch_size": 10,
            },
            "data": {
                "required_columns": [],
                "numeric_ranges": {},
                "categorical_values": {},
                "max_missing_ratio": 1.0,
            },
        },
    )
    service = InferenceService(settings=settings, artifacts=artifacts)

    sample = test_df.iloc[0]
    payload = {
        "total_price": sample["total_price"],
        "total_freight": sample["total_freight"],
        "items_count": int(sample["items_count"]),
        "unique_products": int(sample["unique_products"]),
        "unique_sellers": int(sample["unique_sellers"]),
        "total_payment_value": sample["total_payment_value"],
        "max_payment_installments": int(sample["max_payment_installments"]),
        "payment_types_count": int(sample["payment_types_count"]),
        "customer_state": sample["customer_state"],
        "primary_payment_type": sample["primary_payment_type"],
        "order_purchase_timestamp": sample["order_purchase_timestamp"].isoformat(),
        "order_estimated_delivery_date": sample["order_estimated_delivery_date"].isoformat(),
        "order_approved_at": sample["order_approved_at"].isoformat(),
    }
    served_output = service.predict(payload)

    features = extract_features(test_df.iloc[[0]])
    processed = apply_transformers(features, result.transformers)
    expected_probability = float(result.best_model.predict_proba(processed)[:, 1][0])

    assert served_output["probability"] == pytest.approx(expected_probability)
    assert served_output["prediction"] == int(expected_probability >= result.threshold)
