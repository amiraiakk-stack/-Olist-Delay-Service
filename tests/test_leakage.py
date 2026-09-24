"""Guards against future-column leakage into features and val/test leaking into fit."""

import numpy as np

from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, extract_features
from src.fixtures import build_synthetic_labeled_dataset
from src.preprocessing import apply_transformers, fit_transformers
from src.splitting import time_based_split
from src.training import run_training_pipeline

# only exist after delivery/review; using these as inputs would leak the label
FUTURE_ONLY_COLUMNS = {
    "order_delivered_customer_date",
    "order_delivered_carrier_date",
    "review_score",
    "review_comment_message",
    "has_review_comment",
    "delay_days",
    "is_delayed",
}


def test_model_features_exclude_future_only_columns():
    model_features = set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
    leaked = model_features & FUTURE_ONLY_COLUMNS
    assert not leaked, f"future-only columns leaked into model features: {leaked}"


def test_time_based_split_has_no_temporal_overlap():
    labeled = build_synthetic_labeled_dataset(rows=600)
    train_df, val_df, test_df = time_based_split(labeled)

    assert train_df["order_purchase_timestamp"].max() <= val_df["order_purchase_timestamp"].min()
    assert val_df["order_purchase_timestamp"].max() <= test_df["order_purchase_timestamp"].min()


def test_time_based_split_rows_are_disjoint_and_complete():
    labeled = build_synthetic_labeled_dataset(rows=600).reset_index(drop=True)
    labeled["row_id"] = labeled.index
    train_df, val_df, test_df = time_based_split(labeled)

    train_ids, val_ids, test_ids = (
        set(train_df["row_id"]),
        set(val_df["row_id"]),
        set(test_df["row_id"]),
    )
    assert not (train_ids & val_ids)
    assert not (val_ids & test_ids)
    assert not (train_ids & test_ids)
    assert train_ids | val_ids | test_ids == set(labeled["row_id"])


def test_fitting_transformers_is_insensitive_to_validation_data():
    """Feeding a skewed validation set through apply_transformers must not change the fit."""
    train_features = extract_features(build_synthetic_labeled_dataset(rows=300, seed=1))
    transformers = fit_transformers(train_features)

    fitted_mean = np.array(transformers.scaler.mean_, copy=True)
    fitted_statistics = np.array(transformers.num_imputer.statistics_, copy=True)

    skewed_val = build_synthetic_labeled_dataset(rows=50, seed=99)
    skewed_val["total_price"] = skewed_val["total_price"] * 1000 + 50000
    apply_transformers(extract_features(skewed_val), transformers)

    np.testing.assert_array_equal(transformers.scaler.mean_, fitted_mean)
    np.testing.assert_array_equal(transformers.num_imputer.statistics_, fitted_statistics)


def test_run_training_pipeline_transformers_depend_only_on_train_split():
    """Transformers from run_training_pipeline must match fitting on the train split alone."""
    train_df, val_df, test_df = time_based_split(build_synthetic_labeled_dataset(rows=600))
    result = run_training_pipeline(train_df, val_df, test_df)

    standalone = fit_transformers(extract_features(train_df))

    np.testing.assert_array_equal(result.transformers.scaler.mean_, standalone.scaler.mean_)
    np.testing.assert_array_equal(
        result.transformers.num_imputer.statistics_, standalone.num_imputer.statistics_
    )
    assert result.transformers.feature_names == standalone.feature_names
