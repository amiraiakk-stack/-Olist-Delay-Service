from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, extract_features
from .preprocessing import FittedTransformers, apply_transformers, fit_transformers

DEFAULT_THRESHOLD_GRID = np.linspace(0.1, 0.9, 81)


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Extract the raw model inputs (pre imputation/scaling) and label from a labeled split."""
    features = extract_features(df)
    missing = [name for name in NUMERIC_FEATURES + CATEGORICAL_FEATURES if name not in features]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    x_raw = features[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df["is_delayed"].values
    return x_raw, y


def evaluate_model(
    model: Any, x: pd.DataFrame, y: np.ndarray, threshold: float = 0.5
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(x)[:, 1]
    else:
        y_probs = model.decision_function(x)

    y_preds = (y_probs >= threshold).astype(int)

    precision_vec, recall_vec, _ = precision_recall_curve(y, y_probs)
    pr_auc = auc(recall_vec, precision_vec)

    metrics = {
        "PR-AUC": pr_auc,
        "ROC-AUC": roc_auc_score(y, y_probs),
        "F1-Score": f1_score(y, y_preds, zero_division=0),
        "Precision": precision_score(y, y_preds, zero_division=0),
        "Recall": recall_score(y, y_preds, zero_division=0),
    }
    return metrics, y_probs, y_preds


def train_baseline_models(x_train: pd.DataFrame, y_train: np.ndarray) -> dict[str, Any]:
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_train, y_train)

    log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    log_reg.fit(x_train, y_train)

    return {"Dummy Baseline": dummy, "Logistic Regression": log_reg}


def train_candidate_models(x_train: pd.DataFrame, y_train: np.ndarray) -> dict[str, Any]:
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1, verbose=-1
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42
        ),
    }
    for model in models.values():
        model.fit(x_train, y_train)
    return models


def tune_threshold(
    model: Any,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    thresholds: np.ndarray = DEFAULT_THRESHOLD_GRID,
) -> tuple[float, float]:
    _, y_val_probs, _ = evaluate_model(model, x_val, y_val)
    f1_scores = [
        f1_score(y_val, (y_val_probs >= t).astype(int), zero_division=0) for t in thresholds
    ]
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


@dataclass
class TrainingResult:
    transformers: FittedTransformers
    baseline_models: dict[str, Any]
    candidate_models: dict[str, Any]
    val_metrics: dict[str, dict[str, float]]
    best_model_name: str
    best_model: Any
    threshold: float
    test_metrics: dict[str, float] = field(default_factory=dict)


def run_training_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_model_name: str = "LightGBM",
) -> TrainingResult:
    """Fit transformers, train/tune candidate models, and score the winner on test once."""
    train_features = extract_features(train_df)
    val_features = extract_features(val_df)
    test_features = extract_features(test_df)

    transformers = fit_transformers(train_features)
    x_train = apply_transformers(train_features, transformers)
    x_val = apply_transformers(val_features, transformers)
    x_test = apply_transformers(test_features, transformers)
    y_train, y_val, y_test = (
        train_df["is_delayed"].values,
        val_df["is_delayed"].values,
        test_df["is_delayed"].values,
    )

    baseline_models = train_baseline_models(x_train, y_train)
    candidate_models = train_candidate_models(x_train, y_train)

    val_metrics = {}
    for name, model in {**baseline_models, **candidate_models}.items():
        metrics, _, _ = evaluate_model(model, x_val, y_val)
        val_metrics[name] = metrics

    best_model = candidate_models[best_model_name]
    threshold, _ = tune_threshold(best_model, x_val, y_val)

    test_metrics, _, _ = evaluate_model(best_model, x_test, y_test, threshold=threshold)

    return TrainingResult(
        transformers=transformers,
        baseline_models=baseline_models,
        candidate_models=candidate_models,
        val_metrics=val_metrics,
        best_model_name=best_model_name,
        best_model=best_model,
        threshold=threshold,
        test_metrics=test_metrics,
    )


def save_training_artifacts(result: TrainingResult, artifacts_dir: Path) -> None:
    directory = Path(artifacts_dir)
    directory.mkdir(parents=True, exist_ok=True)

    joblib.dump(result.transformers.num_imputer, directory / "num_imputer.joblib")
    joblib.dump(result.transformers.cat_imputer, directory / "cat_imputer.joblib")
    joblib.dump(result.transformers.encoder, directory / "ohe_encoder.joblib")
    joblib.dump(result.transformers.scaler, directory / "scaler.joblib")
    joblib.dump(result.transformers.feature_names, directory / "feature_names.joblib")

    final_artifact = {
        "model": result.best_model,
        "model_name": f"{result.best_model_name}_Optimized",
        "optimal_threshold": result.threshold,
        "test_metrics": result.test_metrics,
    }
    joblib.dump(final_artifact, directory / "final_best_model.joblib")
