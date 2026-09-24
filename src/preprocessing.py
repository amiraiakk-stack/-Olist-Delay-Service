from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, extract_features


@dataclass
class ArtifactBundle:
    model: object
    feature_names: list[str]
    num_imputer: object
    cat_imputer: object
    encoder: object
    scaler: object
    threshold: float
    model_name: str


@dataclass
class FittedTransformers:
    num_imputer: object
    cat_imputer: object
    encoder: object
    scaler: object
    feature_names: list[str]


def load_artifacts(artifacts_dir: Path, files: dict[str, str]) -> ArtifactBundle:
    directory = Path(artifacts_dir)
    model_artifact = joblib.load(directory / files["model_file"])
    if isinstance(model_artifact, dict):
        model = model_artifact["model"]
        threshold = float(model_artifact.get("optimal_threshold", 0.5))
        model_name = str(model_artifact.get("model_name", type(model).__name__))
    else:
        model = model_artifact
        threshold = 0.5
        model_name = type(model).__name__
    return ArtifactBundle(
        model=model,
        feature_names=list(joblib.load(directory / files["feature_names_file"])),
        num_imputer=joblib.load(directory / files["num_imputer_file"]),
        cat_imputer=joblib.load(directory / files["cat_imputer_file"]),
        encoder=joblib.load(directory / files["encoder_file"]),
        scaler=joblib.load(directory / files["scaler_file"]),
        threshold=threshold,
        model_name=model_name,
    )


def fit_transformers(features: pd.DataFrame) -> FittedTransformers:
    """Fit imputers/encoder/scaler on a training split only."""
    num_imputer = SimpleImputer(strategy="median")
    num_imputer.fit(features[NUMERIC_FEATURES])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    cat_imputer.fit(features[CATEGORICAL_FEATURES])

    train_cat_imputed = cat_imputer.transform(features[CATEGORICAL_FEATURES])
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train_cat_imputed)

    train_num_imputed = num_imputer.transform(features[NUMERIC_FEATURES])
    scaler = StandardScaler()
    scaler.fit(train_num_imputed)

    ohe_columns = list(encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = NUMERIC_FEATURES + ohe_columns
    return FittedTransformers(
        num_imputer=num_imputer,
        cat_imputer=cat_imputer,
        encoder=encoder,
        scaler=scaler,
        feature_names=feature_names,
    )


def apply_transformers(features: pd.DataFrame, transformers: FittedTransformers) -> pd.DataFrame:
    missing = [name for name in NUMERIC_FEATURES + CATEGORICAL_FEATURES if name not in features]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    numeric = transformers.num_imputer.transform(features[NUMERIC_FEATURES])
    categorical = transformers.cat_imputer.transform(features[CATEGORICAL_FEATURES])
    numeric_scaled = transformers.scaler.transform(numeric)
    categorical_encoded = transformers.encoder.transform(categorical)
    processed = np.hstack([numeric_scaled, categorical_encoded])
    result = pd.DataFrame(processed, columns=transformers.feature_names, index=features.index)
    if list(result.columns) != transformers.feature_names:
        raise ValueError("Processed feature names do not match the training artifact")
    return result


def transform_for_model(frame: pd.DataFrame, artifacts: ArtifactBundle) -> pd.DataFrame:
    features = extract_features(frame)
    transformers = FittedTransformers(
        num_imputer=artifacts.num_imputer,
        cat_imputer=artifacts.cat_imputer,
        encoder=artifacts.encoder,
        scaler=artifacts.scaler,
        feature_names=artifacts.feature_names,
    )
    return apply_transformers(features, transformers)
