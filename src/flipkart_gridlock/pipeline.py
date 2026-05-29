from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "demand"
NUMERIC_FEATURES = ["day", "NumberofLanes", "Temperature", "hour", "minute"]
CATEGORICAL_FEATURES = ["geohash", "timestamp", "RoadType", "LargeVehicles", "Landmarks", "Weather"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = df["timestamp"].fillna("0:0").astype(str)
    parts = df["timestamp"].str.split(":", expand=True)
    df["hour"] = pd.to_numeric(parts[0], errors="coerce").fillna(0).astype(int)
    df["minute"] = pd.to_numeric(parts[1], errors="coerce").fillna(0).astype(int)
    return df


def prepare_features(df: pd.DataFrame, target: Optional[str] = TARGET_COLUMN) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Prepare feature matrix X and target vector y from raw DataFrame."""
    df = df.copy()
    df = _parse_timestamp(df)
    df = df.drop(columns=["Index"], errors="ignore")

    if target is not None and target in df.columns:
        y = df[target].astype(float)
        X = df.drop(columns=[target])
    else:
        y = None
        X = df

    missing_features = set(FEATURE_COLUMNS) - set(X.columns)
    if missing_features:
        raise ValueError(f"Missing expected features: {sorted(missing_features)}")

    return X[FEATURE_COLUMNS], y


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_model_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
        ]
    )


def train_pipeline(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    pipeline = build_model_pipeline()
    pipeline.fit(X, y)
    return pipeline


def evaluate_pipeline(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    preds = pipeline.predict(X)
    mse = mean_squared_error(y, preds)
    return {
        "rmse": float(mse**0.5),
        "r2": float(r2_score(y, preds)),
    }
