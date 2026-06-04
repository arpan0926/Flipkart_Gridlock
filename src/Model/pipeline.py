from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import pygeohash as pgh
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "demand"

TARGET_ENCODE_FEATURES = ["geohash"]
# 1. ADDED the new numeric features: day_of_week, lat, and lon
NUMERIC_FEATURES = [
    "day", "NumberofLanes", "Temperature", "minute", "lat", "lon",
    "hour_sin", "hour_cos", "day_sin", "day_cos", "is_weekend", "is_rush_hour"
]

# 2. REMOVED "geohash" and "timestamp" from categorical features
CATEGORICAL_FEATURES = ["RoadType", "LargeVehicles", "Landmarks", "Weather"]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = df["timestamp"].fillna("0:0").astype(str)
    parts = df["timestamp"].str.split(":", expand=True)
    df["hour"] = pd.to_numeric(parts[0], errors="coerce").fillna(0).astype(int)
    df["minute"] = pd.to_numeric(parts[1], errors="coerce").fillna(0).astype(int)
    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create advanced spatio-temporal features."""
    df = df.copy()
    
    # 1. Day of Week
    if "day" in df.columns:
        df["day_of_week"] = df["day"] % 7
        
        # Cyclical Day Encoding
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # 2. Cyclical Hour Encoding
    if "hour" in df.columns:
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['is_rush_hour'] = df['hour'].apply(
            lambda x: 1 if (7 <= x <= 10) or (16 <= x <= 19) else 0
        )
        
    # 3. Decode Geohash
    if "geohash" in df.columns:
        df['lat'] = df['geohash'].apply(lambda x: pgh.decode(str(x))[0] if pd.notna(x) else None)
        df['lon'] = df['geohash'].apply(lambda x: pgh.decode(str(x))[1] if pd.notna(x) else None)

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")
        
    return df


def prepare_features(df: pd.DataFrame, target: Optional[str] = TARGET_COLUMN) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Prepare feature matrix X and target vector y from raw DataFrame."""
    df = df.copy()
    
    # Run the feature engineering steps
    df = _parse_timestamp(df)
    df = _engineer_features(df) 
    
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

    return X, y


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    

    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            # Just pass the categorical columns directly through
            ("categorical", "passthrough", CATEGORICAL_FEATURES),
            ("target_encoding", TargetEncoder(smoothing=10), TARGET_ENCODE_FEATURES), 
        ],
        remainder="drop",
    )
    
    preprocessor.set_output(transform="pandas")
    
    return preprocessor

def build_model_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("regressor", LGBMRegressor(
                n_estimators=600,        # Increased trees for better learning
                learning_rate=0.03,      # Slower learning rate for precision
                num_leaves=63,           # Allows deeper spatial logic
                max_depth=8,             # Caps depth to prevent overfitting the test set
                min_child_samples=20,    # Ensures leaf nodes have enough data
                verbose=-1,
                random_state=42,
                n_jobs=-1
            )),
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