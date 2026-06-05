from typing import Optional, Tuple
import numpy as np
import pandas as pd
import pygeohash as pgh
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

TARGET_COLUMN = "demand"

# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "hour", "minute", "day", "slot_of_day",
    "hour_sin", "hour_cos",
    "minute_sin", "minute_cos",
    "day_sin", "day_cos",
    "is_rush_hour", "is_late_night", "is_off_peak",
    "NumberofLanes", "Temperature",
    "lat", "lon",
    "lat_hour_interaction", "lon_hour_interaction", "lat_lon_combined", 
]

CATEGORICAL_FEATURES = [
    "RoadType", "LargeVehicles", "Landmarks", "Weather",
    "geohash", "geohash_prefix4", "geohash_prefix5",

]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

def _parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"]   = df["timestamp"].fillna("0:0").astype(str)
    parts = df["timestamp"].str.split(":", expand=True)
    df["hour"]        = pd.to_numeric(parts[0], errors="coerce").fillna(0).astype(int)
    df["minute"]      = pd.to_numeric(parts[1], errors="coerce").fillna(0).astype(int)
    df["slot_of_day"] = df["hour"] * 4 + df["minute"] // 15
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "day" in df.columns:
        dow = df["day"] % 7
        
        df["day_sin"] = np.sin(2 * np.pi * dow / 7)
        df["day_cos"] = np.cos(2 * np.pi * dow / 7)

    if "hour" in df.columns:
        df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
        df["is_rush_hour"]  = ((df["hour"].between(7, 10)) | (df["hour"].between(16, 19))).astype(int)
        df["is_late_night"] = ((df["hour"] <= 5) | (df["hour"] >= 23)).astype(int)
        df["is_off_peak"]   = df["hour"].between(11, 15).astype(int)

    if "minute" in df.columns:
        df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
        df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

    if "geohash" in df.columns:
        df["lat"] = df["geohash"].apply(lambda x: pgh.decode(str(x))[0] if pd.notna(x) else np.nan)
        df["lon"] = df["geohash"].apply(lambda x: pgh.decode(str(x))[1] if pd.notna(x) else np.nan)
        df["geohash_prefix4"] = df["geohash"].apply(lambda x: str(x)[:4] if pd.notna(x) else "unknown")
        df["geohash_prefix5"] = df["geohash"].apply(lambda x: str(x)[:5] if pd.notna(x) else "unknown")

    if "lat" in df.columns and "hour_sin" in df.columns:
        df["lat_hour_interaction"] = df["lat"] * df["hour_sin"]
        df["lon_hour_interaction"] = df["lon"] * df["hour_cos"]
    if "lat" in df.columns and "lon" in df.columns:
        df["lat_lon_combined"] = df["lat"] * df["lon"]
    if "NumberofLanes" in df.columns and "RoadType" in df.columns:
        df["Road_lane_Profile"] = df["RoadType"].astype(str) + "_" + df["NumberofLanes"].astype(str)
    if "Temperature" in df.columns:
        df["Temperature"] = df["Temperature"].fillna(
            df.groupby("hour")["Temperature"].transform("mean")
        )
        df["Temperature"] = df["Temperature"].fillna(df["Temperature"].mean())

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).astype("category")

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame,
    target: Optional[str] = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    df = df.copy()
    df = _parse_timestamp(df)
    df = _engineer_features(df)
    df = df.drop(columns=["Index"], errors="ignore")

    if target and target in df.columns:
        y = df[target].astype(float)
        X = df.drop(columns=[target])
    else:
        y = None
        X = df

    missing = set(FEATURE_COLUMNS) - set(X.columns)
    if missing:
        raise ValueError(f"Missing expected features: {sorted(missing)}")

    return X[FEATURE_COLUMNS], y


def build_model() -> LGBMRegressor:
    return LGBMRegressor(
        num_leaves=255,
        max_depth=12,
        min_child_samples=10,
        n_estimators=2000,
        learning_rate=0.02,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,   
        reg_alpha=0.05,
        reg_lambda=1.0,
        verbose=-1,
        random_state=42,
        n_jobs=-1,
    )


def build_model_pipeline() -> LGBMRegressor:
    """Alias for test compatibility."""
    return build_model()


def train_pipeline(X: pd.DataFrame, y: pd.Series) -> LGBMRegressor:
    model = build_model()
    model.fit(X, y)
    return model


def evaluate_pipeline(model, X: pd.DataFrame, y: pd.Series) -> dict:
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    return {
        "rmse": float(mse**0.5),
        "r2": float(r2_score(y, preds)),
    }
