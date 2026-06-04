import math
import pandas as pd
import numpy as np

from Model.pipeline import (
    FEATURE_COLUMNS,
    build_model_pipeline,
    evaluate_pipeline,
    prepare_features,
    prepare_train_with_stats,
    prepare_test_with_stats,
    train_pipeline,
)


def _make_sample_df(n_rows: int = 8) -> pd.DataFrame:
    reps = math.ceil(n_rows / 4)
    geohashes  = (["qp02z1", "qp02zt", "qp08bj", "qp08gt"] * reps)[:n_rows]
    timestamps = (["0:0", "0:15", "0:30", "0:45", "1:0", "1:15", "1:30", "1:45"] * reps)[:n_rows]
    demands    = ([0.05, 0.10, 0.55, 0.80, 0.06, 0.12, 0.50, 0.75] * reps)[:n_rows]
    road_types = (["Residential", "Residential", "Highway", "Residential"] * reps)[:n_rows]
    lanes      = ([1, 3, 4, 2] * reps)[:n_rows]
    large_v    = (["Not Allowed", "Allowed", "Allowed", "Not Allowed"] * reps)[:n_rows]
    landmarks  = (["No", "Yes", "No", "No"] * reps)[:n_rows]
    temps      = ([20.0, 25.0, 22.0, 18.0] * reps)[:n_rows]
    weather    = (["Sunny", "Rainy", "Foggy", "Rainy"] * reps)[:n_rows]
    return pd.DataFrame({
        "geohash":       geohashes,
        "day":           [48] * n_rows,
        "timestamp":     timestamps,
        "demand":        demands,
        "RoadType":      road_types,
        "NumberofLanes": lanes,
        "LargeVehicles": large_v,
        "Landmarks":     landmarks,
        "Temperature":   temps,
        "Weather":       weather,
    })


def test_prepare_features_with_target():
    df = _make_sample_df(4)
    X, y = prepare_features(df)
    assert "demand" not in X.columns
    assert set(X.columns) == set(FEATURE_COLUMNS), (
        f"Expected: {sorted(FEATURE_COLUMNS)}\nGot:      {sorted(X.columns)}"
    )
    assert len(y) == 4


def test_prepare_features_without_target():
    df = _make_sample_df(4).drop(columns=["demand"])
    X, y = prepare_features(df, target=None)
    assert y is None
    assert set(X.columns) == set(FEATURE_COLUMNS)


def test_feature_columns_complete():
    df = _make_sample_df(8)
    X, _ = prepare_features(df)
    missing = set(FEATURE_COLUMNS) - set(X.columns)
    assert not missing, f"Missing features: {missing}"


def test_no_leakage_columns():
    bad = {"demand", "timestamp", "Index"}
    overlap = bad & set(FEATURE_COLUMNS)
    assert not overlap, f"Leakage columns in FEATURE_COLUMNS: {overlap}"


def test_train_pipeline_and_evaluate():
    df = _make_sample_df(8)
    X, y = prepare_features(df)
    pipeline = train_pipeline(X, y)
    metrics = evaluate_pipeline(pipeline, X, y)
    assert metrics["rmse"] >= 0
    assert metrics["r2"] <= 1.0


def test_prepare_train_with_stats():
    df = _make_sample_df(8)
    X, y, stats, global_mean = prepare_train_with_stats(df)
    assert set(X.columns) == set(FEATURE_COLUMNS)
    assert "geo" in stats
    assert "slot" in stats
    assert "geo_slot" in stats
    assert global_mean > 0


def test_prepare_test_with_stats():
    train_df = _make_sample_df(8)
    test_df  = _make_sample_df(4).drop(columns=["demand"])
    _, _, stats, global_mean = prepare_train_with_stats(train_df)
    X_test = prepare_test_with_stats(test_df, stats, global_mean)
    assert set(X_test.columns) == set(FEATURE_COLUMNS)
    assert len(X_test) == 4


def test_stat_features_are_not_nan():
    df = _make_sample_df(8)
    X, _, _, _ = prepare_train_with_stats(df)
    stat_cols = ["geo_mean_demand", "geo_std_demand", "geo_max_demand",
                 "slot_mean_demand", "geo_slot_mean_demand"]
    for col in stat_cols:
        assert X[col].isna().sum() == 0, f"{col} has NaN values"
