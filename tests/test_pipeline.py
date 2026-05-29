import pandas as pd

from flipkart_gridlock.pipeline import build_model_pipeline, evaluate_pipeline, prepare_features, train_pipeline


def test_prepare_features_with_target():
    df = pd.DataFrame(
        {
            "Index": [0, 1],
            "geohash": ["qp02z1", "qp02zt"],
            "day": [48, 48],
            "timestamp": ["0:0", "2:15"],
            "demand": [0.1, 0.2],
            "RoadType": ["Residential", "Residential"],
            "NumberofLanes": [1, 3],
            "LargeVehicles": ["Not Allowed", "Allowed"],
            "Landmarks": ["No", "Yes"],
            "Temperature": [20.0, 25.0],
            "Weather": ["Sunny", "Rainy"],
        }
    )

    X, y = prepare_features(df)

    assert "demand" not in X.columns
    assert X.shape[1] == 11
    assert y.tolist() == [0.1, 0.2]


def test_train_pipeline_and_evaluate():
    df = pd.DataFrame(
        {
            "geohash": ["qp02z1", "qp02zt", "qp08bj", "qp08gt"],
            "day": [48, 48, 48, 48],
            "timestamp": ["0:0", "2:15", "1:30", "3:45"],
            "demand": [0.1, 0.2, 0.15, 0.22],
            "RoadType": ["Residential", "Residential", "Residential", "Residential"],
            "NumberofLanes": [1, 3, 1, 1],
            "LargeVehicles": ["Not Allowed", "Allowed", "Not Allowed", "Not Allowed"],
            "Landmarks": ["No", "Yes", "No", "No"],
            "Temperature": [20.0, 25.0, 22.0, 18.0],
            "Weather": ["Sunny", "Rainy", "Foggy", "Rainy"],
        }
    )

    X, y = prepare_features(df)
    pipeline = train_pipeline(X, y)
    metrics = evaluate_pipeline(pipeline, X, y)

    assert metrics["rmse"] >= 0
    assert metrics["r2"] <= 1.0
