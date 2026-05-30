import pandas as pd

from Model.model import train_model, evaluate_model


def test_train_and_evaluate_model():
    X = pd.DataFrame({"feature": [0.0, 1.0, 0.0, 1.0]})
    y = pd.Series([0.1, 0.9, 0.2, 0.8])

    model = train_model(X, y)
    metrics = evaluate_model(model, X, y)

    assert metrics["rmse"] < 1.0
    assert metrics["r2"] > 0.9
