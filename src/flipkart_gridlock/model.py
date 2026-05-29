from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def train_model(X, y):
    """Train a random forest regressor."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


def evaluate_model(model, X, y):
    """Evaluate a trained model and return regression metrics."""
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    return {
        "rmse": float(mse**0.5),
        "r2": float(r2_score(y, preds)),
    }
