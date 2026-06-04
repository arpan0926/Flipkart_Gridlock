from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error
from lightgbm import early_stopping, log_evaluation

from Model.data_loader import load_dataset
from Model.pipeline import build_model, prepare_features, _parse_timestamp


def main() -> None:
    dataset_dir = Path(__file__).resolve().parents[1] / "dataset"
    data = load_dataset(dataset_dir)

    print("Loaded dataset files:")
    for name, frame in data.items():
        print(f"  {name}: {frame.shape[0]:,} rows, {frame.shape[1]} columns")

    # Chronological 80/20 split
    train_raw = data["train"]
    train_sorted = _parse_timestamp(train_raw).sort_values(["day", "hour", "minute"])
    split_index = int(len(train_sorted) * 0.8)

    train_split_raw = train_raw.iloc[list(train_sorted.index[:split_index])]
    val_split_raw   = train_raw.iloc[list(train_sorted.index[split_index:])]

    X_train, y_train = prepare_features(train_split_raw)
    X_val,   y_val   = prepare_features(val_split_raw)

    print(f"\nSplit — Train: {len(X_train):,}  Val: {len(X_val):,}  Features: {X_train.shape[1]}")

    # Align categories between train/val splits
    for col in X_train.select_dtypes("category").columns:
        cats = X_train[col].cat.categories.union(X_val[col].cat.categories)
        X_train[col] = X_train[col].cat.set_categories(cats)
        X_val[col]   = X_val[col].cat.set_categories(cats)

    print("\nTraining LightGBM with early stopping...")
    model = build_model()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            early_stopping(stopping_rounds=60, verbose=True),
            log_evaluation(period=100),
        ],
    )

    best_iter = model.best_iteration_
    print(f"\nBest iteration: {best_iter}")

    preds_val = model.predict(X_val)
    r2   = r2_score(y_val, preds_val)
    rmse = mean_squared_error(y_val, preds_val) ** 0.5
    print(f"\nValidation metrics (80/20 chronological):")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  R2:   {r2:.6f}")

    # Retrain on 100% of data with best_iter rounds
    print("\nRetraining on full training data...")
    X_full, y_full = prepare_features(train_raw)
    final_model = build_model()
    final_model.set_params(n_estimators=best_iter)
    final_model.fit(X_full, y_full)
    print("Done.")

    # Test
    print("\nPreparing test features...")
    test_X, _ = prepare_features(data["test"], target=None)

    # Align test categories with full training categories
    for col in X_full.select_dtypes("category").columns:
        if col in test_X.columns:
            cats = X_full[col].cat.categories.union(test_X[col].cat.categories)
            X_full[col]  = X_full[col].cat.set_categories(cats)
            test_X[col]  = test_X[col].cat.set_categories(cats)

    predictions = final_model.predict(test_X)

    raw_test_df = pd.read_csv(dataset_dir / "test.csv")
    submission = pd.DataFrame({
        "Index":  raw_test_df["Index"],
        "demand": predictions,
    })
    print(f"Submission shape: {submission.shape}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"predictions_{ts}.csv"
    submission.to_csv(Path(__file__).resolve().parents[1] / fname, index=False)
    print(f"Saved → {fname}")


if __name__ == "__main__":
    main()
