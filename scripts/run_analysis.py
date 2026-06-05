from pathlib import Path
import pandas as pd
from datetime import datetime
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from Model.data_loader import load_dataset
from Model.pipeline import prepare_features, _parse_timestamp, CATEGORICAL_FEATURES

def main() -> None:
    dataset_dir = Path(__file__).resolve().parents[1] / "dataset"
    data = load_dataset(dataset_dir)

    print("Loaded dataset files:")
    for name, frame in data.items():
        print(f"  {name}: {frame.shape[0]:,} rows, {frame.shape[1]} columns")

    # ---------------------------------------------------------
    # 1. PREPARE 100% OF THE DATA
    # ---------------------------------------------------------
    print("\nPreparing full training dataset...")
    train_raw = data["train"]
    train_sorted = _parse_timestamp(train_raw).sort_values(["day", "hour", "minute"])
    X_full, y_full = prepare_features(train_sorted)

    test_X, _ = prepare_features(data["test"], target=None)

    # Align categories between train and test so LightGBM doesn't crash
    for col in X_full.select_dtypes("category").columns:
        if col in test_X.columns:
            cats = X_full[col].cat.categories.union(test_X[col].cat.categories)
            X_full[col] = X_full[col].cat.set_categories(cats)
            test_X[col] = test_X[col].cat.set_categories(cats)

    # ---------------------------------------------------------
    # MODEL 1: LightGBM (The 91.22 Baseline Parameters)
    # ---------------------------------------------------------
    print("\n[1/2] Training LightGBM on 100% of data...")
    final_lgbm = LGBMRegressor(
        num_leaves=255,          
        max_depth=12,            
        min_child_samples=10,
        n_estimators=1600,       
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    final_lgbm.fit(X_full, y_full)
    lgbm_preds = final_lgbm.predict(test_X)

    # ---------------------------------------------------------
    # MODEL 2: CatBoost (The Stabilizer)
    # ---------------------------------------------------------
    print("\n[2/2] Training CatBoost on 100% of data...")
    # Get index positions of categorical features for CatBoost
    cat_indices = [X_full.columns.get_loc(col) for col in CATEGORICAL_FEATURES if col in X_full.columns]
    
    final_cat = CatBoostRegressor(
        iterations=1600,
        learning_rate=0.03,
        depth=8,                 
        l2_leaf_reg=3,           
        cat_features=cat_indices,
        random_seed=42,
        verbose=100              
    )
    final_cat.fit(X_full, y_full)
    cat_preds = final_cat.predict(test_X)

    # ---------------------------------------------------------
    # THE BLEND & SUBMISSION
    # ---------------------------------------------------------
    print("\nBlending predictions...")
    
    # Weight LightGBM slightly higher since it historically performed better
    final_predictions = (lgbm_preds * 0.60) + (cat_preds * 0.40)

    raw_test_df = pd.read_csv(dataset_dir / "test.csv")
    submission = pd.DataFrame({
        "Index":  raw_test_df["Index"],
        "demand": final_predictions,  # Saving the ACTUAL blend this time!
    })
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"predictions_{ts}.csv"
    submission.to_csv(Path(__file__).resolve().parents[1] / fname, index=False)
    print(f"\n✅ Successfully saved blended predictions to {fname}")

if __name__ == "__main__":
    main()