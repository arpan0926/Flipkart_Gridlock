from pathlib import Path
import pandas as pd
from datetime import datetime

from Model.data_loader import load_dataset
from Model.pipeline import build_model_pipeline, prepare_features, evaluate_pipeline


def main() -> None:
    dataset_dir = Path(__file__).resolve().parents[1] / "dataset"
    data = load_dataset(dataset_dir)

    print("Loaded dataset files:")
    for name, frame in data.items():
        print(f"- {name}: {frame.shape[0]} rows, {frame.shape[1]} columns")

    X, y = prepare_features(data["train"])
    
    # Chronological sort to prevent data leakage
    X = X.sort_values(by=["day", "hour", "minute"])
    y = y.loc[X.index] # Keep targets aligned with the sorted features

    # Take the first 80% of time for training, last 20% for validation
    split_index = int(len(X) * 0.8)
    
    X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_val = y.iloc[:split_index], y.iloc[split_index:]

    # 1. Build the empty pipeline
    model = build_model_pipeline() 
    
    # 2. Train the pipeline and tell LightGBM to natively handle 'category' dtypes
    print("\nTraining ensemble model...")
    model.fit(X_train, y_train)

    # 3. Evaluate on the validation set
    metrics = evaluate_pipeline(model, X_val, y_val)

    print("\nValidation metrics:")
    print(f"- RMSE: {metrics['rmse']:.6f}")
    print(f"- R2: {metrics['r2']:.6f}")

    # Prepare test data and predict
    test_X, _ = prepare_features(data["test"], target=None)
    predictions = model.predict(test_X)
    print(f"\nSample test predictions ({min(5, len(predictions))} rows):")
    for value in predictions[:5]:
        print(f"- {value:.6f}")

    print("\nGenerating submission file...")
    
    # Read the raw test file directly to get the dropped 'Index' column
    raw_test_df = pd.read_csv(dataset_dir / "test.csv")
    
    # Create the submission dataframe
    submission = pd.DataFrame({
        "Index": raw_test_df["Index"],
        "demand": predictions
    })
    
    # Use the timestamp to prevent overwriting old submissions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
    file_name = f"predictions_{timestamp}.csv"
    
    # Save it to the root of your project
    submission_file = Path(__file__).resolve().parents[1] / file_name
    submission.to_csv(submission_file, index=False)
    
    print(f"Successfully saved {len(submission)} predictions to {file_name}")


if __name__ == "__main__":
    main()