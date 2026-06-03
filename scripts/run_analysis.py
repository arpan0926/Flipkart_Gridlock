from pathlib import Path
import pandas as pd
import time 
from datetime import datetime
from sklearn.model_selection import train_test_split

from Model.data_loader import load_dataset
from Model.pipeline import prepare_features, train_pipeline, evaluate_pipeline


def main() -> None:
    dataset_dir = Path(__file__).resolve().parents[1] / "dataset"
    data = load_dataset(dataset_dir)

    print("Loaded dataset files:")
    for name, frame in data.items():
        print(f"- {name}: {frame.shape[0]} rows, {frame.shape[1]} columns")

    X, y = prepare_features(data["train"])
    X = X.sort_values(by=["day", "hour", "minute"])
    y = y.loc[X.index] # Keep targets aligned with the sorted features

    # Take the first 80% of time for training, last 20% for validation
    split_index = int(len(X) * 0.8)
    
    X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_val = y.iloc[:split_index], y.iloc[split_index:]

    model = train_pipeline(X_train, y_train)
    metrics = evaluate_pipeline(model, X_val, y_val)

    print("\nValidation metrics:")
    print(f"- RMSE: {metrics['rmse']:.6f}")
    print(f"- R2: {metrics['r2']:.6f}")

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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # e.g., 20260602_165900
    file_name = f"predictions_{timestamp}.csv"
    
    # Save it to the root of your project
    submission_file = Path(__file__).resolve().parents[1] / "predictions.csv"
    submission.to_csv(submission_file, index=False)
    
    print(f"Successfully saved {len(submission)} predictions to {submission_file}")


if __name__ == "__main__":
    main()
