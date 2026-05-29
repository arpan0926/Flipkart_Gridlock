from pathlib import Path

from sklearn.model_selection import train_test_split

from flipkart_gridlock.data_loader import load_dataset
from flipkart_gridlock.pipeline import prepare_features, train_pipeline, evaluate_pipeline


def main() -> None:
    dataset_dir = Path(__file__).resolve().parents[1] / "dataset"
    data = load_dataset(dataset_dir)

    print("Loaded dataset files:")
    for name, frame in data.items():
        print(f"- {name}: {frame.shape[0]} rows, {frame.shape[1]} columns")

    X, y = prepare_features(data["train"])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

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


if __name__ == "__main__":
    main()
