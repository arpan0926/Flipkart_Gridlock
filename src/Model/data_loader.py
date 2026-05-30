import pandas as pd
from pathlib import Path


def load_dataset(dataset_dir: Path) -> dict:
    """Load train and test datasets from the given directory."""
    train_path = dataset_dir / "train.csv"
    test_path = dataset_dir / "test.csv"

    train_df = pd.read_csv(train_path).drop(columns=["Index"], errors="ignore")
    test_df = pd.read_csv(test_path).drop(columns=["Index"], errors="ignore")

    return {
        "train": train_df,
        "test": test_df,
    }
