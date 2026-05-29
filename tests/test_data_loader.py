from pathlib import Path

from flipkart_gridlock.data_loader import load_dataset


def test_load_dataset_paths():
    dataset_dir = Path(__file__).resolve().parents[1] / "dataset"
    data = load_dataset(dataset_dir)

    assert "train" in data
    assert "test" in data
    assert "demand" in data["train"].columns
    assert "geohash" in data["test"].columns
    assert "Index" not in data["train"].columns
