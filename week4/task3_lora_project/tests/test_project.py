from pathlib import Path

import pandas as pd

from src.utils import LABELS, load_params


def test_dataset_schema():
    path = Path("data/raw/train.csv")
    assert path.exists()
    df = pd.read_csv(path, nrows=5)
    assert "comment_text" in df.columns
    assert all(label in df.columns for label in LABELS)


def test_params():
    params = load_params("params.yaml")
    assert params["model"]["name"] == "distilbert-base-uncased"
    assert params["model"]["target_modules"] == ["q_lin", "v_lin"]
