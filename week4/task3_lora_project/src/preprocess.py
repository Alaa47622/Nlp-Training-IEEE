from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import LABELS, ensure_dir, load_params, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    cfg = params["data"]
    set_seed(cfg["seed"])

    df = pd.read_csv(cfg["input_csv"])
    required = ["id", "comment_text", *LABELS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["comment_text"] = df["comment_text"].fillna("").astype(str).str.strip()
    df = df[df["comment_text"].str.len() > 0].copy()

    train_df, val_df = train_test_split(
        df,
        test_size=cfg["validation_size"],
        random_state=cfg["seed"],
        shuffle=True,
    )

    # For faster coursework/CI runs, a reproducible cap can be used.
    max_train = cfg.get("max_train_samples")
    max_val = cfg.get("max_validation_samples")
    if max_train:
        train_df = train_df.sample(min(max_train, len(train_df)), random_state=cfg["seed"])
    if max_val:
        val_df = val_df.sample(min(max_val, len(val_df)), random_state=cfg["seed"])

    out = Path(cfg["processed_dir"])
    ensure_dir(out)

    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(out / "validation.csv", index=False)

    print(f"Raw rows: {len(df):,}")
    print(f"Train rows: {len(train_df):,}")
    print(f"Validation rows: {len(val_df):,}")
    print(f"Saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
