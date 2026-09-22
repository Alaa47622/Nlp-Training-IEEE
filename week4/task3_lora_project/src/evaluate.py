from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import AutoPeftModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer

from src.utils import LABELS, ensure_dir, load_params, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    data_cfg = params["data"]
    train_cfg = params["training"]

    model_dir = Path(train_cfg["output_dir"])
    val_path = Path(data_cfg["processed_dir"]) / "validation.csv"
    report_dir = ensure_dir("reports")

    if not model_dir.exists():
        raise FileNotFoundError("Trained model not found. Run src/train.py first.")

    df = pd.read_csv(val_path)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoPeftModelForSequenceClassification.from_pretrained(model_dir)

    ds = Dataset.from_pandas(df[["comment_text", *LABELS]], preserve_index=False)

    def tokenize(batch):
        return tokenizer(
            batch["comment_text"],
            truncation=True,
            max_length=data_cfg["max_length"],
        )

    ds = ds.map(tokenize, batched=True)
    trainer = Trainer(
        model=model,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        processing_class=tokenizer,
    )
    predictions = trainer.predict(ds)
    logits = predictions.predictions
    probs = 1 / (1 + np.exp(-logits))
    threshold = train_cfg["threshold"]
    pred = (probs >= threshold).astype(int)
    y_true = df[LABELS].values

    metrics = {
        "accuracy_subset": float(accuracy_score(y_true, pred)),
        "precision_micro": float(precision_score(y_true, pred, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, pred, average="micro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
    }

    per_label = {}
    for i, label in enumerate(LABELS):
        per_label[label] = {
            "precision": float(precision_score(y_true[:, i], pred[:, i], zero_division=0)),
            "recall": float(recall_score(y_true[:, i], pred[:, i], zero_division=0)),
            "f1": float(f1_score(y_true[:, i], pred[:, i], zero_division=0)),
        }

        cm = confusion_matrix(y_true[:, i], pred[:, i])
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm)
        ax.set_title(f"Confusion Matrix: {label}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center")
        fig.tight_layout()
        fig.savefig(report_dir / f"confusion_matrix_{label}.png", dpi=150)
        plt.close(fig)

    metrics["per_label"] = per_label
    save_json(metrics, report_dir / "metrics.json")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    import json
    main()
