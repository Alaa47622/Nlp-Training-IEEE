from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def read_json(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    lora = read_json("reports/metrics.json")
    lstm = read_json("reports/lstm_results.json")

    if not lora:
        raise FileNotFoundError("Run evaluation first: uv run python src/evaluate.py")
    if not lstm:
        print("No reports/lstm_results.json found.")
        print("Use the previous LSTM results in that file, or run the optional baseline.")
        return

    rows = [
        ["Precision (micro)", lstm.get("precision_micro"), lora.get("precision_micro")],
        ["Recall (micro)", lstm.get("recall_micro"), lora.get("recall_micro")],
        ["F1 (micro)", lstm.get("f1_micro"), lora.get("f1_micro")],
        ["Trainable parameters", lstm.get("trainable_parameters"), None],
    ]
    df = pd.DataFrame(rows, columns=["Metric", "LSTM", "DistilBERT + LoRA"])
    df.to_csv("reports/comparison.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
