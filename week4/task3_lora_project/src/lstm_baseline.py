"""
Optional LSTM baseline.

Use this only if the previous LSTM implementation/results are unavailable.
The assignment asks for comparison with the previous LSTM implementation.
If you already have the previous LSTM metrics, put them in reports/lstm_results.json
instead of retraining this baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, LSTM, Dense
from tensorflow.keras.layers import TextVectorization


LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    train = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/validation.csv")

    max_tokens = 30000
    sequence_length = 128

    vectorizer = TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=sequence_length,
    )
    vectorizer.adapt(train["comment_text"].values)

    model = Sequential([
        vectorizer,
        Embedding(max_tokens, 128),
        LSTM(64),
        Dense(6, activation="sigmoid"),
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[],
    )
    model.fit(
        train["comment_text"].values,
        train[LABELS].values,
        validation_data=(val["comment_text"].values, val[LABELS].values),
        epochs=args.epochs,
        batch_size=64,
    )

    probs = model.predict(val["comment_text"].values)
    pred = (probs >= 0.5).astype(int)
    y = val[LABELS].values

    result = {
        "precision_micro": float(precision_score(y, pred, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(y, pred, average="micro", zero_division=0)),
        "f1_micro": float(f1_score(y, pred, average="micro", zero_division=0)),
        "trainable_parameters": int(model.count_params()),
    }

    Path("reports").mkdir(exist_ok=True)
    with open("reports/lstm_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
