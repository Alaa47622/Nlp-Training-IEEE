from __future__ import annotations

import argparse
from pathlib import Path
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from Preprocessing_pipeline import preprocess_english
from model_utils import (
    load_table, choose_column, normalize_binary_sentiment,
    TEXT_CANDIDATES, LABEL_CANDIDATES,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/english_reviews.csv")
    parser.add_argument("--text-col", default=None)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--output", default="models/English_model_weights.pkl")
    args = parser.parse_args()

    df = load_table(args.data)
    text_col = choose_column(df, args.text_col, TEXT_CANDIDATES)
    label_col = choose_column(df, args.label_col, LABEL_CANDIDATES)

    X = df[text_col].astype(str).map(preprocess_english)
    y = normalize_binary_sentiment(df[label_col])

    valid = X.str.len().gt(0) & y.notna()
    X = X[valid]
    y = y[valid]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                max_features=100_000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    print("\nEnglish Sentiment Model")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print(classification_report(y_test, predictions))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output)
    print(f"Saved model to: {output}")


if __name__ == "__main__":
    main()
