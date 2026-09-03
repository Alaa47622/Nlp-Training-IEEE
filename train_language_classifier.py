from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from Preprocessing_pipeline import preprocess_for_language_detection
from model_utils import load_table, choose_column, TEXT_CANDIDATES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arabic", default="data/arabic_reviews.csv")
    parser.add_argument("--english", default="data/english_reviews.csv")
    parser.add_argument("--arabic-text-col", default=None)
    parser.add_argument("--english-text-col", default=None)
    parser.add_argument(
        "--output",
        default="models/Language_classifier_weights.pkl"
    )
    args = parser.parse_args()

    ar_df = load_table(args.arabic)
    en_df = load_table(args.english)

    ar_text_col = choose_column(ar_df, args.arabic_text_col, TEXT_CANDIDATES)
    en_text_col = choose_column(en_df, args.english_text_col, TEXT_CANDIDATES)

    ar = pd.DataFrame({
        "text": ar_df[ar_text_col].astype(str),
        "language": "Arabic"
    })
    en = pd.DataFrame({
        "text": en_df[en_text_col].astype(str),
        "language": "English"
    })

    # Balance the classes so one dataset does not dominate.
    n = min(len(ar), len(en))
    ar = ar.sample(n=n, random_state=42)
    en = en.sample(n=n, random_state=42)

    df = pd.concat([ar, en], ignore_index=True)
    df["text"] = df["text"].map(preprocess_for_language_detection)
    df = df[df["text"].str.len() > 0].drop_duplicates("text")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["language"],
        test_size=0.20,
        random_state=42,
        stratify=df["language"],
    )

    # Character n-grams are classical NLP features and excellent for
    # distinguishing Arabic script from English text.
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 5),
                min_df=2,
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

    print("\nLanguage Classifier")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print(classification_report(y_test, predictions))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output)
    print(f"Saved model to: {output}")


if __name__ == "__main__":
    main()
