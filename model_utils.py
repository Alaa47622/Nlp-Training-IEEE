from __future__ import annotations

from pathlib import Path
import pandas as pd


TEXT_CANDIDATES = [
    "text", "review", "reviews", "content", "comment", "feedback",
    "review_text", "review text", "arabic_review", "sentence"
]

LABEL_CANDIDATES = [
    "label", "sentiment", "polarity", "class", "target", "rating"
]


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        # sep=None lets pandas sniff comma/semicolon/tab in many CSV files.
        return pd.read_csv(path, sep=None, engine="python")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")

    raise ValueError(f"Unsupported dataset format: {suffix}")


def choose_column(df: pd.DataFrame, preferred: str | None, candidates: list[str]) -> str:
    if preferred:
        if preferred not in df.columns:
            raise ValueError(
                f"Column '{preferred}' was not found. Available columns: {list(df.columns)}"
            )
        return preferred

    lower_map = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]

    # Fallback: choose first object/string column for text.
    if candidates is TEXT_CANDIDATES:
        object_cols = [c for c in df.columns if df[c].dtype == "object"]
        if object_cols:
            return object_cols[0]

    raise ValueError(
        f"Could not detect a suitable column automatically. "
        f"Available columns: {list(df.columns)}"
    )


def normalize_binary_sentiment(series: pd.Series) -> pd.Series:
    """
    Convert common binary sentiment labels to 'positive' / 'negative'.
    Unrecognized rows become NA and can be dropped.
    """
    def convert(value):
        if pd.isna(value):
            return pd.NA

        if isinstance(value, bool):
            return "positive" if value else "negative"

        # Numeric labels.
        try:
            number = float(value)
            if number == 1:
                return "positive"
            if number == 0:
                return "negative"
            # Common star-rating fallback.
            if 1 <= number <= 2:
                return "negative"
            if 4 <= number <= 5:
                return "positive"
        except (TypeError, ValueError):
            pass

        value = str(value).strip().lower()

        positives = {
            "positive", "pos", "p", "1", "good", "موجب", "ايجابي", "إيجابي",
            "positive review"
        }
        negatives = {
            "negative", "neg", "n", "0", "bad", "سالب", "سلبي",
            "negative review"
        }

        if value in positives:
            return "positive"
        if value in negatives:
            return "negative"
        return pd.NA

    return series.map(convert)
