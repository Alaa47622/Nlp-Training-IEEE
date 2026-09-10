import pandas as pd
from sklearn.model_selection import train_test_split

from toxic_classifier.config import load_config, path
from toxic_classifier.features.vocab import Vocabulary
from toxic_classifier.utils import clean_text, save_json, set_seed


def main():
    cfg = load_config()
    set_seed(cfg["seed"])

    raw = path("data/raw/train.csv")
    if not raw.exists():
        raise FileNotFoundError(
            "Put Kaggle train.csv at data/raw/train.csv before running preparation."
        )

    labels = cfg["data"]["labels"]
    df = pd.read_csv(raw)
    required = [cfg["data"]["text_column"], *labels]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[required].dropna(subset=[cfg["data"]["text_column"]]).copy()
    df["comment_text"] = df["comment_text"].map(clean_text)

    train, temp = train_test_split(
        df, test_size=cfg["data"]["test_size"] + cfg["data"]["val_size"],
        random_state=cfg["seed"]
    )
    relative_test = cfg["data"]["test_size"] / (
        cfg["data"]["test_size"] + cfg["data"]["val_size"]
    )
    val, test = train_test_split(temp, test_size=relative_test, random_state=cfg["seed"])

    out = path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    val.to_csv(out / "val.csv", index=False)
    test.to_csv(out / "test.csv", index=False)

    vocab = Vocabulary(cfg["data"]["max_vocab_size"], cfg["data"]["min_frequency"])
    vocab.build(train["comment_text"].tolist())
    vocab.save(out / "vocab.json")

    stats = {
        "train": len(train), "validation": len(val), "test": len(test),
        "vocab_size": len(vocab),
        "positive_counts": {c: int(train[c].sum()) for c in labels},
    }
    save_json(stats, out / "stats.json")
    print(stats)

if __name__ == "__main__":
    main()
