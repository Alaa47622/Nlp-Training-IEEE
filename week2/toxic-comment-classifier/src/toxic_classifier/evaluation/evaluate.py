import json

import pandas as pd
import torch
from torch.utils.data import DataLoader

from toxic_classifier.config import load_config, path
from toxic_classifier.data.dataset import ToxicDataset
from toxic_classifier.evaluation.metrics import multilabel_metrics
from toxic_classifier.features.vocab import Vocabulary
from toxic_classifier.models.lstm import ToxicLSTM


def main():
    cfg = load_config()
    labels = cfg["data"]["labels"]
    proc = path("data/processed")
    df = pd.read_csv(proc / "test.csv")
    vocab = Vocabulary.load(proc / "vocab.json")
    ds = ToxicDataset(df, vocab, labels, cfg["data"]["max_length"])
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ToxicLSTM(len(vocab), cfg["model"]["embedding_dim"], cfg["model"]["hidden_dim"],
                      cfg["model"]["num_layers"], cfg["model"]["dropout"],
                      cfg["model"]["bidirectional"], len(labels)).to(device)
    model.load_state_dict(torch.load(path("models/model.pt"), map_location=device))
    model.eval()

    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            prob = torch.sigmoid(model(x.to(device))).cpu()
            ys.append(y.numpy()); ps.append(prob.numpy())

    import numpy as np
    metrics = multilabel_metrics(np.vstack(ys), np.vstack(ps), labels)
    print(json.dumps(metrics, indent=2))
    path("artifacts").mkdir(exist_ok=True)
    (path("artifacts") / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
