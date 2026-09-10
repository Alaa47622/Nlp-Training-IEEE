import json

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from toxic_classifier.config import load_config, path
from toxic_classifier.data.dataset import ToxicDataset
from toxic_classifier.features.vocab import Vocabulary
from toxic_classifier.models.lstm import ToxicLSTM
from toxic_classifier.utils import set_seed


def class_weights(df, labels):
    positives = df[labels].sum().values
    negatives = len(df) - positives
    return torch.tensor(negatives / np.maximum(positives, 1), dtype=torch.float32)

def run_epoch(model, loader, loss_fn, optimizer, device, train=True):
    model.train(train)
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += loss.item() * len(x)
    return total / len(loader.dataset)

def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    labels = cfg["data"]["labels"]
    proc = path("data/processed")

    train_df = pd.read_csv(proc / "train.csv")
    val_df = pd.read_csv(proc / "val.csv")
    vocab = Vocabulary.load(proc / "vocab.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = ToxicDataset(train_df, vocab, labels, cfg["data"]["max_length"])
    val_ds = ToxicDataset(val_df, vocab, labels, cfg["data"]["max_length"])
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"])

    model = ToxicLSTM(len(vocab), cfg["model"]["embedding_dim"], cfg["model"]["hidden_dim"],
                      cfg["model"]["num_layers"], cfg["model"]["dropout"],
                      cfg["model"]["bidirectional"], len(labels)).to(device)

    weights = class_weights(train_df, labels).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                                   weight_decay=cfg["training"]["weight_decay"])

    mlflow.set_tracking_uri(cfg.get("mlflow_tracking_uri", "file:./mlruns"))
    mlflow.set_experiment("toxic-comment-lstm")

    best = float("inf")
    bad_epochs = 0
    model_dir = path("models")
    model_dir.mkdir(exist_ok=True)

    with mlflow.start_run():
        mlflow.log_params({
            "embedding_dim": cfg["model"]["embedding_dim"],
            "hidden_dim": cfg["model"]["hidden_dim"],
            "max_length": cfg["data"]["max_length"],
            "batch_size": cfg["training"]["batch_size"],
            "learning_rate": cfg["training"]["learning_rate"],
            "bidirectional": cfg["model"]["bidirectional"],
        })
        for epoch in range(1, cfg["training"]["epochs"] + 1):
            tr = run_epoch(model, train_loader, loss_fn, optimizer, device, True)
            va = run_epoch(model, val_loader, loss_fn, optimizer, device, False)
            print(f"epoch={epoch} train_loss={tr:.4f} val_loss={va:.4f}")
            mlflow.log_metrics({"train_loss": tr, "val_loss": va}, step=epoch)

            if va < best:
                best = va
                bad_epochs = 0
                torch.save(model.state_dict(), model_dir / "model.pt")
            else:
                bad_epochs += 1
                if bad_epochs >= cfg["training"]["patience"]:
                    print("Early stopping.")
                    break

        (model_dir / "metadata.json").write_text(json.dumps({
            "labels": labels, "vocab_size": len(vocab),
            "max_length": cfg["data"]["max_length"], "device": str(device)
        }, indent=2), encoding="utf-8")
        mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    main()
