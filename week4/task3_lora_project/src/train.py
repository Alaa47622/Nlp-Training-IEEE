from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.utils import LABELS, ensure_dir, load_params, save_json, set_seed


def make_dataset(df: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    ds = Dataset.from_pandas(df[["comment_text", *LABELS]], preserve_index=False)

    def tokenize(batch):
        encoded = tokenizer(
            batch["comment_text"],
            truncation=True,
            max_length=max_length,
        )
        encoded["labels"] = [
            [float(batch[label][i]) for label in LABELS]
            for i in range(len(batch["comment_text"]))
        ]
        return encoded

    return ds.map(tokenize, batched=True, remove_columns=ds.column_names)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    return {
        "f1": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    data_cfg = params["data"]
    model_cfg = params["model"]
    train_cfg = params["training"]
    set_seed(data_cfg["seed"])

    train_path = Path(data_cfg["processed_dir"]) / "train.csv"
    val_path = Path(data_cfg["processed_dir"]) / "validation.csv"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            "Processed data not found. Run: uv run python src/preprocess.py"
        )

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["name"],
        num_labels=model_cfg["num_labels"],
        problem_type=model_cfg["problem_type"],
        id2label={i: label for i, label in enumerate(LABELS)},
        label2id={label: i for i, label in enumerate(LABELS)},
    )

    # DistilBERT's attention projection layers are q_lin and v_lin.
    # LoRA adapts these low-rank matrices while freezing the original model.
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=model_cfg["lora_r"],
        lora_alpha=model_cfg["lora_alpha"],
        lora_dropout=model_cfg["lora_dropout"],
        target_modules=model_cfg["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = make_dataset(train_df, tokenizer, data_cfg["max_length"])
    val_ds = make_dataset(val_df, tokenizer, data_cfg["max_length"])

    output_dir = Path(train_cfg["output_dir"])
    ensure_dir(output_dir)

    use_fp16 = bool(train_cfg["fp16"] and torch.cuda.is_available())

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        learning_rate=train_cfg["learning_rate"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=train_cfg["warmup_ratio"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy=train_cfg["eval_strategy"],
        save_strategy=train_cfg["save_strategy"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=train_cfg["greater_is_better"],
        fp16=use_fp16,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Record LoRA-specific information for the report.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    summary = {
        "model": model_cfg["name"],
        "lora_target_modules": model_cfg["target_modules"],
        "lora_r": model_cfg["lora_r"],
        "lora_alpha": model_cfg["lora_alpha"],
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percentage": 100 * trainable / total,
        "evaluation": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
    }
    save_json(summary, output_dir / "training_summary.json")


if __name__ == "__main__":
    main()
