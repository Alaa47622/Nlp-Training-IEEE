# Task 3 Results Section

## Model choice

We selected DistilBERT and applied LoRA because it provides a compact Transformer architecture suitable for efficient parameter-efficient fine-tuning.

## LoRA modules

LoRA was applied to the DistilBERT attention `q_lin` and `v_lin` projection modules. These modules were selected because they control the query and value transformations in self-attention.

## Evaluation

After running `uv run python src/evaluate.py`, copy the actual values from `reports/metrics.json`.

| Metric | DistilBERT + LoRA |
|---|---:|
| Accuracy | INSERT |
| Precision | INSERT |
| Recall | INSERT |
| F1-score | INSERT |
| Macro F1 | INSERT |

## Confusion Matrix

Insert the generated six confusion-matrix images from `reports/`.

## Comparison with LSTM

| Category | Previous LSTM | DistilBERT + LoRA |
|---|---:|---:|
| Accuracy | INSERT | INSERT |
| Precision | INSERT | INSERT |
| Recall | INSERT | INSERT |
| F1-score | INSERT | INSERT |
| Trainable parameters | INSERT | INSERT |
| Training time | INSERT | INSERT |
| Memory | INSERT | INSERT |

## Discussion

The final discussion should be based on the measured results. Explain whether LoRA achieved better classification performance and whether the reduction in trainable parameters and memory requirements improved overall efficiency.
