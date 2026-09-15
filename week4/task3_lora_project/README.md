# Task 3 — Fine-Tuning DistilBERT Using LoRA

## 1. Assignment objective

This project implements Task 3 using **DistilBERT + LoRA** for the same multi-label toxicity classification dataset.

The dataset contains:

- `comment_text`
- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

### Why DistilBERT?

DistilBERT was selected because it is a compact Transformer model that is suitable for text classification and makes a practical choice for parameter-efficient LoRA fine-tuning.

### Why LoRA?

LoRA freezes the pretrained Transformer weights and learns small low-rank update matrices. Therefore, only a small fraction of the model parameters are trainable.

---

## 2. LoRA architecture

For DistilBERT, this project adapts:

- `q_lin` — query projection
- `v_lin` — value projection

These are the attention projection layers. Adapting them allows LoRA to modify how the model attends to important parts of the input while keeping the original pretrained weights frozen.

The LoRA update can be viewed as:

`W' = W + B A`

where `W` is the frozen pretrained matrix and `A`, `B` are the trainable low-rank matrices.

Configuration:

- rank `r = 8`
- alpha `= 16`
- dropout `= 0.10`

---

## 3. Dataset

The supplied `train.csv` contains 159,571 rows and eight columns:

`id`, `comment_text`, and six binary toxicity labels.

The task is **multi-label classification**, because one comment can belong to more than one toxicity category.

---

## 4. Project structure

```text
task3_lora_project/
├── data/
│   └── raw/train.csv
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── compare_results.py
│   ├── lstm_baseline.py
│   └── utils.py
├── tests/
├── configs/
├── params.yaml
├── dvc.yaml
├── Dockerfile
├── pyproject.toml
└── .github/workflows/ci-cd.yml
```

---

## 5. Setup with uv

Install `uv`, then:

```bash
uv sync
```

If you want to regenerate the lock file:

```bash
uv lock
```

---

## 6. Run preprocessing

```bash
uv run python src/preprocess.py
```

For coursework demonstration, `params.yaml` caps the training set at 30,000 rows and validation at 5,000 rows to make experimentation manageable.

For a full-data experiment, set:

```yaml
max_train_samples: null
max_validation_samples: null
```

---

## 7. Train LoRA model

```bash
uv run python src/train.py
```

The first run downloads the pretrained DistilBERT model from Hugging Face.

The trained adapter/model is stored under:

```text
artifacts/lora_distilbert/
```

---

## 8. Evaluate

```bash
uv run python src/evaluate.py
```

Outputs include:

- Accuracy
- Precision
- Recall
- Micro F1
- Macro F1
- Per-label Precision/Recall/F1
- One confusion matrix for each toxicity label

Reports are stored in:

```text
reports/
```

---

## 9. LSTM comparison

The assignment asks for comparison with the previous LSTM implementation.

### Preferred method

Use the metrics from your previous LSTM project and save them as:

```text
reports/lstm_results.json
```

Example:

```json
{
  "precision_micro": 0.80,
  "recall_micro": 0.70,
  "f1_micro": 0.74,
  "trainable_parameters": 2500000
}
```

Then:

```bash
uv run python src/compare_results.py
```

### Optional fallback

If the old LSTM code/results are unavailable, the project contains:

```bash
uv run python src/lstm_baseline.py
```

This requires TensorFlow to be added to the environment separately because TensorFlow is not needed for the LoRA project itself.

---

## 10. DVC

Initialize DVC if needed:

```bash
uv run dvc init
```

The project contains a reproducible pipeline:

```text
raw data
   ↓
preprocess
   ↓
processed train/validation data
   ↓
LoRA training
   ↓
trained model
   ↓
evaluation
   ↓
metrics + confusion matrices
```

Run:

```bash
uv run dvc repro
```

Check the pipeline:

```bash
uv run dvc dag
```

For a remote DVC repository, configure a remote such as your team's storage and then:

```bash
uv run dvc push
```

---

## 11. Docker

Build:

```bash
docker build -t task3-lora-distilbert .
```

Run preprocessing:

```bash
docker run --rm task3-lora-distilbert
```

For actual model training, GPU support is recommended. On a machine configured for NVIDIA Container Toolkit, the container can be run with GPU access.

---

## 12. CI/CD

GitHub Actions is configured in:

```text
.github/workflows/ci-cd.yml
```

Every push/pull request performs:

1. Checkout
2. Install uv
3. Install Python
4. Install dependencies
5. Ruff linting
6. Pytest
7. Docker build

A push to `main` additionally passes through a deployment gate. Registry deployment can be enabled later using GitHub repository secrets.

---

## 13. Expected comparison discussion

After running both models, discuss:

### Performance
Compare Precision, Recall and F1. Transformer representations are generally expected to provide stronger contextual understanding than a basic LSTM, but the actual conclusion must be based on your measured results.

### Training time
LoRA reduces the number of parameters that need to be updated, but Transformer training can still require significant compute.

### Trainable parameters
LoRA trains only its adapter parameters rather than updating the entire DistilBERT model.

### Memory
Because most pretrained parameters remain frozen, LoRA can reduce optimization memory compared with full fine-tuning.

### Overall efficiency
The final conclusion should balance predictive performance, training time, trainable parameter count and hardware requirements.

---

## 14. Commands — quick start

```bash
uv sync

uv run python src/preprocess.py

uv run python src/train.py

uv run python src/evaluate.py

uv run python src/compare_results.py
```

Or with DVC:

```bash
uv run dvc repro
```

---

## 15. Important note about results

Do not invent numerical results for the report.

Run the training/evaluation scripts and copy the actual values from:

```text
reports/metrics.json
```

Then compare them with the actual results from the previous LSTM implementation.

---

## 16. Streamlit UI

The project includes a complete Streamlit interface in:

```text
app.py
```

The UI provides:

- Comment input
- Classification threshold control
- Analyze button
- Probability for each of the six labels
- Detected/not-detected status
- Overall result
- Prediction table
- Model information

### Run locally

After training the model:

```bash
uv run streamlit run app.py
```

Open the displayed local address, normally:

```text
http://localhost:8501
```

### Run the UI with Docker

Build:

```bash
docker build -f Dockerfile.ui -t task3-lora-ui .
```

Run:

```bash
docker run --rm -p 8501:8501 -v "$(pwd)/artifacts:/app/artifacts:ro" task3-lora-ui
```

On Windows PowerShell, use:

```powershell
docker run --rm -p 8501:8501 -v "${PWD}/artifacts:/app/artifacts:ro" task3-lora-ui
```

### Docker Compose

```bash
docker compose up --build
```

Then open:

```text
http://localhost:8501
```

The UI loads the actual trained LoRA adapter from:

```text
artifacts/lora_distilbert/
```

It does not use a fake/demo prediction model.

---

## 17. Final architecture

```text
                    ┌─────────────────────┐
                    │   Streamlit UI      │
                    │   User comment      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ DistilBERT Tokenizer│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ DistilBERT + LoRA   │
                    │ q_lin + v_lin       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ 6 Toxicity Scores   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Results Dashboard   │
                    └─────────────────────┘
```

The complete engineering workflow is:

```text
Dataset
   ↓
DVC
   ↓
Preprocessing
   ↓
DistilBERT + LoRA training
   ↓
Evaluation
   ↓
Saved model
   ↓
Streamlit UI
   ↓
Docker
   ↓
GitHub Actions CI/CD
```
