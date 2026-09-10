# Toxic Comment Classifier — LSTM + MLOps

End-to-end implementation of the Jigsaw Toxic Comment Classification task:
- multi-label classification
- six toxicity labels
- text preprocessing and vocabulary
- Embedding + LSTM
- BCEWithLogitsLoss with class imbalance weighting
- Precision, Recall, F1, ROC-AUC, PR-AUC
- MLflow experiment tracking
- FastAPI inference
- Docker
- GitHub Actions CI/CD
- pytest + Ruff

## 1. Setup on Windows

Open PowerShell:

```powershell
uv venv
.venv\Scripts\Activate.ps1
uv sync
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 2. Dataset

Download `train.csv` from the Kaggle Jigsaw Toxic Comment Classification Challenge dataset and place it here:

```text
data/raw/train.csv
```

The CSV must contain:
`comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate`

## 3. Build vocabulary + splits

```powershell
uv run python -m toxic_classifier.data.prepare
```

## 4. Train

```powershell
uv run python -m toxic_classifier.training.train
```

MLflow results are written to `mlruns/`.

View the UI:

```powershell
uv run mlflow ui
```

Then open the local address printed by MLflow.

## 5. Evaluate

```powershell
uv run python -m toxic_classifier.evaluation.evaluate
```

## 6. API

```powershell
uv run uvicorn app.main:app --reload
```

Open Swagger at the local `/docs` page.

Example request:

```json
{"text":"You are a terrible person"}
```

## 7. Tests

```powershell
uv run pytest
uv run ruff check .
```

## 8. Docker

```powershell
docker build -t toxic-classifier .
docker run -p 8000:8000 toxic-classifier
```

## Learning path

Start with the code in this order:
1. `config.yaml`
2. `data/prepare.py`
3. `features/vocab.py`
4. `data/dataset.py`
5. `models/lstm.py`
6. `training/train.py`
7. `evaluation/metrics.py`
8. `app/main.py`
9. Docker and GitHub Actions

Do not treat the project as a black box: change one component at a time and rerun the tests/experiments.
