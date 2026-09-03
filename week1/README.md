# Classical NLP + FastAPI Project

This project performs:

1. Arabic / English language detection.
2. Language-specific binary sentiment analysis.
3. REST API deployment using FastAPI.

It uses classical NLP only:
- Text preprocessing
- TF-IDF
- word n-grams / character n-grams
- Logistic Regression

No Transformers or deep learning are used.

## Project structure

```text
classical_nlp_fastapi_project/
├── data/
│   ├── arabic_reviews.csv
│   └── english_reviews.csv
├── models/
│   ├── Arabic_model_weights.pkl
│   ├── English_model_weights.pkl
│   └── Language_classifier_weights.pkl
├── Arabic_model.py
├── English_model.py
├── Language_classifier.py
├── Preprocessing_pipeline.py
├── model_utils.py
├── train_arabic_model.py
├── train_english_model.py
├── train_language_classifier.py
├── main.py
└── requirements.txt
```

## 1. Put the datasets in `data/`

Rename the downloaded files to:

```text
data/arabic_reviews.csv
data/english_reviews.csv
```

The training scripts try to detect common text and label column names automatically.
You can also specify them explicitly with command-line arguments.

## 2. Create environment and install packages

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Train all models

```bash
python train_language_classifier.py
python train_english_model.py
python train_arabic_model.py
```

If a dataset uses unusual column names, for example:

```bash
python train_english_model.py --text-col review --label-col label
python train_arabic_model.py --text-col Review --label-col Sentiment
```

After training, the `.pkl` files will appear inside `models/`.

## 4. Start FastAPI

```bash
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Use `POST /predict`.

Example JSON request:

```json
{
  "text": "This movie was amazing"
}
```

Example response:

```json
{
  "user_text": "This movie was amazing",
  "language": "English",
  "sentiment_classification": "positive"
}
```

Arabic example:

```json
{
  "text": "الخدمة ممتازة جدا"
}
```

Possible response:

```json
{
  "user_text": "الخدمة ممتازة جدا",
  "language": "Arabic",
  "sentiment_classification": "positive"
}
```
