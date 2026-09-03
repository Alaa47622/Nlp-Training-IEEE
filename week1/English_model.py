from __future__ import annotations

import joblib
from pathlib import Path


class EnglishSentimentModel:
    def __init__(self, model_path: str | Path = "models/English_model_weights.pkl"):
        self.model_path = Path(model_path)
        self.model = None

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"English sentiment model not found: {self.model_path}. "
                "Run train_english_model.py first."
            )
        self.model = joblib.load(self.model_path)
        return self

    def predict(self, text: str) -> str:
        if self.model is None:
            self.load()
        return str(self.model.predict([text])[0])

    def predict_with_confidence(self, text: str):
        if self.model is None:
            self.load()

        label = str(self.model.predict([text])[0])
        confidence = None

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba([text])[0]
            confidence = float(max(probs))

        return label, confidence
