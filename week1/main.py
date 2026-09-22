from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from Language_classifier import LanguageClassifier
from English_model import EnglishSentimentModel
from Arabic_model import ArabicSentimentModel
from Preprocessing_pipeline import (
    preprocess_arabic,
    preprocess_english,
    preprocess_for_language_detection,
)


language_model = LanguageClassifier()
english_model = EnglishSentimentModel()
arabic_model = ArabicSentimentModel()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once when the API starts.
    language_model.load()
    english_model.load()
    arabic_model.load()
    yield


app = FastAPI(
    title="Classical Multilingual NLP API",
    description=(
        "Classical NLP pipeline for Arabic/English language detection "
        "and language-specific sentiment analysis."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["This movie was excellent!"])


class PredictionResponse(BaseModel):
    user_text: str
    language: str
    sentiment_classification: str


@app.get("/")
def root():
    return {
        "message": "Classical NLP API is running.",
        "docs": "/docs",
        "prediction_endpoint": "POST /predict",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TextRequest):
    original_text = request.text.strip()

    if not original_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    # Stage 1: language detection.
    language_input = preprocess_for_language_detection(original_text)
    language = language_model.predict(language_input)

    # Stage 2: route to the correct sentiment model.
    if language == "Arabic":
        sentiment_input = preprocess_arabic(original_text)
        sentiment = arabic_model.predict(sentiment_input)
    elif language == "English":
        sentiment_input = preprocess_english(original_text)
        sentiment = english_model.predict(sentiment_input)
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected language prediction: {language}",
        )

    # Exactly the three main pieces requested by the project.
    return PredictionResponse(
        user_text=original_text,
        language=language,
        sentiment_classification=sentiment,
    )
