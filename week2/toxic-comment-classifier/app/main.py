from fastapi import FastAPI
from pydantic import BaseModel

from toxic_classifier.inference import Predictor

app = FastAPI(title="Toxic Comment Classifier", version="0.1.0")
predictor = None

class Comment(BaseModel):
    text: str

@app.on_event("startup")
def startup():
    global predictor
    predictor = Predictor()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(comment: Comment):
    return {"text": comment.text, "probabilities": predictor.predict(comment.text)}
