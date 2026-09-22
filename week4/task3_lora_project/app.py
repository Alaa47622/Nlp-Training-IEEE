from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from peft import AutoPeftModelForSequenceClassification
from transformers import AutoTokenizer


LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

MODEL_DIR = Path("artifacts/lora_distilbert")

st.set_page_config(
    page_title="Toxicity Detector",
    page_icon="🛡️",
    layout="centered",
)


@st.cache_resource
def load_model():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            "Trained model not found. Run the preprocessing and training commands first."
        )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoPeftModelForSequenceClassification.from_pretrained(MODEL_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict(text: str, threshold: float = 0.5):
    tokenizer, model, device = load_model()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probabilities = torch.sigmoid(logits).cpu().numpy()[0]
    predictions = probabilities >= threshold

    return probabilities, predictions


st.title("🛡️ Toxicity Detection System")
st.caption("DistilBERT + LoRA • Multi-label text classification")

st.markdown(
    """
Enter a comment below and the fine-tuned model will estimate the probability
of six toxicity categories.
"""
)

text = st.text_area(
    "Enter your comment",
    placeholder="Write a comment here...",
    height=150,
)

threshold = st.slider(
    "Classification threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
)

if st.button("🔍 Analyze Comment", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Please enter a comment first.")
    else:
        try:
            with st.spinner("Analyzing..."):
                probabilities, predictions = predict(text.strip(), threshold)

            st.subheader("Results")

            overall = bool(predictions.any())
            if overall:
                st.warning("⚠️ At least one toxicity category was detected.")
            else:
                st.success("✅ No toxicity category crossed the selected threshold.")

            cols = st.columns(2)
            for i, label in enumerate(LABELS):
                with cols[i % 2]:
                    st.metric(
                        label.replace("_", " ").title(),
                        f"{probabilities[i] * 100:.1f}%",
                        "Detected" if predictions[i] else "Not detected",
                    )
                    st.progress(float(probabilities[i]))

            st.subheader("Prediction table")
            rows = []
            for i, label in enumerate(LABELS):
                rows.append(
                    {
                        "Category": label.replace("_", " ").title(),
                        "Probability": f"{probabilities[i] * 100:.2f}%",
                        "Prediction": "Detected" if predictions[i] else "Not detected",
                    }
                )
            st.dataframe(rows, use_container_width=True, hide_index=True)

            with st.expander("Model information"):
                st.write("**Base model:** DistilBERT")
                st.write("**Fine-tuning:** LoRA")
                st.write("**LoRA target modules:** `q_lin`, `v_lin`")
                st.write("**Task:** Multi-label toxicity classification")

        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
