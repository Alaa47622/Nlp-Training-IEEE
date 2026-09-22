"""LangChain + OpenRouter toxic-content classification logic."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter

load_dotenv()


@dataclass
class ClassificationResult:
    """Normalized result returned by the classifier."""

    classification: str
    explanation: str


SYSTEM_PROMPT = """You are a careful text-classification assistant.

Classify the user's text into exactly one of these labels:
- Toxic
- Non-Toxic

Toxic means the text contains abusive, hateful, harassing, threatening,
severely insulting, or intentionally harmful language directed at a person
or group.

Non-Toxic means normal conversation, neutral statements, questions,
constructive criticism, or harmless opinions.

Return ONLY valid JSON with exactly these two string fields:
{
  "classification": "Toxic" or "Non-Toxic",
  "explanation": "one short sentence explaining the classification"
}
"""


def _get_api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. Add it to .env or your environment."
        )
    return key


def create_model() -> ChatOpenRouter:
    """Create the OpenRouter chat model from environment configuration."""
    model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b").strip()
    return ChatOpenRouter(
        model=model_name,
        temperature=0,
        max_tokens=250,
        api_key=_get_api_key(),
    )


def _extract_json(content: str) -> dict[str, Any] | None:
    """Extract a JSON object even if the provider wrapped it in markdown."""
    text = content.strip()
    candidates = [text]

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1))

    object_match = re.search(r"\{.*\}", text, re.DOTALL)
    if object_match:
        candidates.append(object_match.group(0))

    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return None


def _normalize_result(content: str) -> ClassificationResult:
    data = _extract_json(content)
    if data:
        label = str(data.get("classification", "")).strip()
        explanation = str(data.get("explanation", "")).strip()
        if label.lower() in {"toxic", "non-toxic"}:
            label = "Toxic" if label.lower() == "toxic" else "Non-Toxic"
            return ClassificationResult(
                classification=label,
                explanation=explanation or "The model did not provide an explanation.",
            )

    # Fallback for a provider that ignores the JSON-only instruction.
    lowered = content.lower()
    if "non-toxic" in lowered:
        label = "Non-Toxic"
    elif "toxic" in lowered:
        label = "Toxic"
    else:
        raise ValueError(f"Could not parse model classification: {content}")

    return ClassificationResult(
        classification=label,
        explanation=content.strip(),
    )


def classify_text(text: str) -> ClassificationResult:
    """Classify text through LangChain and an OpenRouter-hosted LLM."""
    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    model = create_model()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=text.strip()),
    ]
    response = model.invoke(messages)

    content = response.content
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )

    return _normalize_result(str(content))
