from __future__ import annotations

import os
from typing import Iterable, List, Optional

from langchain_groq import ChatGroq

from utils.config import DEFAULT_MODEL, SUPPORTED_MODELS


_RATE_LIMIT_SIGNALS = (
    "429",
    "413",
    "rate limit",
    "rate-limited",
    "rate_limit_exceeded",
    "tokens per day",
    "request too large",
    "too large for model",
    "message size",
    "quota",
)


def _is_rate_limit_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(signal in message for signal in _RATE_LIMIT_SIGNALS)


def build_model_fallback_order(preferred_model: Optional[str] = None) -> List[str]:
    primary_model = preferred_model or DEFAULT_MODEL
    ordered_models: List[str] = []

    for model_name in [primary_model, *SUPPORTED_MODELS]:
        if model_name and model_name not in ordered_models:
            ordered_models.append(model_name)

    return ordered_models


def _create_llm(model_name: str) -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")
    return ChatGroq(temperature=0, model_name=model_name)


def invoke_with_fallback(prompt: str, preferred_model: Optional[str] = None):
    fallback_models = build_model_fallback_order(preferred_model)
    if not fallback_models:
        raise RuntimeError("No Groq models are configured.")

    last_rate_limit_error: Optional[Exception] = None
    last_other_error: Optional[Exception] = None

    for model_name in fallback_models:
        try:
            llm = _create_llm(model_name)
            return llm.invoke(prompt)
        except Exception as error:
            if _is_rate_limit_error(error):
                last_rate_limit_error = error
            else:
                last_other_error = error
            continue

    if last_rate_limit_error is not None:
        raise RuntimeError(
            "The selected model is temporarily rate-limited. Please retry in a bit or switch to another model from the sidebar."
        ) from last_rate_limit_error

    if last_other_error is not None:
        raise RuntimeError(
            "The language model is currently unavailable. Please try again later."
        ) from last_other_error

    raise RuntimeError("The language model is currently unavailable. Please try again later.")
