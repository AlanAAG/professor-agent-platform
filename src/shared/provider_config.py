"""Utilities for choosing between Gemini and Mistral deployments."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal, Optional

Provider = Literal["gemini", "mistral"]

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_MISTRAL_MODEL = "mistral-large-latest"
DEFAULT_MISTRAL_EMBED_MODEL = "mistral-embed"
DEFAULT_MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
SUPPORTED_PROVIDERS: tuple[Provider, ...] = ("gemini", "mistral")


def _normalize(value: Optional[str]) -> str:
    return (value or "").strip()


def _looks_like_gemini_key(value: str) -> bool:
    """Heuristic: public Gemini keys usually begin with AIza or AI..."""
    return value.startswith("AIza") or value.startswith("AI")


@lru_cache(maxsize=1)
def get_llm_provider() -> Provider:
    """Infer the active LLM provider, defaulting to Gemini for backwards compatibility."""
    explicit = _normalize(os.getenv("LLM_PROVIDER")).lower()
    if explicit in SUPPORTED_PROVIDERS:
        return explicit  # type: ignore[return-value]

    mistral_key = _normalize(os.getenv("MISTRAL_API_KEY"))
    if mistral_key:
        return "mistral"

    gemini_key = _normalize(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    if gemini_key and not _looks_like_gemini_key(gemini_key):
        # User likely stored a non-Gemini key (e.g., Mistral) in GEMINI_API_KEY for convenience.
        return "mistral"

    return "gemini"


def is_mistral() -> bool:
    return get_llm_provider() == "mistral"


def is_gemini() -> bool:
    return get_llm_provider() == "gemini"


def get_gemini_api_key() -> str:
    return _normalize(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def get_mistral_api_key() -> str:
    """Return the configured Mistral key, falling back to GEMINI_API_KEY when provider=mistral."""
    key = _normalize(os.getenv("MISTRAL_API_KEY"))
    if key:
        return key
    if is_mistral():
        fallback = _normalize(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        if fallback and not _looks_like_gemini_key(fallback):
            return fallback
    return ""


def get_mistral_base_url() -> str:
    return _normalize(os.getenv("MISTRAL_BASE_URL")) or DEFAULT_MISTRAL_BASE_URL


def get_chat_model_name() -> str:
    if is_mistral():
        return _normalize(os.getenv("MISTRAL_MODEL")) or DEFAULT_MISTRAL_MODEL
    return _normalize(os.getenv("GEMINI_MODEL")) or DEFAULT_GEMINI_MODEL


def get_embedding_model_name() -> str:
    """Respect EMBEDDING_MODEL_NAME but provide sensible defaults per provider."""
    configured = _normalize(os.getenv("EMBEDDING_MODEL_NAME"))
    if configured:
        return configured
    if is_mistral():
        return DEFAULT_MISTRAL_EMBED_MODEL
    return "text-embedding-004"


def get_expected_embedding_dim() -> int:
    override = _normalize(os.getenv("EMBEDDING_DIM"))
    if override.isdigit():
        return int(override)
    if is_mistral():
        return 1024
    return 768
