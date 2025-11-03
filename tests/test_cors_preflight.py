from __future__ import annotations

import importlib
import sys
import types

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def cors_test_client(monkeypatch) -> TestClient:
    """Create a TestClient with dependencies stubbed to focus on CORS behaviour."""

    monkeypatch.setenv("SECRET_API_KEY", "test-secret")
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv("ALLOWED_ORIGIN_REGEX", raising=False)
    monkeypatch.setenv("ALLOW_DYNAMIC_LOVABLE_ORIGINS", "true")
    monkeypatch.setenv("LOVABLE_DYNAMIC_SUFFIXES", "app")

    for module_name in [
        "api_server",
        "src.shared.utils",
        "google.generativeai",
        "google",
    ]:
        sys.modules.pop(module_name, None)

    utils_stub = types.ModuleType("src.shared.utils")
    utils_stub.EMBEDDING_MODEL_NAME = "test-model"

    def _empty(*args, **kwargs):
        return []

    utils_stub.cohere_rerank = lambda query, documents: documents
    utils_stub.retrieve_rag_documents = _empty
    utils_stub.retrieve_rag_documents_keyword_fallback = _empty
    utils_stub._get_supabase_client = lambda: None

    sys.modules["src.shared.utils"] = utils_stub

    google_package = types.ModuleType("google")
    genai_stub = types.ModuleType("google.generativeai")
    genai_stub.Client = lambda *args, **kwargs: None
    genai_stub.GenerativeModel = object

    sys.modules["google"] = google_package
    sys.modules["google.generativeai"] = genai_stub

    app_module = importlib.import_module("api_server")

    client = TestClient(app_module.app)

    yield client

    sys.modules.pop("api_server", None)
    sys.modules.pop("src.shared.utils", None)
    sys.modules.pop("google.generativeai", None)
    sys.modules.pop("google", None)


def test_lovable_preview_preflight_allowed(cors_test_client: TestClient) -> None:
    origin = "https://random-preview--1234abcd.lovable.app"
    response = cors_test_client.options(
        "/api/chat",
        headers={
            "origin": origin,
            "access-control-request-method": "POST",
            "access-control-request-headers": "content-type,x-api-key",
        },
    )

    assert response.status_code == 200

    headers = {k.lower(): v for k, v in response.headers.items()}
    assert headers["access-control-allow-origin"] == origin
    assert "post" in headers["access-control-allow-methods"].lower()
    assert "options" in headers["access-control-allow-methods"].lower()

    allowed_headers = headers.get("access-control-allow-headers", "")
    assert "content-type" in allowed_headers.lower()
    assert "x-api-key" in allowed_headers.lower()
