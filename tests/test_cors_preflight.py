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
    utils_stub._get_supabase_client = lambda: None

    sys.modules["src.shared.utils"] = utils_stub

    google_package = types.ModuleType("google")
    genai_stub = types.ModuleType("google.generativeai")
    genai_stub.Client = lambda *args, **kwargs: None
    genai_stub.GenerativeModel = object

    sys.modules["google"] = google_package
    sys.modules["google.generativeai"] = genai_stub

    # Mock google.ai which is used by langchain_google_genai
    google_ai_package = types.ModuleType("google.ai")
    google_package.ai = google_ai_package
    sys.modules["google.ai"] = google_ai_package

    # Mock google.ai.generativelanguage_v1beta used by langchain_google_genai
    google_ai_genlang_package = types.ModuleType("google.ai.generativelanguage_v1beta")
    google_ai_package.generativelanguage_v1beta = google_ai_genlang_package
    sys.modules["google.ai.generativelanguage_v1beta"] = google_ai_genlang_package

    # Mock SafetySetting class and attributes
    class SafetySetting:
        class HarmBlockThreshold:
            pass
        class HarmCategory:
            pass
    google_ai_genlang_package.SafetySetting = SafetySetting
    google_ai_genlang_package.HarmCategory = SafetySetting.HarmCategory

    class GenerationConfig:
        class Modality:
            pass
        class MediaResolution:
            pass
    google_ai_genlang_package.GenerationConfig = GenerationConfig

    # Mock google.protobuf
    google_protobuf_package = types.ModuleType("google.protobuf")
    google_package.protobuf = google_protobuf_package
    sys.modules["google.protobuf"] = google_protobuf_package

    # Mock descriptor_pb2
    descriptor_pb2 = types.ModuleType("google.protobuf.descriptor_pb2")
    google_protobuf_package.descriptor_pb2 = descriptor_pb2
    sys.modules["google.protobuf.descriptor_pb2"] = descriptor_pb2

    # Mock rag_core to avoid importing langchain and google dependencies
    rag_core_stub = types.ModuleType("src.app.rag_core")
    rag_core_stub.classify_subject = lambda *args, **kwargs: "Startup"
    rag_core_stub.PROFESSOR_PERSONAS = {}
    rag_core_stub.DEFAULT_PERSONA_KEY = "Startup"
    rag_core_stub._get_fallback_persona = lambda: {}
    sys.modules["src.app.rag_core"] = rag_core_stub

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


def test_lovable_project_preflight_allowed(cors_test_client: TestClient) -> None:
    origin = "https://93a4e185-b263-4e0d-83e0-9cf4863ef461.lovableproject.com"
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
