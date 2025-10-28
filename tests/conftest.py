import types
import pytest


@pytest.fixture(scope="session", autouse=True)
def set_test_env(monkeypatch):
    """Set required environment variables so modules initialize without real secrets."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-supabase-key")
    monkeypatch.setenv("COACH_USERNAME", "test-user")
    monkeypatch.setenv("SECRET_API_KEY", "secret")
    # Provide defaults used by utils
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "models/embedding-001")


@pytest.fixture()
def mock_genai(monkeypatch):
    """Mock google.generativeai's embed_content to return deterministic embeddings."""
    try:
        import src.shared.utils as utils
    except Exception:
        return  # Not needed for tests that don't import utils

    class FakeGenAI:
        def configure(self, **kwargs):
            return None

        def embed_content(self, model=None, content=None, task_type=None):
            # Deterministic small embedding for tests
            def fixed_vec():
                return [0.1, 0.2, 0.3, 0.4]

            if isinstance(content, list):
                return {"embedding": [fixed_vec() for _ in content]}
            return {"embedding": fixed_vec()}

    monkeypatch.setattr(utils, "genai", FakeGenAI(), raising=True)


@pytest.fixture()
def mock_supabase_rpc(monkeypatch):
    """Mock Supabase client used in utils.retrieve_rag_documents()."""
    try:
        import src.shared.utils as utils
    except Exception:
        return

    class FakeResponse:
        def __init__(self, data):
            self.data = data

    class FakeClient:
        def rpc(self, name, payload):
            # Return a predictable small dataset
            return types.SimpleNamespace(execute=lambda: FakeResponse([
                {"id": "doc1", "content": "alpha", "metadata": {"embedding": [1, 0, 0]}}
            ]))

    # When utils creates a client, return our fake client and also cache it
    monkeypatch.setattr(utils, "create_client", lambda url, key: FakeClient(), raising=True)
    # Clear the cached client so tests get the fake
    if hasattr(utils, "_SUPABASE_CLIENT"):
        utils._SUPABASE_CLIENT = None


@pytest.fixture()
def mock_requests(monkeypatch):
    """Mock requests.get and requests.head to avoid network access."""
    import requests

    class FakeResponse:
        def __init__(self, status_code=200, headers=None, content=b"", text="OK"):
            self.status_code = status_code
            self.headers = headers or {"content-type": "text/plain"}
            self.content = content
            self.text = text
            self.ok = status_code == 200

        def iter_content(self, chunk_size=8192):
            yield self.content

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(requests, "head", lambda *a, **k: FakeResponse(headers={"content-type": "text/html"}))
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse(content=b"data"))


@pytest.fixture()
def mock_selenium(monkeypatch):
    """Mock Selenium webdriver.Chrome and basic WebElement interactions."""
    try:
        import selenium.webdriver as webdriver
    except Exception:
        return

    class FakeElement:
        def __init__(self, text=""):
            self.text = text

        def click(self):
            return None

        def get_attribute(self, name):
            return "attr-value"

        def find_element(self, *a, **k):
            return FakeElement()

        def find_elements(self, *a, **k):
            return [FakeElement(), FakeElement()]

    class FakeDriver:
        def get(self, url):
            self.current_url = url

        def execute_script(self, *a, **k):
            return None

        def find_element(self, *a, **k):
            return FakeElement()

        def find_elements(self, *a, **k):
            return [FakeElement(), FakeElement()]

        def quit(self):
            return None

    monkeypatch.setattr(webdriver, "Chrome", lambda *a, **k: FakeDriver())


@pytest.fixture()
def mock_fitz(monkeypatch):
    """Mock fitz.open() and basic page APIs for PDF processing."""
    try:
        import fitz  # type: ignore
    except Exception:
        return

    class FakePage:
        def get_text(self):
            return "Mock PDF page text"

        def get_images(self, full=True):
            return []

        def get_links(self):
            return []

    class FakeDoc(list):
        def __init__(self, n=1):
            super().__init__([FakePage() for _ in range(n)])

    monkeypatch.setattr(fitz, "open", lambda path: FakeDoc(2))
