import types
import pytest
import os
from typing import Any, Dict, List, Optional


class _FakeElement:
    def __init__(self, text: str = ""):
        self.text = text

    def click(self) -> None:
        return None

    def find_elements(self, *args, **kwargs):
        return []


class _FakeSwitchTo:
    def __init__(self, driver: "_FakeDriver"):
        self._driver = driver

    def window(self, handle: str) -> None:
        self._driver._switch_calls += 1
        # Update the current window handle if exists
        if handle in self._driver.window_handles:
            self._driver.current_window_handle = handle
        else:
            # Allow switching to unknown handles for robustness in tests
            self._driver.window_handles.append(handle)
            self._driver.current_window_handle = handle


class _FakeDriver:
    def __init__(self):
        # Navigation
        self._get_calls: int = 0
        self._execute_script_calls: List[str] = []
        self._execute_async_script_calls: List[str] = []
        self._switch_calls: int = 0
        self._close_calls: int = 0
        # Windows
        self.current_window_handle: str = "main"
        self.window_handles: List[str] = ["main"]
        # Elements registry keyed by CSS selector for simple tests
        self._elements_by_selector: Dict[str, List[_FakeElement]] = {}
        # Selenium-like API facets
        self.switch_to = _FakeSwitchTo(self)
        self.current_url: str = ""

    # --- Called by code under test ---
    def get(self, url: str) -> None:
        self._get_calls += 1
        self.current_url = url

    def execute_script(self, script: str, *args: Any) -> Optional[Any]:
        self._execute_script_calls.append(script)
        # Minimal window.open simulation
        if isinstance(script, str) and script.startswith("window.open"):
            # Append a deterministic new handle
            new_index = sum(1 for h in self.window_handles if h.startswith("window-")) + 1
            new_handle = f"window-{new_index}"
            self.window_handles.append(new_handle)
            return None
        return None

    def execute_async_script(self, script: str, *args: Any) -> Optional[Any]:
        self._execute_async_script_calls.append(script)
        handler = getattr(self, "_execute_async_script_handler", None)
        if callable(handler):
            return handler(script, *args)
        return None

    def find_elements(self, by: Any = None, selector: str = "") -> List[_FakeElement]:
        # Allow tests to pre-register elements for a given selector
        return list(self._elements_by_selector.get(selector, []))

    def find_element(self, by: Any = None, selector: str = "") -> _FakeElement:
        found = self.find_elements(by, selector)
        return found[0] if found else _FakeElement("")

    def close(self) -> None:
        self._close_calls += 1
        # Remove current handle if it's not the original main
        if self.current_window_handle != "main":
            try:
                self.window_handles.remove(self.current_window_handle)
            except ValueError:
                pass
            # Fallback to main
            self.current_window_handle = "main"


@pytest.fixture(scope="session", autouse=True)
def set_test_env():
    """Set required environment variables so modules initialize without real secrets."""
    os.environ["GEMINI_API_KEY"] = "test-gemini-key"
    os.environ["LLM_PROVIDER"] = "gemini"
    os.environ["SUPABASE_URL"] = "https://example.supabase.co"
    os.environ["SUPABASE_KEY"] = "test-supabase-key"
    os.environ["COACH_USERNAME"] = "test-user"
    os.environ["SECRET_API_KEY"] = "secret"
    # Provide defaults used by utils
    os.environ["EMBEDDING_MODEL_NAME"] = "models/embedding-001"


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


@pytest.fixture()
def mock_driver():
    """Configurable fake Selenium driver for tests.

    Provides:
    - get(), execute_script(), find_element(s)()
    - window_handles management, switch_to.window(), close()

    Tests may mutate:
    - driver._elements_by_selector[css] = [ _FakeElement("text"), ... ]
    to control returned elements for a given CSS selector.
    """
    return _FakeDriver()
