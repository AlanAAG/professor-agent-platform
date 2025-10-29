import pytest

from selenium.common.exceptions import TimeoutException

from src.refinery.document_processing import extract_drive_content


class _FakeWait:
    def __init__(self, driver, timeout):
        self.driver = driver
        self.timeout = timeout

    def until(self, condition):
        # Immediately succeed
        return True


class _TimeoutWait(_FakeWait):
    def until(self, condition):
        raise TimeoutException("timed out")


def test_extract_drive_content_success(monkeypatch, mock_driver):
    # Arrange: Simulate Google Docs structure
    from src.refinery import document_processing as dp

    # Monkeypatch WebDriverWait to immediate success
    monkeypatch.setattr(dp, "WebDriverWait", _FakeWait, raising=True)

    # Provide pseudo elements returned by driver.find_elements for docs pages
    mock_driver._elements_by_selector[".kix-page"] = [
        # texts should be trimmed and joined with blank lines
        type("El", (), {"text": "First page text"})(),
        type("El", (), {"text": "Second page text"})(),
    ]

    # Act
    content = extract_drive_content(mock_driver, "https://docs.google.com/document/d/123", "google_docs")

    # Assert
    assert content == "First page text\n\nSecond page text"


def test_extract_drive_content_timeout(monkeypatch, mock_driver):
    # Arrange: Make WebDriverWait.until raise TimeoutException
    from src.refinery import document_processing as dp

    monkeypatch.setattr(dp, "WebDriverWait", _TimeoutWait, raising=True)

    # Act
    content = extract_drive_content(mock_driver, "https://docs.google.com/document/d/123", "google_docs")

    # Assert: Graceful failure returns None
    assert content is None
