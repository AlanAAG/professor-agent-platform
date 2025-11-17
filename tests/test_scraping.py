import requests

from src.harvester import scraping


class DummyResponse:
    def __init__(self, status_code: int, headers: dict | None = None):
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.exceptions.HTTPError(f"{self.status_code} error")
            error.response = self
            raise error


def test_check_url_content_type_forbidden_defaults_to_html(monkeypatch):
    def fake_head(*_, **__):
        return DummyResponse(403, {})

    monkeypatch.setattr(scraping.requests, "head", fake_head)

    result = scraping.check_url_content_type("https://openai.com/research/whisper")
    assert "text/html" in result


def test_check_url_content_type_forbidden_pdf_inferred(monkeypatch):
    def fake_head(*_, **__):
        return DummyResponse(403, {})

    monkeypatch.setattr(scraping.requests, "head", fake_head)

    result = scraping.check_url_content_type("https://example.com/files/report.PDF?token=abc")
    assert "application/pdf" in result


def test_check_url_content_type_connection_error(monkeypatch):
    def fake_head(*_, **__):
        raise requests.exceptions.ConnectionError("network down")

    monkeypatch.setattr(scraping.requests, "head", fake_head)

    result = scraping.check_url_content_type("https://example.com")
    assert result == "error"
