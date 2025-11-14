import pytest

from src.refinery import recording_processor as rp


class _FakeWait:
    def __init__(self, driver, timeout):
        self.driver = driver
        self.timeout = timeout

    def until(self, condition):
        # For lambda-based waits, just evaluate once
        try:
            if callable(condition):
                return condition(self.driver)
        except Exception:
            pass
        return True


def test_extract_transcript_zoom_success(monkeypatch, mock_driver):
    # Arrange
    monkeypatch.setattr(rp, "WebDriverWait", _FakeWait, raising=True)
    monkeypatch.setattr(rp, "scrape_zoom_transcript_content", lambda d: "Zoom transcript text", raising=True)

    # Act
    result = rp.extract_transcript(mock_driver, "https://zoom.us/rec/abc", "ZOOM_RECORDING")

    # Assert
    assert result == "Zoom transcript text"
    # Ensure a new window was opened via execute_script
    assert any(s.startswith("window.open") for s in mock_driver._execute_script_calls)


def test_extract_transcript_drive_success(monkeypatch, mock_driver):
    # Arrange
    monkeypatch.setattr(rp, "WebDriverWait", _FakeWait, raising=True)
    monkeypatch.setattr(rp, "scrape_drive_transcript_content", lambda d: "Drive transcript text", raising=True)

    # Act
    result = rp.extract_transcript(mock_driver, "https://drive.google.com/file/d/abc/view", "DRIVE_RECORDING")

    # Assert
    assert result == "Drive transcript text"
    assert any(s.startswith("window.open") for s in mock_driver._execute_script_calls)


def test_extract_transcript_window_cleanup(monkeypatch, mock_driver):
    # Arrange: success path should still cleanup windows
    monkeypatch.setattr(rp, "WebDriverWait", _FakeWait, raising=True)
    monkeypatch.setattr(rp, "scrape_zoom_transcript_content", lambda d: "ok", raising=True)

    # Act
    _ = rp.extract_transcript(mock_driver, "https://zoom.us/rec/abc", "ZOOM_RECORDING")

    # Assert: new window closed and switched back
    assert mock_driver._close_calls >= 1
    assert mock_driver.current_window_handle == "main"
    # Switched at least twice: to new window and back to original
    assert mock_driver._switch_calls >= 2


def test_parse_drive_timedtext():
    payload = {
        "events": [
            {"segs": [{"utf8": "Hello"}, {"utf8": "world"}]},
            {"segs": [{"utf8": "Second"}, {"utf8": "line"}]},
        ]
    }
    result = rp._parse_drive_timedtext(payload)
    assert result == "Hello world\nSecond line"


def test_scrape_drive_transcript_content_success(monkeypatch, mock_driver):
    monkeypatch.setattr(
        rp,
        "_locate_drive_caption_url",
        lambda driver, wait_timeout: "https://drive.google.com/timedtext?fmt=json3",
        raising=True,
    )
    monkeypatch.setattr(
        rp,
        "_fetch_drive_timedtext_payload",
        lambda driver, url: {"events": [{"segs": [{"utf8": "Hi"}, {"utf8": "there"}]}]},
        raising=True,
    )

    result = rp.scrape_drive_transcript_content(mock_driver)
    assert result == "Hi there"


def test_scrape_drive_transcript_content_missing_url(monkeypatch, mock_driver):
    monkeypatch.setattr(
        rp,
        "_locate_drive_caption_url",
        lambda driver, wait_timeout: None,
        raising=True,
    )

    result = rp.scrape_drive_transcript_content(mock_driver)
    assert result == ""
