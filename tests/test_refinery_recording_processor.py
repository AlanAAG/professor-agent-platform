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


def test_scrape_drive_transcript_uses_timedtext(monkeypatch, mock_driver):
    # Arrange: timedtext payload is returned immediately
    mock_driver.queue_async_script_result({
        "status": "ok",
        "data": {
            "events": [
                {"segs": [{"utf8": "Hello"}, {"utf8": "world"}]},
                {"segs": [{"utf8": "from Drive"}]},
            ]
        },
    })
    monkeypatch.setattr(rp, "WebDriverWait", _FakeWait, raising=True)

    # Act
    result = rp.scrape_drive_transcript_content(mock_driver)

    # Assert
    assert result == "Hello world from Drive"
    # Only one async script call needed
    assert len(mock_driver._execute_async_script_calls) == 1


def test_extract_transcript_drive_success(monkeypatch, mock_driver):
    # Arrange
    monkeypatch.setattr(rp, "WebDriverWait", _FakeWait, raising=True)
    monkeypatch.setattr(rp, "scrape_drive_transcript_content", lambda d: "Drive transcript text", raising=True)

    # Act
    result = rp.extract_transcript(mock_driver, "https://drive.google.com/uc?id=abc", "DRIVE_RECORDING")

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
