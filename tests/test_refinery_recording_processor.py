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


def test_extract_transcript_drive_skip(monkeypatch, mock_driver):
    # Act
    result = rp.extract_transcript(mock_driver, "https://drive.google.com/uc?id=abc", "DRIVE_RECORDING")

    # Assert
    assert result == "Transcription skipped (Drive Recording)."
    # Ensure no navigation was attempted
    assert mock_driver._get_calls == 0
    assert not mock_driver._execute_script_calls


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
