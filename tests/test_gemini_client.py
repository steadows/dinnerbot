"""Tests for gemini_client — timeout, retry, and error handling."""

import time
from unittest.mock import MagicMock, patch

import pytest
from google.api_core.exceptions import DeadlineExceeded, InvalidArgument, ServiceUnavailable

from gemini_client import BASE_DELAY, DEFAULT_TIMEOUT, MAX_RETRIES, GeminiResult, call_gemini


@pytest.fixture()
def mock_model() -> MagicMock:
    """Create a mock Gemini model."""
    model = MagicMock()
    response = MagicMock()
    response.text = "Generated text content"
    model.generate_content.return_value = response
    return model


class TestGeminiResult:
    """Tests for the GeminiResult dataclass."""

    def test_success_result(self) -> None:
        result = GeminiResult(text="hello", success=True, attempts=1)
        assert result.text == "hello"
        assert result.success is True
        assert result.attempts == 1
        assert result.error is None

    def test_failure_result(self) -> None:
        result = GeminiResult(text=None, success=False, attempts=3, error="Timeout")
        assert result.text is None
        assert result.success is False
        assert result.error == "Timeout"

    def test_immutable(self) -> None:
        result = GeminiResult(text="hello", success=True, attempts=1)
        with pytest.raises(AttributeError):
            result.text = "modified"  # type: ignore[misc]


class TestCallGemini:
    """Tests for the call_gemini function."""

    def test_success_first_attempt(self, mock_model: MagicMock) -> None:
        result = call_gemini(mock_model, "test prompt")

        assert result.success is True
        assert result.text == "Generated text content"
        assert result.attempts == 1
        assert result.error is None
        mock_model.generate_content.assert_called_once_with(
            "test prompt",
            request_options={"timeout": DEFAULT_TIMEOUT},
        )

    def test_custom_timeout(self, mock_model: MagicMock) -> None:
        call_gemini(mock_model, "test", timeout=60)

        mock_model.generate_content.assert_called_once_with(
            "test",
            request_options={"timeout": 60},
        )

    def test_empty_response_text(self, mock_model: MagicMock) -> None:
        mock_model.generate_content.return_value.text = None
        result = call_gemini(mock_model, "test")

        assert result.success is True
        assert result.text is None
        assert result.attempts == 1

    @patch("gemini_client.time.sleep")
    def test_retry_on_deadline_exceeded(
        self, mock_sleep: MagicMock, mock_model: MagicMock
    ) -> None:
        success_response = MagicMock()
        success_response.text = "recovered"
        mock_model.generate_content.side_effect = [
            DeadlineExceeded("timeout"),
            success_response,
        ]

        result = call_gemini(mock_model, "test")

        assert result.success is True
        assert result.text == "recovered"
        assert result.attempts == 2
        mock_sleep.assert_called_once_with(BASE_DELAY)

    @patch("gemini_client.time.sleep")
    def test_retry_on_service_unavailable(
        self, mock_sleep: MagicMock, mock_model: MagicMock
    ) -> None:
        success_response = MagicMock()
        success_response.text = "recovered"
        mock_model.generate_content.side_effect = [
            ServiceUnavailable("503"),
            success_response,
        ]

        result = call_gemini(mock_model, "test")

        assert result.success is True
        assert result.text == "recovered"
        assert result.attempts == 2

    @patch("gemini_client.time.sleep")
    def test_exponential_backoff_delays(
        self, mock_sleep: MagicMock, mock_model: MagicMock
    ) -> None:
        success_response = MagicMock()
        success_response.text = "finally"
        mock_model.generate_content.side_effect = [
            DeadlineExceeded("1"),
            DeadlineExceeded("2"),
            success_response,
        ]

        result = call_gemini(mock_model, "test", max_retries=3)

        assert result.success is True
        assert result.attempts == 3
        # Backoff: 1s after attempt 1, 2s after attempt 2
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(BASE_DELAY)
        mock_sleep.assert_any_call(BASE_DELAY * 2)

    @patch("gemini_client.time.sleep")
    def test_all_retries_exhausted(
        self, mock_sleep: MagicMock, mock_model: MagicMock
    ) -> None:
        mock_model.generate_content.side_effect = DeadlineExceeded("always fails")

        result = call_gemini(mock_model, "test", max_retries=3)

        assert result.success is False
        assert result.text is None
        assert result.attempts == 3
        assert "DeadlineExceeded" in result.error
        assert mock_model.generate_content.call_count == 3

    def test_no_retry_on_non_retryable_error(self, mock_model: MagicMock) -> None:
        mock_model.generate_content.side_effect = InvalidArgument("bad prompt")

        result = call_gemini(mock_model, "test")

        assert result.success is False
        assert result.attempts == 1
        assert "InvalidArgument" in result.error
        mock_model.generate_content.assert_called_once()

    @patch("gemini_client.time.sleep")
    def test_no_sleep_after_last_retry(
        self, mock_sleep: MagicMock, mock_model: MagicMock
    ) -> None:
        mock_model.generate_content.side_effect = DeadlineExceeded("timeout")

        call_gemini(mock_model, "test", max_retries=2)

        # Only 1 sleep (after attempt 1), not after attempt 2
        mock_sleep.assert_called_once_with(BASE_DELAY)

    def test_custom_max_retries(self, mock_model: MagicMock) -> None:
        result = call_gemini(mock_model, "test", max_retries=1)

        assert result.success is True
        assert result.attempts == 1
        mock_model.generate_content.assert_called_once()
