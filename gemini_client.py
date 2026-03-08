"""
Resilient Gemini API client with timeout, exponential backoff, and retry logic.

Wraps google.generativeai's generate_content with:
- Explicit per-request timeout (default 30s)
- Exponential backoff retry (default 3 attempts: 1s, 2s, 4s delays)
- Retryable error detection (DeadlineExceeded, ServiceUnavailable, TimeoutError)
- Structured logging for each attempt
"""

import time
from dataclasses import dataclass
from typing import Any

from google.api_core.exceptions import DeadlineExceeded, ServiceUnavailable


RETRYABLE_EXCEPTIONS = (DeadlineExceeded, ServiceUnavailable, TimeoutError)

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
BASE_DELAY = 1.0


@dataclass(frozen=True)
class GeminiResult:
    """Immutable result from a Gemini API call."""

    text: str | None
    success: bool
    attempts: int
    error: str | None = None


def call_gemini(
    model: Any,
    prompt: str,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY,
) -> GeminiResult:
    """Call Gemini API with timeout and exponential backoff retry.

    Args:
        model: A google.generativeai.GenerativeModel instance.
        prompt: The prompt string to send.
        timeout: Per-request timeout in seconds.
        max_retries: Maximum number of attempts before giving up.
        base_delay: Base delay in seconds for exponential backoff.

    Returns:
        GeminiResult with text on success, or error details on failure.
    """
    last_error = None

    for attempt in range(1, max_retries + 1):
        start_time = time.monotonic()
        try:
            print(f"Gemini attempt {attempt}/{max_retries}: calling API (timeout={timeout}s)")

            response = model.generate_content(
                prompt,
                request_options={"timeout": timeout},
            )

            elapsed = time.monotonic() - start_time
            print(f"Gemini attempt {attempt}: success in {elapsed:.1f}s")

            text = response.text if response.text else None
            return GeminiResult(text=text, success=True, attempts=attempt)

        except RETRYABLE_EXCEPTIONS as e:
            elapsed = time.monotonic() - start_time
            last_error = f"{type(e).__name__}: {e}"
            print(
                f"Gemini attempt {attempt}/{max_retries}: "
                f"retryable error after {elapsed:.1f}s — {last_error}"
            )

            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                print(f"Gemini: retrying in {delay:.1f}s")
                time.sleep(delay)

        except Exception as e:
            elapsed = time.monotonic() - start_time
            error_msg = f"{type(e).__name__}: {e}"
            print(f"Gemini attempt {attempt}: non-retryable error after {elapsed:.1f}s — {error_msg}")
            return GeminiResult(
                text=None, success=False, attempts=attempt, error=error_msg
            )

    # All retries exhausted
    print(f"Gemini: all {max_retries} attempts exhausted")
    return GeminiResult(
        text=None, success=False, attempts=max_retries, error=last_error
    )
