# llm_client.py
# Description: Provides a client for interacting with the Ollama LLM API.
# Handles request formatting, streaming responses, and error handling.

from __future__ import annotations
import json
import logging
from typing import Any, Dict, Generator
import requests
# Import the centralized configuration
from config import settings
from urllib.parse import urlparse

logger = logging.getLogger(__name__)
# This constant is now sourced from settings, assuming you complete Tier 2 recommendations.
# For now, we define it here to ensure the client is self-contained.
MAX_PROMPT_LENGTH = 8_000

def _validate_ollama_url(url: str):
    """Raises ValueError if the URL is not secure for non-local hosts."""
    parsed_url = urlparse(url)
    if parsed_url.hostname not in ("localhost", "127.0.0.1") and parsed_url.scheme != "https":
        raise ValueError(
            f"Insecure Ollama URL configured for non-local host: {url}. "
            "Please use HTTPS."
        )

# ---------------------------------------------------------------------------
# custom exceptions
# ---------------------------------------------------------------------------

class OllamaClientError(Exception):
    """Base exception for Ollama client errors."""

class OllamaConnectionError(OllamaClientError):
    """Raised for connection failures to the Ollama server."""

class OllamaResponseError(OllamaClientError):
    """Raised when Ollama returns an error response."""

class OllamaTimeoutError(OllamaClientError):
    """Raised when a request to Ollama times out."""

# ---------------------------------------------------------------------------
# client function
# ---------------------------------------------------------------------------

def stream_llm_response(prompt: str, model: str | None = None) -> Generator[str, None, None]:
    """
    Streams token chunks from the LLM server for a given prompt.

    Args:
        prompt: The full prompt string for the LLM.
        model: Optional override for the Ollama model to use.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string.")
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise ValueError(f"Prompt exceeds max length of {MAX_PROMPT_LENGTH} chars.")

    model_to_use = model or settings.ollama_model
    url = f"{settings.ollama_host.rstrip('/')}:{settings.ollama_port}/api/generate"
    _validate_ollama_url(url)
    payload = {"model": model_to_use, "prompt": prompt, "stream": True}

    logger.info(
        "Sending request to Ollama",
        extra={"url": url, "model": model_to_use},
    )

    try:
        with requests.post(
            url, json=payload, stream=True, timeout=settings.ollama_timeout
        ) as response:
            response.raise_for_status()
            for raw in response.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Skipping non-JSON line in stream", extra={"line": raw})
                    continue
                if "error" in data:
                    raise OllamaResponseError(data["error"])
                chunk = data.get("response", "")
                if chunk:
                    yield chunk
                if data.get("done", False):
                    break
    except requests.exceptions.ConnectionError as e:
        raise OllamaConnectionError(f"Connection to {url} failed.") from e
    except requests.exceptions.Timeout as e:
        raise OllamaTimeoutError("Request timed out.") from e
    except requests.exceptions.HTTPError as e:
        raise OllamaResponseError(
            f"Ollama returned HTTP {e.response.status_code}."
        ) from e
    except requests.exceptions.RequestException as e:
        raise OllamaClientError("An unexpected request error occurred.") from e
    
# ---------------------------------------------------------------------------
# synchronous helper used for short, non-streaming checks
# ---------------------------------------------------------------------------
def get_llm_completion(prompt: str, model: str | None = None) -> str:
    """
    Send a single, non-streaming request to the Ollama API and
    return the plain text response.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string.")
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise ValueError(f"Prompt exceeds max length of {MAX_PROMPT_LENGTH} characters.")

    model_to_use = model or settings.ollama_model
    url = f"{settings.ollama_host.rstrip('/')}" \
          f":{settings.ollama_port}/api/generate"
    _validate_ollama_url(url)
    payload = {
        "model": model_to_use,
        "prompt": prompt,
        "stream": False,
    }

    logger.info("Non-stream request to Ollama",
                extra={"url": url, "model": model_to_use})

    try:
        response = requests.post(url, json=payload,
                                 timeout=settings.ollama_timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.exceptions.RequestException as exc:
        raise OllamaConnectionError(str(exc)) from exc
