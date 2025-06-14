#
# main.py
#
# Description: Single-turn client for a local Ollama LLM.
#
"""
This script provides a client to interact with a local Ollama LLM instance.

It handles sending a prompt to the Ollama API and streams the response back.
Configuration is managed via environment variables and Pydantic settings.
Error handling is managed through custom exceptions and structured logging.
"""

from __future__ import annotations

import json
import logging
import logging.config
import sys
from typing import Any, Generator

import requests
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------- #
# Constants & Configuration
# --------------------------------------------------------------------------- #

# Set up structured JSON logging
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
        },
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "json",
        },
    },
    "loggers": {
        "__main__": {"handlers": ["stdout"], "level": "INFO", "propagate": True},
        "ollama_client": {"handlers": ["stdout"], "level": "INFO", "propagate": True},
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Maximum prompt length to avoid overly long requests in this example.
MAX_PROMPT_LENGTH = 8_000


class OllamaSettings(BaseSettings):
    """Manages Ollama client configuration using Pydantic."""

    model_config = SettingsConfigDict(
        env_prefix="OLLAMA_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    host: str = "http://localhost"
    port: int = 11434
    model: str = "gemma3:4b"
    timeout: int = 60


# --------------------------------------------------------------------------- #
# Custom Exceptions
# --------------------------------------------------------------------------- #


class OllamaClientError(Exception):
    """Base exception for all Ollama client errors."""


class OllamaConnectionError(OllamaClientError):
    """Raised when the client cannot connect to the Ollama server."""


class OllamaResponseError(OllamaClientError):
    """Raised for non-2xx HTTP responses from the Ollama server."""


class OllamaTimeoutError(OllamaClientError):
    """Raised when a request to the Ollama server times out."""


# --------------------------------------------------------------------------- #
# Core Client Logic
# --------------------------------------------------------------------------- #


def ask_llm(prompt: str, settings: OllamaSettings | None = None) -> str:
    """
    Sends a prompt to an Ollama LLM and returns the complete response.

    This function consolidates the streamed response into a single string.

    Args:
        prompt: The question or instruction for the model.
        settings: Configuration for the Ollama connection. Defaults to None,
                  which initializes new settings.

    Returns:
        A string containing the model's complete answer.

    Raises:
        ValueError: If the prompt is empty or exceeds MAX_PROMPT_LENGTH.
    """
    # Use default settings if none are provided
    if settings is None:
        settings = OllamaSettings()

    # Consolidate the generator into a single string response
    response_parts = list(stream_llm_response(prompt, settings))
    return "".join(response_parts)


def stream_llm_response(
    prompt: str, settings: OllamaSettings | None = None
) -> Generator[str, None, None]:
    """
    Sends a prompt to an Ollama LLM and streams the response.

    This function yields response chunks as they are received from the API.

    Args:
        prompt: The question or instruction for the model.
        settings: Configuration for the Ollama connection. Defaults to None,
                  which initializes new settings.

    Yields:
        A generator of strings, where each string is a piece of the response.

    Raises:
        ValueError: If the prompt is empty or exceeds MAX_PROMPT_LENGTH.
        OllamaConnectionError: If a connection to the server fails.
        OllamaTimeoutError: If the request times out.
        OllamaResponseError: If the server returns an HTTP error status or
                             if the API signals an error in the stream.
        OllamaClientError: For other client-side errors during the request.
    """
    # use default settings if none are provided
    if settings is None:
        settings = OllamaSettings()

    # input validation
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string.")
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise ValueError(f"Prompt exceeds maximum length of {MAX_PROMPT_LENGTH} chars.")

    # build request
    url = f"{settings.host.rstrip('/')}:{settings.port}/api/generate"
    payload = {"model": settings.model, "prompt": prompt, "stream": True}
    log_context = {"url": url, "model": settings.model}

    logger.info("Sending request to Ollama", extra=log_context)

    # execute request and stream response
    try:
        with requests.post(
            url,
            json=payload,
            stream=True,
            timeout=settings.timeout,
        ) as response:
            response.raise_for_status()
            for raw in response.iter_lines(decode_unicode=True):
                # skip empty lines
                if not raw:
                    continue

                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Skipping non-JSON line in stream", extra={"line": raw})
                    continue

                # if the API returned an error in the stream payload
                if "error" in data:
                    raise OllamaResponseError(data["error"])

                # yield only non-empty response chunks
                chunk = data.get("response", "")
                if chunk:
                    yield chunk

                # stop streaming when done is True
                if data.get("done", False):
                    break

    except requests.exceptions.ConnectionError as e:
        raise OllamaConnectionError(f"Connection to {url} failed.") from e
    except requests.exceptions.Timeout as e:
        raise OllamaTimeoutError("Request timed out. Is the model loaded?") from e
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        raise OllamaResponseError(f"Ollama returned HTTP {status_code}.") from e
    except requests.exceptions.RequestException as e:
        raise OllamaClientError("An unexpected request error occurred.") from e


# --------------------------------------------------------------------------- #
# Entry Point
# --------------------------------------------------------------------------- #


def main() -> None:
    """Prompts the user once, sends the prompt to the LLM, and prints the reply."""
    try:
        settings = OllamaSettings()
        user_prompt = input("Ask the LLM a question: ").strip()

        if not user_prompt:
            logger.error("No prompt given. Aborting.")
            return

        print("\nModel says:\n")
        # Stream the response directly to the console
        for chunk in stream_llm_response(user_prompt, settings):
            print(chunk, end="", flush=True)
        print("\n")

    except (KeyboardInterrupt, EOFError):
        print("\nBye!")
    except (ValueError, OllamaClientError) as e:
        logger.error("A client-side error occurred.", extra={"error": str(e)})
        sys.exit(1)
    except Exception as e:
        logger.critical(
            "An unexpected error occurred.",
            extra={"error": str(e)},
            exc_info=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()