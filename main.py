# main.py
#
# description: single-turn client for a local Ollama LLM.
#

# --------------------------------------------------------------------------- #
# imports
# --------------------------------------------------------------------------- #
from __future__ import annotations  # enable postponed evaluation of annotations (PEP 563)
import json  # for serializing and parsing JSON payloads and responses
import logging  # for emitting structured logs
import logging.config  # for configuring logging settings
import sys  # for interacting with the interpreter (e.g., exiting)
from typing import Any, Generator  # for type hinting functions and generators

import requests  # for HTTP requests to the Ollama API
from pydantic_settings import BaseSettings, SettingsConfigDict  # for environment-based configuration

# --------------------------------------------------------------------------- #
# constants & configuration
# --------------------------------------------------------------------------- #
# structured JSON logging configuration for the Ollama client
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",  # use JSON formatter
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",  # include timestamp, level, logger, message
        },
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",  # output logs to stdout
            "stream": "ext://sys.stdout",
            "formatter": "json",  # use the JSON formatter defined above
        },
    },
    "loggers": {
        "__main__": {"handlers": ["stdout"], "level": "INFO", "propagate": True},  # main module logger
        "ollama_client": {"handlers": ["stdout"], "level": "INFO", "propagate": True},  # client-specific logger
    },
}
logging.config.dictConfig(LOGGING_CONFIG)  # apply structured logging settings
logger = logging.getLogger(__name__)  # module-level logger

MAX_PROMPT_LENGTH = 8_000  # maximum allowed prompt length to avoid overly long requests

class OllamaSettings(BaseSettings):
    """Manages Ollama client configuration via environment variables."""
    model_config = SettingsConfigDict(
        env_prefix="OLLAMA_",  # prefix for environment variables
        env_file=".env",  # optional .env file for local development
        env_file_encoding="utf-8",  # encoding for the .env file
        extra="ignore",  # ignore unknown environment variables
    )

    host: str = "http://localhost"  # default Ollama server host
    port: int = 11434  # default Ollama server port
    model: str = "gemma3:4b"  # default model identifier
    timeout: int = 60  # default request timeout in seconds

# --------------------------------------------------------------------------- #
# custom exceptions
# --------------------------------------------------------------------------- #
class OllamaClientError(Exception):
    """Base exception for all Ollama client errors."""

class OllamaConnectionError(OllamaClientError):
    """Raised when the client cannot connect to the Ollama server."""

class OllamaResponseError(OllamaClientError):
    """Raised for non-2xx HTTP responses from the Ollama server or error in streamed payload."""

class OllamaTimeoutError(OllamaClientError):
    """Raised when a request to the Ollama server times out."""

# --------------------------------------------------------------------------- #
# core client logic
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
    # use default settings if none are provided
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
    # - ensure the prompt is a non-empty string
    # - enforce maximum length constraint
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string.")
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise ValueError(f"Prompt exceeds maximum length of {MAX_PROMPT_LENGTH} chars.")

    # build request payload and URL
    # - construct API endpoint URL
    # - include model, prompt, and streaming flag in payload
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
            response.raise_for_status()  # raise for HTTP error statuses

            for raw in response.iter_lines(decode_unicode=True):
                if not raw:
                    continue  # skip empty lines

                try:
                    data = json.loads(raw)  # parse JSON line
                except json.JSONDecodeError:
                    logger.warning("Skipping non-JSON line in stream", extra={"line": raw})
                    continue

                if "error" in data:
                    # if the API returned an error in the stream
                    raise OllamaResponseError(data["error"])

                chunk = data.get("response", "")
                if chunk:
                    yield chunk  # yield non-empty response chunks

                if data.get("done", False):
                    break  # stop when the stream signals completion

    except requests.exceptions.ConnectionError as e:
        # network-level connection failure
        raise OllamaConnectionError(f"Connection to {url} failed.") from e
    except requests.exceptions.Timeout as e:
        # request timed out
        raise OllamaTimeoutError("Request timed out. Is the model loaded?") from e
    except requests.exceptions.HTTPError as e:
        # HTTP protocol error (non-2xx)
        status_code = e.response.status_code
        raise OllamaResponseError(f"Ollama returned HTTP {status_code}.") from e
    except requests.exceptions.RequestException as e:
        # catch-all for other request-related errors
        raise OllamaClientError("An unexpected request error occurred.") from e

# --------------------------------------------------------------------------- #
# command-line interface
# --------------------------------------------------------------------------- #
def main() -> None:
    """Prompts the user once, sends the prompt to the LLM, and prints the reply."""
    try:
        # - initialize configuration settings
        settings = OllamaSettings()

        # - prompt user for input and strip whitespace
        user_prompt = input("Ask the LLM a question: ").strip()

        if not user_prompt:
            logger.error("No prompt given. Aborting.")
            return

        print("\nModel says:\n")
        # - stream response chunks to console for real-time display
        for chunk in stream_llm_response(user_prompt, settings):
            print(chunk, end="", flush=True)
        print("\n")

    except (KeyboardInterrupt, EOFError):
        # handle user cancellation gracefully
        print("\nBye!")
    except (ValueError, OllamaClientError) as e:
        # log client-side errors and exit with error code
        logger.error("A client-side error occurred.", extra={"error": str(e)})
        sys.exit(1)
    except Exception as e:
        # catch-all for unexpected errors
        logger.critical(
            "An unexpected error occurred.",
            extra={"error": str(e)},
            exc_info=True,
        )
        sys.exit(1)

if __name__ == "__main__":
    main()
