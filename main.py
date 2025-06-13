"""
main.py – single-turn client for a local Ollama LLM.

This script asks the user for a prompt, sends it to an Ollama instance
running on localhost, and prints the model's response.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import List

import requests
from requests.exceptions import ConnectionError, HTTPError, Timeout

# --------------------------------------------------------------------------- #
# Configuration constants
# --------------------------------------------------------------------------- #

DEFAULT_HOST = "http://localhost"   # Ollama's REST server
DEFAULT_PORT = 11434               # Default Ollama port
DEFAULT_MODEL = "gemma3:4b"        # Model pulled earlier with `ollama pull`
DEFAULT_TIMEOUT = 10               # Seconds to wait before we give up

# Configure the root logger once, near the top of the file.
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def ask_llm(
    prompt: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    model: str = DEFAULT_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """
    Send a prompt to an LLM hosted by Ollama and return its response text.

    The function never raises; all errors are caught and returned
    as explanatory strings so the caller can decide what to do.

    Args:
        prompt: The question or instruction for the model.
        host:   Base URL of the Ollama server.
        port:   Listening port of the server.
        model:  Model name (must have been pulled already).
        timeout:Network timeout in seconds.

    Returns:
        A string with either the model's answer or an error message.
    """
    # ---------- validate input ----------
    if not isinstance(prompt, str) or not prompt.strip():
        return "Error: you must supply a non-empty prompt."

    # Limit length to something reasonable for a demo.
    if len(prompt) > 8_000:
        return "Error: prompt is too long (8 000 characters max)."

    # ---------- build the request ----------
    url = f"{host}:{port}/api/generate"
    payload = {"model": model, "prompt": prompt}

    # ---------- talk to Ollama ----------
    try:
        with requests.Session() as session:
            logger.info("Sending request to Ollama…")
            response = session.post(
                url,
                json=payload,
                stream=True,         # allows us to read the reply chunk by chunk
                timeout=timeout,
            )
            response.raise_for_status()  # converts 4xx/5xx to HTTPError
    except ConnectionError:
        return "Error: cannot connect to Ollama. Is the server running?"
    except Timeout:
        return "Error: request to Ollama timed out. Is the model loaded?"
    except HTTPError as exc:
        return f"Error: Ollama returned {exc.response.status_code}."

    # ---------- stream and assemble the reply ----------
    parts: List[str] = []
    for line in response.iter_lines(decode_unicode=True):
        if not line:                         # skip keep-alive lines
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Skipping a non-JSON line in the stream.")
            continue

        text_piece = data.get("response")
        if isinstance(text_piece, str):
            parts.append(text_piece)

    return "".join(parts)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    """Prompt the user once and display the model's reply."""
    try:
        user_prompt = input("Ask the LLM a question: ").strip()
    except KeyboardInterrupt:
        print("\nBye!")
        return

    if not user_prompt:
        logger.error("No prompt given. Abort.")
        return

    reply = ask_llm(user_prompt)
    print("\nModel says:\n")
    print(reply)


if __name__ == "__main__":
    main()
