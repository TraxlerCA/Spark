
"""
This script sends prompts to a specified LLM model via Ollama's API and prints the result.
It includes clear structure, type hints, and detailed comments for beginners.
"""

import json
import sys
from typing import Optional

import requests
from requests.exceptions import ConnectionError, HTTPError, Timeout


def ask_llm(
    prompt: str,
    host: str = "http://localhost",
    port: int = 11434,
    model: str = "gemma3:4b",
    timeout: int = 10
) -> Optional[str]:
    """
    Send a prompt to an LLM model hosted on a local Ollama server.

    Args:
        prompt: The text question or instruction to send.
        host: The API host (default is localhost).
        port: The API port (default is 11434).
        model: The name of the Ollama model to query (e.g., "gemma3:4b").
        timeout: Seconds to wait before timing out.

    Returns:
        The generated text response from the model, or None if an error occurs.
    """
    # construct the full API endpoint URL
    url = f"{host}:{port}/api/generate"

    # prepare the JSON payload
    payload = {
        "model": model,
        "prompt": prompt
    }

    try:
        # use a session for better performance on multiple requests
        with requests.Session() as session:
            response = session.post(url, json=payload, stream=True, timeout=timeout)
            # raise exception for HTTP errors
            response.raise_for_status()
    except ConnectionError:
        print(
            "Error: cannot connect to Ollama."
            " Is the server running?"
            " Try `ollama serve` in another terminal."
        )
        return None
    except Timeout:
        print(
            "Error: request to Ollama timed out."
            " Is the model loaded and running?"
        )
        return None
    except HTTPError:
        print(f"Error: Ollama returned status code {response.status_code}.")
        print("Details:", response.text)
        return None

    # collect pieces of the streamed response
    parts = []
    for chunk in response.iter_lines(decode_unicode=True):
        if not chunk:
            continue

        try:
            data = json.loads(chunk)
        except json.JSONDecodeError:
            # skip lines that aren't valid JSON
            continue

        # append any 'response' fields
        if 'response' in data:
            parts.append(data['response'])

    # join all parts into the final output
    return ''.join(parts)


def main() -> None:
    """
    Main entry point: prompt the user, call the API, and display the model's reply.
    """
    # prompt the user for input
    try:
        user_input = input("Ask the LLM a question: ")
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)

    # get the response from the model
    response_text = ask_llm(user_input)

    # print the result or an error message
    if response_text is not None:
        print("\nModel says:\n", response_text)
    else:
        print("No response received.")


if __name__ == "__main__":
    main()