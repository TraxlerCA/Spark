"""
This script sends prompts to an LLM model via Ollama's API and prints the result.
It follows best practices, includes clear structure and type hints, and adds ample comments for beginners.
"""

import json
import sys
from typing import Optional, List

import requests
from requests.exceptions import ConnectionError, HTTPError, Timeout


def ask_llm(
    prompt: str,
    host: str = "http://localhost",
    port: int = 11434,
    model: str = "gemma3:4b",
    timeout: int = 10
) -> str:
    """
    Send a prompt to an LLM model hosted on a local Ollama server.

    Args:
        prompt: the text question or instruction to send
        host: the API host (default localhost)
        port: the API port (default 11434)
        model: the name of the Ollama model to query
        timeout: seconds to wait before timing out

    Returns:
        the generated text response from the model,
        or an informative message if prompt is empty or an error occurs
    """
    # if user provides no prompt, give feedback immediately
    if not prompt.strip():
        return "No prompt provided. Please enter a valid prompt."

    # construct full URL for the generate endpoint
    url = f"{host}:{port}/api/generate"

    # prepare JSON payload with model name and prompt
    payload = {
        "model": model,
        "prompt": prompt
    }

    try:
        # use a session for better performance if calling multiple times
        with requests.Session() as session:
            response = session.post(
                url,
                json=payload,
                stream=True,
                timeout=timeout
            )
            # raise exception on HTTP error status codes (4xx, 5xx)
            response.raise_for_status()
    except ConnectionError:
        return "Error: cannot connect to Ollama. Is the server running?"
    except Timeout:
        return "Error: request to Ollama timed out. Is the model loaded and running?"
    except HTTPError:
        return (
            f"Error: Ollama returned status code {response.status_code}. "
            f"Details: {response.text}"
        )

    # collect pieces of the streamed response
    parts: List[str] = []
    for chunk in response.iter_lines(decode_unicode=True):
        # skip empty lines
        if not chunk:
            continue

        try:
            data = json.loads(chunk)
        except json.JSONDecodeError:
            # skip lines that are not valid JSON
            continue

        # if there's a 'response' field, add its text
        if 'response' in data and isinstance(data['response'], str):
            parts.append(data['response'])

    # join all parts into a final string
    return ''.join(parts)


def main() -> None:
    """
    Entry point for the script:
    1. prompt the user
    2. send to LLM
    3. display the reply
    """
    try:
        user_input = input("Ask the LLM a question: ")
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)

    # get model response (always a string)
    response_text = ask_llm(user_input)

    # print the model's output or error/default message
    print("\nModel says:\n", response_text)


if __name__ == "__main__":
    main()