# main.py
#
# Description: A command-line interface (CLI) client for interacting with the
#              Ollama LLM. It can optionally retrieve context from the RAG
#              pipeline before generating a response. It handles input, errors,
#              and uses structured JSON logging.
#

# --------------------------------------------------------------------------- #
# imports
# --------------------------------------------------------------------------- #
from __future__ import annotations
import json
import logging
import logging.config
import os
import sys
from typing import Any, Dict, Generator, Set

import requests
import typer

# Import the centralized configuration
from config import settings
# Import the unified prompt formatter
from history_utils import format_prompt

# --------------------------------------------------------------------------- #
# optional RAG integration
# --------------------------------------------------------------------------- #
try:
    # Attempt to import retriever functions
    from rag.retriever import retrieve, format_context
    RAG_ENABLED = True
except ImportError:
    # If imports fail, disable RAG functionality
    RAG_ENABLED = False

    def retrieve(*args, **kwargs):  # type: ignore
        return []

    def format_context(*args, **kwargs):  # type: ignore
        return ""

# --------------------------------------------------------------------------- #
# constants and configuration
# --------------------------------------------------------------------------- #
MAX_PROMPT_LENGTH = 8_000
SHOW_CONTEXT = os.getenv("SHOW_CONTEXT", "0") == "1"
ABS_MIN_SCORE   = 0.35   # ignore anything below this outright
REL_WINDOW      = 0.05   # keep chunks that are within this of the top score

# Setup structured JSON logging for the application
LOGGING_CONFIG: Dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
        }
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "json",
        }
    },
    "loggers": {
        "__main__": {"handlers": ["stdout"], "level": "INFO", "propagate": True},
        "ollama_client": {"handlers": ["stdout"], "level": "INFO", "propagate": True},
        "rag": {"handlers": ["stdout"], "level": "INFO", "propagate": True},
    },
}
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# custom exceptions
# --------------------------------------------------------------------------- #
class OllamaClientError(Exception):
    """Base exception for Ollama client errors."""


class OllamaConnectionError(OllamaClientError):
    """Raised for connection failures to the Ollama server."""


class OllamaResponseError(OllamaClientError):
    """Raised when Ollama returns an error response."""


class OllamaTimeoutError(OllamaClientError):
    """Raised when a request to Ollama times out."""


# --------------------------------------------------------------------------- #
# helper functions
# --------------------------------------------------------------------------- #
def stream_llm_response(prompt: str) -> Generator[str, None, None]:
    """
    Streams token chunks from the LLM server for a given prompt.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string.")
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise ValueError(f"Prompt exceeds max length of {MAX_PROMPT_LENGTH} chars.")

    url = f"{settings.ollama_host.rstrip('/')}:{settings.ollama_port}/api/generate"
    payload = {"model": settings.ollama_model, "prompt": prompt, "stream": True}
    logger.info(
        "Sending request to Ollama",
        extra={"url": url, "model": settings.ollama_model},
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


def extract_sources(nodes) -> Set[str]:
    """
    Collect unique file names from retrieved nodes for the footer.
    """
    return {n.metadata.get("file_name", "unknown") for n in nodes}


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    """
    Prompts the user for a question, optionally retrieves RAG context,
    formats a complete prompt, and streams the model's response to the console.
    """
    try:
        user_prompt = input("Ask the LLM a question: ").strip()
        if not user_prompt:
            logger.error("No prompt given; aborting.")
            return

        rag_context = ""
        sources: Set[str] = set()
        use_rag_requested = os.getenv("USE_RAG", "1") == "1"

        if use_rag_requested and RAG_ENABLED:
            logger.info("RAG is enabled; retrieving context...")
            try:
                nodes = retrieve(user_prompt, k=settings.similarity_top_k)

                 # DEBUG: show scores returned by the retriever
                for n in nodes:
                    print(f"{n.score:.4f}  {n.metadata.get('file_name', 'unknown')}")

                good_nodes = []
                if nodes:
                    top_score = nodes[0].score
                    if top_score >= ABS_MIN_SCORE:
                        good_nodes.append(nodes[0])

                        for n in nodes[1:]:
                            if n.score >= max(ABS_MIN_SCORE, top_score - REL_WINDOW):
                                good_nodes.append(n)

                if good_nodes:
                    rag_context = format_context(good_nodes)
                    sources = extract_sources(good_nodes)
                    logger.info("Context retrieved successfully.")
                    if SHOW_CONTEXT and rag_context:
                        print("\n--- Retrieved Context ---\n")
                        print(rag_context)
                        print("\n-------------------------\n")
                else:
                    logger.info("No nodes passed the similarity threshold; skipping RAG.")
            except Exception as e:
                logger.error("Failed to retrieve context", extra={"error": str(e)})
                print("\n[Warning] Could not retrieve context. Proceeding without it.\n")
        elif use_rag_requested:
            logger.warning(
                "USE_RAG is set, but RAG components are not available."
            )

        # Use the unified prompt formatting function for consistency
        final_prompt = format_prompt(
            history=[],
            next_user_message=user_prompt,
            rag_context=rag_context,
        )

        print("\nModel says:\n")
        for chunk in stream_llm_response(final_prompt):
            print(chunk, end="", flush=True)

        # footer with sources
        if sources:
            print(f"\n\nsources: {', '.join(sorted(sources))}")
        else:
            print("\n\nsources: none (model knowledge)")
        print()

    except (KeyboardInterrupt, EOFError):
        print("\nBye!")
    except (ValueError, OllamaClientError) as e:
        logger.error("A client-side error occurred", extra={"error": str(e)})
        sys.exit(1)
    except Exception as e:
        logger.critical(
            "An unexpected error occurred", extra={"error": str(e)}, exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    import typer
    typer.run(main)