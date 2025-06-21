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
import logging
import logging.config
import os
import sys
import typer

import typer
from config import settings
from llm_client import OllamaClientError   # We only need the exception here
from core import process_query             # The most important new import

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
# helper functions
# --------------------------------------------------------------------------- #

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
    Prompts the user for a question, processes it via the core engine,
    and streams the model's response to the console.
    """
    try:
        # We are not managing history in this simple CLI client
        history = []
        user_prompt = input("Ask the LLM a question: ").strip()
        if not user_prompt:
            logger.error("No prompt given; aborting.")
            return

        use_rag_requested = os.getenv("USE_RAG", "1") == "1"

        # Call the core processing function
        response_generator, sources = process_query(
            user_prompt=user_prompt,
            history=history,
            use_rag=use_rag_requested,
        )

        print("\nModel says:\n")
        for chunk in response_generator:
            print(chunk, end="", flush=True)

        # Footer with sources
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