# retriever.py
#
# Description: Provides retrieval utilities for the RAG pipeline. This module
#              is responsible for initializing a connection to the vector
#              store, executing similarity searches based on a user query,
#              and formatting the retrieved context for the LLM.
#

# --------------------------------------------------------------------------- #
# imports
# --------------------------------------------------------------------------- #
from __future__ import annotations
import json                     # for serializing logs
import logging                  # for structured logging
import sys                      # for stdout stream handler
from functools import lru_cache # for caching the retriever instance
from typing import Any, List    # for type hints
import re, textwrap

import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# Import the centralized configuration
from config import settings

# --------------------------------------------------------------------------- #
# custom exceptions
# --------------------------------------------------------------------------- #
class RetrieverError(Exception):
    """Custom exception for retriever-related failures."""

# --------------------------------------------------------------------------- #
# logger setup
# --------------------------------------------------------------------------- #
class JSONFormatter(logging.Formatter):
    """Logging formatter that outputs JSON."""
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
# retriever functions
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def get_retriever() -> Any:
    """
    Initializes and caches a retriever for the configured vector store.
    The LRU cache ensures this expensive initialization only happens once.

    Returns:
        A configured LlamaIndex retriever instance.
    """
    try:
        logger.info(
            f"Initializing retriever for collection "
            f"'{settings.collection}' at '{settings.persist_dir}'"
        )
        if not settings.persist_dir.is_dir():
            raise RetrieverError(f"Persist directory '{settings.persist_dir}' does not exist.")

        embed_model = HuggingFaceEmbedding(model_name=settings.embed_model_id)
        Settings.embed_model = embed_model

        client = chromadb.PersistentClient(path=str(settings.persist_dir))
        collection = client.get_collection(settings.collection)

        vector_store = ChromaVectorStore(chroma_collection=collection)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        retriever = index.as_retriever(similarity_top_k=settings.similarity_top_k)
        logger.info("Retriever initialized successfully")
        return retriever
    except ValueError as exc:
        logger.error(f"Collection '{settings.collection}' not found.", exc_info=exc)
        raise RetrieverError("Vector store collection not found") from exc
    except Exception as exc:
        logger.error("Failed to initialize retriever", exc_info=exc)
        raise RetrieverError("Retriever initialization failed") from exc

def retrieve(query: str, k: int | None = None) -> List[NodeWithScore]:
    """
    Retrieves the top k relevant text chunks for a given query.

    Args:
        query: The search query string.
        k: Optional override for the number of results to return.

    Returns:
        A list of NodeWithScore instances, each containing a text chunk and score.
    """
    top_k = k or settings.similarity_top_k
    if not query.strip():
        return []

    try:
        retriever = get_retriever()
        retriever.similarity_top_k = top_k
        logger.info(f"Retrieving top {top_k} results for query in '{settings.collection}'")
        return retriever.retrieve(query)
    except Exception as exc:
        logger.error("Retrieval failed", exc_info=exc)
        raise RetrieverError("Retrieval process failed") from exc

def format_context(nodes: List[NodeWithScore]) -> str:
    """
    Formats the retrieved nodes into a single context string to be injected
    into the LLM prompt. Each source is clearly delineated.

    Returns:
        A concatenated string of all sources and their content.
    """
    if not nodes:
        return "No context found."

    parts: list[str] = []
    for idx, node in enumerate(nodes, start=1):
        file_name = node.metadata.get("file_name", "Unknown Source")
        content = " ".join(node.get_content().split())          # tidy up the text
        parts.append(f"Source {idx} (from {file_name}):\n---\n{content}\n---")
    return "\\n\\n".join(parts)
