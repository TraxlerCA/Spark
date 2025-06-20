# inspect_store.py
#
# Description: A command-line utility to inspect the persistent ChromaDB
#              vector store. It can show a collection overview, perform
#              similarity searches, and dump raw document data, which is
#              useful for debugging the RAG pipeline.
#

# --------------------------------------------------------------------------- #
# imports
# --------------------------------------------------------------------------- #
from __future__ import annotations  # enable postponed evaluation of annotations
import json                         # for serializing log payloads
import logging                      # for structured logging
import sys                          # for stdout stream handler
from pathlib import Path            # for filesystem path handling
from typing import Optional         # for optional type hints

import chromadb                     # for ChromaDB client operations
import numpy as np                  # for numerical operations on embeddings
import typer                        # for CLI argument parsing
from llama_index.embeddings.huggingface import HuggingFaceEmbedding # for embedding model
from rich.console import Console    # for rich console output
from rich.pretty import pprint      # for pretty-printing complex data

# Import the centralized configuration
from config import settings

# --------------------------------------------------------------------------- #
# custom exceptions
# --------------------------------------------------------------------------- #
class InspectionError(Exception):
    """Custom exception raised when inspection cannot proceed due to errors."""

# --------------------------------------------------------------------------- #
# logger setup
# --------------------------------------------------------------------------- #
class JSONFormatter(logging.Formatter):
    """Custom log formatter that emits single-line JSON logs."""
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)

# Set up module-level logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def get_collection() -> chromadb.Collection:
    """
    Connects to and returns the specified ChromaDB collection.

    Returns:
      chromadb.Collection: The loaded collection instance.

    Raises:
      InspectionError: If the persistence directory is missing or connection fails.
    """
    if not settings.persist_dir.is_dir():
        raise InspectionError(f"Persist directory '{settings.persist_dir}' does not exist.")
    try:
        logger.info(f"Connecting to ChromaDB at '{settings.persist_dir}'")
        client = chromadb.PersistentClient(path=str(settings.persist_dir))
        logger.info(f"Loading collection '{settings.collection}'")
        return client.get_collection(settings.collection)
    except Exception as exc:
        logger.error("Failed to connect to collection", exc_info=exc)
        raise InspectionError("Collection load failed") from exc

def perform_query(col: chromadb.Collection, query_text: str, n_results: int) -> dict:
    """
    Performs a similarity search query against the collection.
    """
    try:
        logger.info(f"Initialising embed model '{settings.embed_model_id}'")
        embed_model = HuggingFaceEmbedding(model_name=settings.embed_model_id)
        emb = embed_model.get_query_embedding(query_text)
        logger.info(f"Performing similarity search for: '{query_text}'")
        return col.query(
            query_embeddings=[emb],
            n_results=n_results,
            include=["metadatas", "documents", "distances"],
        )
    except Exception as exc:
        logger.error("Similarity query failed", exc_info=exc)
        raise InspectionError("Query failed") from exc

# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
@typer.run
def main(
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Similarity search query."),
    limit: int = typer.Option(5, "--limit", "-l", help="Number of sample documents to show."),
    raw: bool = typer.Option(False, "--raw", help="Print full embedding vectors."),
) -> None:
    """
    Inspects the ChromaDB store: shows an overview, an optional query result,
    and a sample of stored documents with their metadata.
    """
    console = Console()
    try:
        col = get_collection()

        console.rule("[bold cyan]Collection Overview[/bold cyan]", style="cyan")
        count = col.count()
        console.print(f"[bold]Name[/bold]: {col.name}")
        console.print(f"[bold]Total documents[/bold]: {count}")
        console.print()

        if query:
            console.rule(f"[bold yellow]Query Results for '{query}'[/bold yellow]", style="yellow")
            res = perform_query(col, query, limit)
            pprint(res, expand_all=True)
            console.print()

        if count:
            console.rule(f"[bold green]Sample of {min(limit, count)} Documents[/bold green]", style="green")
            sample = col.get(limit=limit, include=["metadatas", "documents", "embeddings"])

            for idx, (meta, doc, emb) in enumerate(
                zip(sample["metadatas"], sample["documents"], sample["embeddings"])
            ):
                console.print(f"[bold magenta]─ Node {idx} ─[/bold magenta]")
                console.print("[italic]Metadata[/italic]:")
                pprint(meta, expand_all=False)
                console.print("[italic]Document Snippet[/italic]:")
                # collapse any whitespace (newlines, tabs, multiple spaces) into single spaces
                cleaned = " ".join(doc.split())
                snippet = cleaned[:240] + ("..." if len(cleaned) > 240 else "")
                console.print(snippet)
                console.print("[italic]Embedding[/italic]:")
                if emb is None:
                    console.print("[red]Null vector[/red]")
                elif raw:
                    pprint(emb)
                else:
                    vec = np.asarray(emb, dtype=np.float32)
                    console.print(
                        f"dim={vec.size} ‖vec‖₂={np.linalg.norm(vec):.4f} "
                        f"min={vec.min():.4f} max={vec.max():.4f}"
                    )
                console.print()
        else:
            logger.warning("Collection is empty.")

    except InspectionError:
        raise typer.Exit(code=1)
