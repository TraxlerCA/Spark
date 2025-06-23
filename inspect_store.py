# inspect_store.py
#
# Description: A command‑line utility to inspect the persistent ChromaDB
#              vector store. It can show a collection overview, perform
#              similarity searches, and dump raw document data, which is
#              useful for debugging the RAG pipeline.
#
# --------------------------------------------------------------------------- #
# imports
# --------------------------------------------------------------------------- #
from __future__ import annotations  # enable postponed evaluation of annotations
import json                         # for serialising log payloads
import logging                      # for structured logging
import sys                          # for stdout stream handler
from pathlib import Path            # for filesystem path handling
from typing import Any, List, Optional, TypedDict, cast  # extended typing help

import chromadb                     # for ChromaDB client operations
import numpy as np                  # for numerical operations on embeddings
import typer                        # for CLI argument parsing
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # for embedding model
from rich.console import Console    # for rich console output
from rich.pretty import pprint      # for pretty‑printing complex data

# Import the centralised configuration
from config import settings

# --------------------------------------------------------------------------- #
# type helpers
# --------------------------------------------------------------------------- #
try:
    # Newer Chroma versions expose a stub for this
    from chromadb.api.types import QueryResult  # type: ignore
except ImportError:  # fallback for older versions without stubs

    class QueryResult(TypedDict, total=False):
        """Minimal subset of keys returned by Collection.query and get."""

        metadatas: List[dict[str, Any]]
        documents: List[str]
        embeddings: List[list[float]]
        distances: List[float]

# --------------------------------------------------------------------------- #
# custom exceptions
# --------------------------------------------------------------------------- #
class InspectionError(Exception):
    """Raised when inspection cannot proceed due to errors."""


# --------------------------------------------------------------------------- #
# logger setup
# --------------------------------------------------------------------------- #
class JSONFormatter(logging.Formatter):
    """Custom log formatter that emits single‑line JSON logs."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload = {
            "timestamp": self.formatTime(record, "%Y‑%m‑%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


# Set up module‑level logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def get_collection() -> chromadb.Collection:
    """Connect to and return the specified ChromaDB collection."""

    if not settings.persist_dir.is_dir():
        raise InspectionError(f"Persist directory '{settings.persist_dir}' does not exist.")

    try:
        logger.info(f"Connecting to ChromaDB at '{settings.persist_dir}'")
        client = chromadb.PersistentClient(path=str(settings.persist_dir))
        logger.info(f"Loading collection '{settings.collection}'")
        return client.get_collection(settings.collection)
    except Exception as exc:  # pragma: no cover – needs live DB for coverage
        logger.error("Failed to connect to collection", exc_info=exc)
        raise InspectionError("Collection load failed") from exc


def perform_query(col: chromadb.Collection, query_text: str, n_results: int) -> QueryResult:
    """Perform a similarity search query against the collection."""

    try:
        logger.info(f"Initialising embed model '{settings.embed_model_id}'")
        embed_model = HuggingFaceEmbedding(model_name=settings.embed_model_id)
        emb = embed_model.get_query_embedding(query_text)
        logger.info(f"Performing similarity search for: '{query_text}'")
        res = col.query(
            query_embeddings=[emb],
            n_results=n_results,
            include=["metadatas", "documents", "distances"],
        )
        return cast(QueryResult, res)
    except Exception as exc:  # pragma: no cover – depends on external system
        logger.error("Similarity query failed", exc_info=exc)
        raise InspectionError("Query failed") from exc


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
@typer.run
def main(  # noqa: D401
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Similarity search query."),
    limit: int = typer.Option(5, "--limit", "-l", help="Number of sample documents to show."),
    raw: bool = typer.Option(False, "--raw", help="Print full embedding vectors."),
) -> None:
    """Inspect the ChromaDB store – overview, optional query, and sample docs."""

    console = Console()
    try:
        col = get_collection()

        console.rule("[bold cyan]Collection overview[/bold cyan]", style="cyan")
        count = col.count()
        console.print(f"[bold]Name[/bold]: {col.name}")
        console.print(f"[bold]Total documents[/bold]: {count}")
        console.print()

        if query:
            console.rule(f"[bold yellow]Query results for '{query}'[/bold yellow]", style="yellow")
            res = perform_query(col, query, limit)
            pprint(res, expand_all=True)
            console.print()

        if count:
            console.rule(
                f"[bold green]Sample of {min(limit, count)} documents[/bold green]",
                style="green",
            )
            sample = col.get(
                limit=limit,
                include=["metadatas", "documents", "embeddings"],
            )

            # Turn None into lists and convince the type checker
            metadatas = cast(List[dict[str, Any]], sample.get("metadatas") or [])
            documents = cast(List[str], sample.get("documents") or [])
            embeddings = cast(List[list[float]], sample.get("embeddings") or [])

            for idx, (meta, doc, emb) in enumerate(zip(metadatas, documents, embeddings)):
                console.print(f"[bold magenta]─ Node {idx} ─[/bold magenta]")
                console.print("[italic]Metadata[/italic]:")
                pprint(meta, expand_all=False)
                console.print("[italic]Document snippet[/italic]:")
                # collapse whitespace (newlines, tabs, multiple spaces) into single spaces
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
