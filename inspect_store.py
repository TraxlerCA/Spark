# inspect_store.py
#
# Inspect a persistent ChromaDB vector store used by the RAG pipeline.
# ─────────────────────────────────────────────────────────────────────
#   python inspect_store.py                  # overview + sample
#   python inspect_store.py -q "your query"  # similarity search
#   python inspect_store.py --raw            # dump full embeddings
# ─────────────────────────────────────────────────────────────────────

# --------------------------------------------------------------------------- #
# imports
# --------------------------------------------------------------------------- #
from __future__ import annotations  # enable postponed evaluation of annotations
import json                         # for serializing log payloads
import logging                      # for structured logging
import sys                          # for stdout stream handler
from pathlib import Path           # for filesystem path handling
from typing import Optional        # for optional type hints

import chromadb                     # for ChromaDB client operations
import numpy as np                  # for numerical operations on embeddings
import typer                        # for CLI argument parsing
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # for embedding model
from pydantic import Field         # to define settings fields
from pydantic_settings import BaseSettings  # for environment-based config
from rich.console import Console    # for rich console output
from rich.pretty import pprint      # for pretty-printing complex data

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
class AppConfig(BaseSettings):
    """
    load settings from environment variables or defaults.

    attrs:
      persist_dir (Path): directory containing vector store files.
      collection (str): name of the ChromaDB collection.
      embed_model_id (str): Hugging Face model identifier for embeddings.
    """
    persist_dir: Path = Field(
        default=Path("rag/vectorstores"), 
        description="directory for ChromaDB persistence"
    )
    collection: str = Field(
        default="default_collection", 
        description="name of the collection to inspect"
    )
    embed_model_id: str = Field(
        default="BAAI/bge-small-en-v1.5", 
        description="Hugging Face embedding model ID"
    )

    class Config:
        env_prefix = "RAG_"             # prefix for env vars (e.g., RAG_PERSIST_DIR)
        env_file = ".env"               # file to load environment variables from
        env_file_encoding = "utf-8"     # encoding of the .env file

class InspectionError(Exception):
    """raised when inspection cannot proceed due to errors."""

# --------------------------------------------------------------------------- #
# structured logging
# --------------------------------------------------------------------------- #
class JSONFormatter(logging.Formatter):
    """
    custom log formatter that emits single-line JSON logs.

    methods:
      format: build JSON payload from LogRecord attributes.
    """
    def format(self, record: logging.LogRecord) -> str:
        # build structured log payload
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # include exception details if present
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)

# set up module-level logger
logger = logging.getLogger(__name__)     # create logger for this module
handler = logging.StreamHandler(sys.stdout)  # log to stdout
handler.setFormatter(JSONFormatter())    # apply JSON formatter
logger.addHandler(handler)               # attach handler
logger.setLevel(logging.INFO)            # default log level

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def get_collection(cfg: AppConfig) -> chromadb.Collection:
    """
    connect to and return the specified ChromaDB collection.

    args:
      cfg (AppConfig): application configuration.

    returns:
      chromadb.Collection: loaded collection instance.

    raises:
      InspectionError: if directory missing or connection fails.
    """
    # ensure persistence directory exists
    if not cfg.persist_dir.is_dir():
        raise InspectionError(f"persist directory '{cfg.persist_dir}' does not exist")

    try:
        logger.info(f"connecting to ChromaDB at '{cfg.persist_dir}'")
        client = chromadb.PersistentClient(path=str(cfg.persist_dir))
        logger.info(f"loading collection '{cfg.collection}'")
        return client.get_collection(cfg.collection)
    except Exception as exc:
        logger.error("failed to connect to collection", exc_info=exc)
        raise InspectionError("collection load failed") from exc

def perform_query(
    col: chromadb.Collection,
    cfg: AppConfig,
    query_text: str,
    n_results: int,
) -> dict:
    """
    perform a similarity search query on the collection.

    args:
      col (chromadb.Collection): target collection.
      cfg (AppConfig): application configuration.
      query_text (str): text to embed and query.
      n_results (int): number of nearest neighbors to retrieve.

    returns:
      dict: query results including metadatas, documents, distances.

    raises:
      InspectionError: if query or embedding fails.
    """
    try:
        logger.info(f"initialising embed model '{cfg.embed_model_id}'")
        embed_model = HuggingFaceEmbedding(model_name=cfg.embed_model_id)
        emb = embed_model.get_query_embedding(query_text)
        logger.info(f"similarity search for: '{query_text}'")
        return col.query(
            query_embeddings=[emb],
            n_results=n_results,
            include=["metadatas", "documents", "distances"],
        )
    except Exception as exc:
        logger.error("similarity query failed", exc_info=exc)
        raise InspectionError("query failed") from exc

# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def main(
    query: Optional[str] = typer.Option(
        None, "--query", "-q", help="similarity search against the store"
    ),
    limit: int = typer.Option(
        5, "--limit", "-l", help="number of sample documents to show"
    ),
    raw: bool = typer.Option(
        False, "--raw", help="print full embedding vectors instead of summary"
    ),
) -> None:
    """
    inspect the ChromaDB store: overview, optional query, and sample documents.

    workflow:
      - load config and collection
      - display overview (name, count)
      - if query provided: perform similarity search and display results
      - show a sample of documents with metadata and embedding summaries
      - exit with code 1 on InspectionError
    """
    console = Console()  # for rich-formatted console output
    cfg = AppConfig()    # load application settings

    try:
        # load collection or raise error
        col = get_collection(cfg)

        # overview section
        console.rule("[bold cyan]Collection overview[/bold cyan]", style="cyan")
        count = col.count()
        console.print(f"[bold]Name[/bold]: {col.name}")
        console.print(f"[bold]Total documents[/bold]: {count}")
        console.print()

        # optional similarity query
        if query:
            console.rule(f"[bold yellow]Query results for '{query}'[/bold yellow]", style="yellow")
            res = perform_query(col, cfg, query, limit)
            pprint(res, expand_all=True)
            console.print()

        # sample documents section
        if count:
            console.rule(f"[bold green]Sample of {min(limit, count)} documents[/bold green]", style="green")
            sample = col.get(
                limit=limit,
                include=["metadatas", "documents", "embeddings"],
            )

            for idx, (meta, doc, emb) in enumerate(
                zip(sample["metadatas"], sample["documents"], sample["embeddings"])
            ):
                # display node header
                console.print(f"[bold magenta]─ node {idx} ─[/bold magenta]")
                console.print("[italic]metadata[/italic]:")
                pprint(meta, expand_all=False)

                console.print("[italic]document snippet[/italic]:")
                console.print(doc.replace("\n", " ")[:240] + ("…" if len(doc) > 240 else ""))

                console.print("[italic]embedding[/italic]:")
                if emb is None:
                    console.print("[red]null vector[/red]")
                elif raw:
                    pprint(emb)
                else:
                    # convert to numpy array for stats and norms
                    vec = np.asarray(emb, dtype=np.float32)
                    console.print(
                        f"dim={vec.size} ‖vec‖₂={np.linalg.norm(vec):.4f} "
                        f"min={vec.min():.4f} max={vec.max():.4f}"
                    )
                console.print()

            # check for any null embeddings indicating ingestion failure
            if any(e is None for e in sample["embeddings"]):
                console.print("[bold red]null vectors detected – ingestion failed[/bold red]")
        else:
            logger.warning("collection is empty")

    except InspectionError:
        # exit with error code on inspection failure
        raise typer.Exit(code=1)

if __name__ == "__main__":
    typer.run(main)
