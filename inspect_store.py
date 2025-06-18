# inspect_store.py
#
# Inspect a persistent ChromaDB vector store used by the RAG pipeline.
# ─────────────────────────────────────────────────────────────────────
#   python inspect_store.py                  # overview + sample
#   python inspect_store.py -q "your query"  # similarity search
#   python inspect_store.py --raw            # dump full embeddings
# ─────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import chromadb
import numpy as np
import typer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pydantic import Field
from pydantic_settings import BaseSettings
from rich.console import Console
from rich.pretty import pprint

# ───────────────────────── configuration ──────────────────────────


class AppConfig(BaseSettings):
    """Load settings from env (.env) or defaults."""

    persist_dir: Path = Field(default=Path("rag/vectorstores"))
    collection: str = Field(default="default_collection")
    embed_model_id: str = Field(default="BAAI/bge-small-en-v1.5")

    class Config:
        env_prefix = "RAG_"
        env_file = ".env"
        env_file_encoding = "utf-8"


class InspectionError(Exception):
    """Raised when inspection cannot proceed."""


# ───────────────────────── structured logging ─────────────────────


class JSONFormatter(logging.Formatter):
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


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ───────────────────────── helpers ────────────────────────────────


def get_collection(cfg: AppConfig) -> chromadb.Collection:
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


# ───────────────────────── CLI ────────────────────────────────────


def main(
    query: Optional[str] = typer.Option(
        None, "--query", "-q", help="Similarity search against the store."
    ),
    limit: int = typer.Option(
        5, "--limit", "-l", help="Number of sample documents to show."
    ),
    raw: bool = typer.Option(
        False,
        "--raw",
        help="Print full embedding vectors instead of a summary.",
    ),
) -> None:
    console = Console()
    cfg = AppConfig()

    try:
        col = get_collection(cfg)

        # overview
        console.rule("[bold cyan]Collection overview[/bold cyan]", style="cyan")
        count = col.count()
        console.print(f"[bold]Name[/bold]: {col.name}")
        console.print(f"[bold]Total documents[/bold]: {count}")
        console.print()

        # optional query
        if query:
            console.rule(f"[bold yellow]Query results for '{query}'[/bold yellow]", style="yellow")
            res = perform_query(col, cfg, query, limit)
            pprint(res, expand_all=True)
            console.print()

        # sample
        if count:
            console.rule(f"[bold green]Sample of {min(limit, count)} documents[/bold green]", style="green")
            sample = col.get(
                limit=limit, include=["metadatas", "documents", "embeddings"]
            )

            for idx, (meta, doc, emb) in enumerate(
                zip(sample["metadatas"], sample["documents"], sample["embeddings"])
            ):
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
                    vec = np.asarray(emb, dtype=np.float32)
                    console.print(
                        f"dim={vec.size} ‖vec‖₂={np.linalg.norm(vec):.4f} "
                        f"min={vec.min():.4f} max={vec.max():.4f}"
                    )
                console.print()

            # sanity-check for nulls
            if any(e is None for e in sample["embeddings"]):
                console.print("[bold red]null vectors detected – ingestion failed[/bold red]")
        else:
            logger.warning("collection is empty")

    except InspectionError:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)