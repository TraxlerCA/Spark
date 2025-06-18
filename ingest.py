# ingest.py
#
# Ingest documents → split → embed → store in ChromaDB
# -----------------------------------------------------------------
#   python ingest.py --reset   # wipe and rebuild
#   python ingest.py           # incremental ingest
# -----------------------------------------------------------------

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import List

import chromadb
import typer
from llama_index.core import Document, Settings, SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from pydantic import Field
from pydantic_settings import BaseSettings

# ───────────── configuration ─────────────


class AppConfig(BaseSettings):
    source_dir: Path = Field(default=Path("source_documents"))
    persist_dir: Path = Field(default=Path("rag/vectorstores"))
    collection: str = Field(default="default_collection")
    embed_model_id: str = Field(default="BAAI/bge-small-en-v1.5")
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=50)

    class Config:
        env_prefix = "RAG_"
        env_file = ".env"
        env_file_encoding = "utf-8"


# ───────────── logger ─────────────


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

# ───────────── helpers ─────────────


def build_pipeline(cfg: AppConfig) -> IngestionPipeline:
    """splitter + embedder → Chroma"""
    cfg.persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(cfg.persist_dir))
    collection = client.get_or_create_collection(cfg.collection)
    vector_store = ChromaVectorStore(
        chroma_collection=collection,
        stores_text=True,  # keep raw text for inspection
    )

    embedder = HuggingFaceEmbedding(model_name=cfg.embed_model_id)
    Settings.embed_model = embedder
    Settings.chunk_size = cfg.chunk_size
    Settings.chunk_overlap = cfg.chunk_overlap

    return IngestionPipeline(
        transformations=[
            SentenceSplitter(
                chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap
            ),
            embedder,  # does the embedding
        ],
        vector_store=vector_store,
    )


def load_docs(src: Path) -> List[Document]:
    if not src.is_dir():
        raise RuntimeError(f"source dir '{src}' is missing")
    reader = SimpleDirectoryReader(
        input_dir=str(src),
        recursive=True,
        file_metadata=lambda fn: {"file_name": Path(fn).name},
    )
    return reader.load_data()


# ───────────── CLI ─────────────


def main(
    reset: bool = typer.Option(False, "--reset", help="delete existing store first")
) -> None:
    cfg = AppConfig()

    if reset:
        shutil.rmtree(cfg.persist_dir, ignore_errors=True)
        logger.info("old vector store deleted")

    pipeline = build_pipeline(cfg)
    docs = load_docs(cfg.source_dir)
    if not docs:
        logger.warning("no documents found – exiting")
        raise typer.Exit(code=0)

    # keyword args only
    pipeline.run(documents=docs, show_progress=True)

    logger.info(
        json.dumps(
            {
                "event": "ingestion_complete",
                "documents_processed": len(docs),
                "persist_location": str(cfg.persist_dir / cfg.collection),
            }
        )
    )


if __name__ == "__main__":
    typer.run(main)
