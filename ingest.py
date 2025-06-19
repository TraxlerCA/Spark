# ingest.py
#
# Description: ingest documents, split them into sentences, embed the chunks,
#              and store both vectors and raw text in a local ChromaDB collection.
#
# Usage examples:
#   $ python ingest.py --reset    # remove existing database and fully rebuild
#   $ python ingest.py            # add new documents incrementally

from __future__ import annotations

import json           # for serializing logs and completion payload
import logging        # for structured logging
import shutil         # to remove directories when resetting
import sys            # to access stdout for logging handler
from pathlib import Path  # for filesystem paths
from typing import List   # for type hints on returned document lists

import chromadb       # ChromaDB client library
import typer          # CLI argument parsing
from llama_index.core import Document, Settings, SimpleDirectoryReader  # core ingestion components
from llama_index.core.ingestion import IngestionPipeline  # pipeline orchestration
from llama_index.core.node_parser import SentenceSplitter  # splitter to chunk text
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # embedding model
from llama_index.vector_stores.chroma import ChromaVectorStore  # vector store wrapper
from pydantic import Field  # to define settings fields
from pydantic_settings import BaseSettings  # for environment-based configuration

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

class AppConfig(BaseSettings):
    """
    load application settings from environment variables or defaults.
    - source_dir: directory containing input documents
    - persist_dir: where to store vector database files
    - collection: name of the ChromaDB collection
    - embed_model_id: Hugging Face model identifier
    - chunk_size: max tokens per text chunk
    - chunk_overlap: token overlap between consecutive chunks
    """

    source_dir: Path = Field(default=Path("source_documents"))
    persist_dir: Path = Field(default=Path("rag/vectorstores"))
    collection: str = Field(default="default_collection")
    embed_model_id: str = Field(default="BAAI/bge-small-en-v1.5")
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=50)

    class Config:
        # prefix for environment vars, e.g. RAG_SOURCE_DIR
        env_prefix = "RAG_"
        env_file = ".env"
        env_file_encoding = "utf-8"

# --------------------------------------------------------------------------- #
# logger setup
# --------------------------------------------------------------------------- #

class JSONFormatter(logging.Formatter):
    """
    custom log formatter that emits events as single-line JSON.
    includes timestamp, log level, logger name, message, and optional exception.
    """

    def format(self, record: logging.LogRecord) -> str:
        # build structured log payload
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # if an exception occurred, include its stack trace
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)

# create and configure the root logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
# helper functions
# --------------------------------------------------------------------------- #

def build_pipeline(cfg: AppConfig) -> IngestionPipeline:
    """
    assemble and return an IngestionPipeline:
      1. ensure persistence directory exists
      2. initialize ChromaDB client & collection
      3. configure llama-index settings for embedding
      4. create pipeline with a sentence splitter and the embedder
    """
    # make sure the directory for vector store exists
    cfg.persist_dir.mkdir(parents=True, exist_ok=True)

    # connect to or create a persistent ChromaDB collection
    client = chromadb.PersistentClient(path=str(cfg.persist_dir))
    collection = client.get_or_create_collection(cfg.collection)

    # wrap the collection with llama-index vector store interface
    vector_store = ChromaVectorStore(
        chroma_collection=collection,
        stores_text=True,  # keep raw text alongside vectors
    )

    # instantiate the Hugging Face embedder
    embedder = HuggingFaceEmbedding(model_name=cfg.embed_model_id)

    # set global embedding and chunk parameters for llama-index
    Settings.embed_model = embedder
    Settings.chunk_size = cfg.chunk_size
    Settings.chunk_overlap = cfg.chunk_overlap

    # return the composed pipeline: splitter followed by embedding
    return IngestionPipeline(
        transformations=[
            SentenceSplitter(
                chunk_size=cfg.chunk_size,
                chunk_overlap=cfg.chunk_overlap,
            ),
            embedder,
        ],
        vector_store=vector_store,
    )


def load_docs(src: Path) -> List[Document]:
    """
    read all files from the source directory into llama-index Documents.
    raises if the source directory is missing.
    """
    if not src.is_dir():
        # fail fast if the path is invalid
        raise RuntimeError(f"source dir '{src}' is missing")

    # SimpleDirectoryReader crawls files, attaches file_name metadata
    reader = SimpleDirectoryReader(
        input_dir=str(src),
        recursive=True,
        file_metadata=lambda fn: {"file_name": Path(fn).name},
    )
    return reader.load_data()

# --------------------------------------------------------------------------- #
# command-line interface
# --------------------------------------------------------------------------- #

def main(
    reset: bool = typer.Option(
        False,
        "--reset",
        help="delete existing store first"
    )
) -> None:
    """
    parse CLI args, optionally reset store, then ingest documents:
      1. load config
      2. if --reset: delete existing persisted data
      3. build the ingestion pipeline
      4. load documents
      5. run the pipeline and show progress
      6. log completion event as JSON
    """
    # load config from env or defaults
    cfg = AppConfig()

    # remove old vector store files if requested
    if reset:
        shutil.rmtree(cfg.persist_dir, ignore_errors=True)
        logger.info("old vector store deleted")

    # prepare pipeline and load docs
    pipeline = build_pipeline(cfg)
    docs = load_docs(cfg.source_dir)

    # if no documents found, warn and exit cleanly
    if not docs:
        logger.warning("no documents found, exiting")
        raise typer.Exit(code=0)

    # perform ingestion: splitting, embedding, saving
    pipeline.run(documents=docs, show_progress=True)

    # emit final summary event with counts and storage path
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
