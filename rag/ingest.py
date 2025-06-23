# ingest.py
#
# Description: Ingests documents from a source directory, splits them into
#              chunks, creates vector embeddings, and stores them in a local
#              ChromaDB collection. Implements incremental ingestion to only
#              process new or updated files, improving efficiency.
#

# --------------------------------------------------------------------------- #
# imports
# --------------------------------------------------------------------------- #
from __future__ import annotations  # enable postponed evaluation of annotations
import json                     # for serializing logs and metadata
import logging                  # for structured logging
import shutil                   # to remove directories when resetting
import sys                      # to access stdout for logging handler
import hashlib                  # for creating file content hashes
from pathlib import Path        # for filesystem paths
from typing import List         # for type hints on returned document lists

import chromadb                 # ChromaDB client library
import typer                    # for building the command-line interface
from llama_index.core import Document, Settings, SimpleDirectoryReader # core ingestion
from llama_index.core.ingestion import IngestionPipeline # pipeline orchestration
from llama_index.core.node_parser import SentenceSplitter # splitter for chunking
from llama_index.embeddings.huggingface import HuggingFaceEmbedding # embedding model
from llama_index.vector_stores.chroma import ChromaVectorStore # vector store wrapper
from config import settings
from logging_config import setup_logging


# Import the centralized configuration
from config import settings

# --------------------------------------------------------------------------- #
# logger setup
# --------------------------------------------------------------------------- #
logger = setup_logging()

# --------------------------------------------------------------------------- #
# helper functions
# --------------------------------------------------------------------------- #
def _load_ingestion_log() -> dict[str, str]:
    """
    Loads the ingestion log file which maps filenames to their content hashes.
    This log is used to track which files have been processed.

    Returns:
        dict[str, str]: A dictionary of {filename: hash}.
    """
    if settings.ingestion_log_path.exists():
        with open(settings.ingestion_log_path, "r") as f:
            return json.load(f)
    return {}

def _save_ingestion_log(log: dict[str, str]) -> None:
    """Saves the provided ingestion log to a file."""
    with open(settings.ingestion_log_path, "w") as f:
        json.dump(log, f, indent=2)

def _get_file_hash(filepath: Path) -> str:
    """
    Computes the SHA-256 hash of a file's content to detect changes.

    Returns:
        str: The hex digest of the file's hash.
    """
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def build_pipeline() -> IngestionPipeline:
    """
    Assembles and returns a LlamaIndex IngestionPipeline.

    The pipeline is configured with:
      1. A persistent ChromaDB vector store.
      2. A HuggingFace embedding model.
      3. A sentence splitter for chunking text.
    """
    settings.persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.persist_dir))
    collection = client.get_or_create_collection(settings.collection)
    vector_store = ChromaVectorStore(chroma_collection=collection, stores_text=True)
    embedder = HuggingFaceEmbedding(model_name=settings.embed_model_id)

    Settings.embed_model = embedder
    Settings.chunk_size = settings.chunk_size
    Settings.chunk_overlap = settings.chunk_overlap

    return IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap),
            embedder,
        ],
        vector_store=vector_store,
    )

def load_new_docs() -> List[Document]:
    """
    Reads files from the source directory, but intelligently skips any files
    that have already been ingested and have not changed. This is determined
    by comparing file hashes stored in an ingestion log. It now uses
    relative paths as keys to avoid filename collisions.
    """
    src = settings.source_dir
    if not src.is_dir():
        raise RuntimeError(f"Source directory '{src}' is missing")

    ingestion_log = _load_ingestion_log()
    new_or_updated_files = []

    # Use relative paths as unique identifiers
    current_files = {
        str(p.relative_to(src)): p for p in src.rglob("*") if p.is_file()
    }

    for rel_path, filepath in current_files.items():
        file_hash = _get_file_hash(filepath)
        if ingestion_log.get(rel_path) != file_hash:
            logger.info(f"Detected new or updated file: '{rel_path}'")
            new_or_updated_files.append(str(filepath))
            ingestion_log[rel_path] = file_hash

    if not new_or_updated_files:
        return []

    _save_ingestion_log(ingestion_log)

    # The file_metadata lambda now uses the relative path for consistency
    reader = SimpleDirectoryReader(
        input_files=new_or_updated_files,
        file_metadata=lambda fn: {"file_name": str(Path(fn).relative_to(src))},
    )
    return reader.load_data()

# --------------------------------------------------------------------------- #
# command-line interface
# --------------------------------------------------------------------------- #
@typer.run
def main(
    reset: bool = typer.Option(
        False,
        "--reset",
        help="Delete existing vector store and ingestion log before starting."
    )
) -> None:
    """
    Main entry point for the ingestion script.

    Workflow:
      1. Optionally resets the vector store and log.
      2. Builds the ingestion pipeline.
      3. Loads only new or updated documents.
      4. Runs the documents through the pipeline.
      5. Logs a summary of the operation.
    """
    if reset:
        if settings.persist_dir.exists():
            shutil.rmtree(settings.persist_dir)
        if settings.ingestion_log_path.exists():
            settings.ingestion_log_path.unlink()
        logger.info("Old vector store and ingestion log deleted")

    pipeline = build_pipeline()
    docs = load_new_docs()

    if not docs:
        logger.info("No new or updated documents found to ingest.")
        return

    pipeline.run(documents=docs, show_progress=True)

    logger.info(json.dumps({
        "event": "ingestion_complete",
        "documents_processed": len(docs),
        "persist_location": str(settings.persist_dir / settings.collection),
    }))
