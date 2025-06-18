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
import os
import shutil
import sys
from pathlib import Path

import chromadb
import typer
from llama_index.core import Document, Settings, SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# --- Configuration ---
# Use pathlib for robust path handling
CWD = Path(__file__).parent
DIR_INPUT = CWD / "source_documents"
DIR_VECTORSTORE = CWD / "vectorstores"
CHROMA_COLLECTION = "default_collection"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# --- Structured Logging Setup ---
# A simple JSON logger to match your desired output format
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(log_record)

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)
# Prevent double-logging if root logger is also configured
logger.propagate = False


def main(reset: bool = typer.Option(False, "--reset", help="Wipe the vector store and rebuild it.")):
    """
    Ingestion pipeline to load documents, split them, generate embeddings,
    and store them in a ChromaDB vector store.
    """

    # 1. Reset vector store if requested
    if reset and DIR_VECTORSTORE.exists():
        shutil.rmtree(DIR_VECTORSTORE)
        logger.info("old vector store deleted")
    
    DIR_VECTORSTORE.mkdir(parents=True, exist_ok=True)

    # 2. Initialize ChromaDB client and collection
    logger.info(f"connecting to ChromaDB at '{DIR_VECTORSTORE}'")
    db = chromadb.PersistentClient(path=str(DIR_VECTORSTORE))
    chroma_collection = db.get_or_create_collection(CHROMA_COLLECTION)
    logger.info(f"loading collection '{CHROMA_COLLECTION}'")

    # 3. Configure LlamaIndex global settings
    # Set the embedding model for all subsequent operations.
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    # 4. Define and run the Ingestion Pipeline
    # The vector_store object tells the pipeline WHERE to store the processed nodes.
    # This was the missing piece in the original setup.
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=1024, chunk_overlap=200),
        ],
        vector_store=vector_store,
    )

    # 5. Load documents and run them through the pipeline
    # SimpleDirectoryReader will automatically handle various file types.
    # We limit it to PDFs here as in your example.
    if not any(DIR_INPUT.iterdir()):
        logger.warning(f"No documents found in {DIR_INPUT}. Exiting.")
        return

    reader = SimpleDirectoryReader(input_dir=str(DIR_INPUT), required_exts=[".pdf"])
    documents = reader.load_data(show_progress=True)

    # The pipeline.run() method will process and store the documents.
    # LlamaIndex is smart enough to only process new/changed documents
    # unless you force reprocessing.
    pipeline.run(documents=documents, show_progress=True)

    # 6. Log completion summary
    ingestion_summary = {
        "event": "ingestion_complete",
        "documents_processed": len(documents),
        "persist_location": str(DIR_VECTORSTORE / CHROMA_COLLECTION),
    }
    logger.info(json.dumps(ingestion_summary))


if __name__ == "__main__":
    # Ensure the input directory exists
    DIR_INPUT.mkdir(parents=True, exist_ok=True)
    typer.run(main)