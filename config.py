# config.py
#
# Description: Centralized configuration for the RAG (Retrieval-Augmented
#              Generation) application. It uses Pydantic to load settings
#              from a .env file or environment variables, ensuring all
#              modules share a single, consistent configuration.
#

# --------------------------------------------------------------------------- #
# imports
# --------------------------------------------------------------------------- #
from __future__ import annotations  # enable postponed evaluation of annotations
from pathlib import Path            # for handling filesystem paths
from pydantic import Field          # to define configuration fields
from pydantic_settings import BaseSettings # for loading settings from env

# --------------------------------------------------------------------------- #
# settings
# --------------------------------------------------------------------------- #
class AppConfig(BaseSettings):
    """
    Load all application settings from environment variables or defaults.
    This single configuration class is used across all scripts to ensure
    consistency in paths, model identifiers, and other parameters.

    Returns:
        AppConfig: A populated and validated settings instance.
    """

    # --- Source and Storage Paths ---
    source_dir: Path = Field(
        default=Path("source_documents"),
        description="Directory containing input documents for ingestion."
    )
    persist_dir: Path = Field(
        default=Path("rag/vectorstores"),
        description="Directory to store persistent vector database files."
    )
    ingestion_log_path: Path = Field(
        default=Path("rag/ingestion_log.json"),
        description="Path to a log file tracking ingested document hashes."
    )

    # --- Vector Store Settings ---
    collection: str = Field(
        default="default_collection",
        description="Name of the ChromaDB collection."
    )

    # --- Embedding Model ---
    embed_model_id: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Hugging Face model identifier for the embedding model."
    )

    # --- Text Splitting ---
    chunk_size: int = Field(
        default=512,
        description="Maximum number of tokens per text chunk."
    )
    chunk_overlap: int = Field(
        default=50,
        description="Number of token overlap between consecutive chunks."
    )

    # --- Retriever Settings ---
    similarity_top_k: int = Field(
        default=4,
        description="Number of top similar nodes to retrieve from the vector store."
    )
    abs_min_score: float = Field(
        default=0.35,
        description="Absolute minimum similarity score to consider a node relevant."
    )
    rel_window: float = Field(
        default=0.05,
        description="Keep nodes with scores within this window of the top score."
    )

    # --- Ollama LLM Settings ---
    ollama_host: str = Field(
        default="http://localhost",
        description="Ollama server host URL. Use 'https://' for non-local hosts."
    )

    ollama_port: int = Field(
        default=11434,
        description="Ollama server port."
    )
    ollama_model: str = Field(
        default="gemma3:4b",
        description="Default Ollama model name to use for generation."
    )
    ollama_timeout: int = Field(
        default=60,
        description="Request timeout for the Ollama server in seconds."
    )

    # Feature flags
    enable_answerability_check: bool = Field(
        default=True,
        description="Activate yes/no answerability check before composing the final answer.",
    )

    # pydantic v2 style configuration
    model_config = {
        "env_prefix": "RAG_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }

# --------------------------------------------------------------------------- #
# global instance
# --------------------------------------------------------------------------- #
# Create a single, cached instance of the configuration that can be
# imported by any other module in the application.
settings = AppConfig()