# core.py
# Description: Core business logic for the RAG agent. Handles query
# processing, context retrieval, and response generation.

from __future__ import annotations
import logging
from typing import Set, Generator, Tuple, List, Any
from config import settings
from history_utils import format_prompt, History
from llm_client import stream_llm_response

# Optional RAG integration
try:
    from rag.retriever import retrieve, format_context, NodeWithScore
    RAG_ENABLED = True
except ImportError:
    RAG_ENABLED = False

    # fallback stubs – only defined if the import really failed
    class NodeWithScore:  # type: ignore
        pass

    def retrieve(*args, **kwargs) -> List[NodeWithScore]:  # type: ignore
        return []

    def format_context(*args, **kwargs) -> str:  # type: ignore
        return ""

logger = logging.getLogger(__name__)
# Centralized RAG filtering parameters
ABS_MIN_SCORE = 0.35
REL_WINDOW = 0.05

def extract_sources(nodes: List[NodeWithScore]) -> Set[str]:
    """Collects unique file names from retrieved nodes."""
    return {n.metadata.get("file_name", "unknown") for n in nodes}

def process_query(
    user_prompt: str,
    history: History,
    use_rag: bool = True,
    system_prompt: str | None = None,
    model: str | None = None,
) -> Tuple[Generator[str, None, None], Set[str]]:
    """
    Processes a user query, optionally uses RAG, and returns a response generator and sources.
    Returns:
    - A generator that streams the LLM response.
    - A set of source document filenames.
    """
    rag_context = ""
    sources: Set[str] = set()

    if use_rag and RAG_ENABLED:
        logger.info("RAG is enabled; retrieving context...")
        try:
            retrieved_nodes = retrieve(user_prompt, k=settings.similarity_top_k)

            good_nodes = []
            if retrieved_nodes:
                top_score = retrieved_nodes[0].score
                if top_score and top_score >= ABS_MIN_SCORE:
                    good_nodes.append(retrieved_nodes[0])
                for n in retrieved_nodes[1:]:
                    if n.score and n.score >= max(ABS_MIN_SCORE, top_score - REL_WINDOW):
                        good_nodes.append(n)

            if good_nodes:
                rag_context = format_context(good_nodes)
                sources = extract_sources(good_nodes)
                logger.info("Context retrieved successfully.")
            else:
                logger.info("No nodes passed the similarity threshold; skipping RAG.")

        except Exception as e:
            logger.error("Failed to retrieve context", extra={"error": str(e)})
            # Proceed without context, but log the failure

    # Use the unified prompt formatting function
    final_prompt = format_prompt(
        history=history,
        next_user_message=user_prompt,
        rag_context=rag_context,
        system_prompt=system_prompt,
    )

    response_generator = stream_llm_response(final_prompt, model=model)

    return response_generator, sources
