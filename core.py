# core.py
# Description: Core business logic for the RAG agent. Handles query
# processing, context retrieval, and response generation.

from __future__ import annotations

import logging
from typing import (
    Any,
    Generator,
    Mapping,
    Protocol,
    Sequence,
    Set,
    Tuple,
    TYPE_CHECKING,
    List,
)

from config import settings
from history_utils import format_prompt, History
from llm_client import stream_llm_response, get_llm_completion

# ---------------------------------------------------------------------------
# structural protocol used for static checking
# ---------------------------------------------------------------------------
class _NodeLike(Protocol):
    metadata: Mapping[str, Any]
    score: float | None


# ---------------------------------------------------------------------------
# optional RAG integration
# ---------------------------------------------------------------------------
try:
    from rag.retriever import retrieve as _retrieve, format_context as _format_context
    RAG_ENABLED = True
except ImportError:  # pragma: no cover
    RAG_ENABLED = False
    _retrieve = _format_context = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from llama_index.core.schema import NodeWithScore as NodeWithScore  # noqa: F401

    def retrieve(query: str, k: int | None = None) -> Sequence[NodeWithScore]: ...
    def format_context(nodes: Sequence[NodeWithScore]) -> str: ...
else:
    if _retrieve is not None:
        retrieve = _retrieve  # type: ignore[assignment]
    else:  # pragma: no cover
        def retrieve(*args, **kwargs):  # type: ignore[return-value]
            return []

    if _format_context is not None:
        format_context = _format_context  # type: ignore[assignment]
    else:  # pragma: no cover
        def format_context(*args, **kwargs):  # type: ignore[return-value]
            return ""


logger = logging.getLogger(__name__)
# Centralized RAG filtering parameters
ABS_MIN_SCORE = 0.35
REL_WINDOW = 0.05

# ---------------------------------------------------------------------------
# answerability gate – a quick yes/no guard before we commit to RAG
# ---------------------------------------------------------------------------
ANSWERABILITY_TEMPLATE = """
You are a decision function. Read the question and the context.
If the context already contains everything needed for a complete and correct answer,
reply with "yes". Otherwise reply with "no". Output only "yes" or "no".

<question>
{question}
</question>

<context>
{context}
</context>
"""


def context_is_answerable(
    question: str, context: str, model: str | None = None
) -> bool:
    """
    Return True when the supplied context is sufficient to answer the question.
    """
    if not context.strip():
        return False  # no context means not answerable
    prompt = ANSWERABILITY_TEMPLATE.format(question=question, context=context)
    try:
        result = get_llm_completion(prompt, model=model)
        return result.lower().startswith("y")
    except Exception:
        # fail open – if the check crashes, keep the context
        logger.exception("Answerability check failed, proceeding with context")
        return True


def extract_sources(nodes: Sequence[_NodeLike]) -> Set[str]:
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
            retrieved_nodes: Sequence[_NodeLike] = retrieve(  # type: ignore
                user_prompt, k=settings.similarity_top_k
            )

            good_nodes: list[_NodeLike] = []
            if retrieved_nodes:
                top_score = retrieved_nodes[0].score

                # keep the original first-node filter, but be explicit
                if top_score is not None and top_score >= ABS_MIN_SCORE:
                    good_nodes.append(retrieved_nodes[0])

                # decide once which threshold we are going to use
                relative_cutoff: float = (
                    top_score - REL_WINDOW if top_score is not None else float("-inf")
                )
                threshold = max(ABS_MIN_SCORE, relative_cutoff)

                for n in retrieved_nodes[1:]:
                    if n.score is not None and n.score >= threshold:
                        good_nodes.append(n)

            if good_nodes:
                rag_context = format_context(good_nodes)  # type: ignore[arg-type]
                sources = extract_sources(good_nodes)
                logger.info("Context retrieved successfully.")

                # answerability gate – run after we have the context
                if settings.enable_answerability_check and rag_context:
                    if not context_is_answerable(
                        user_prompt, rag_context, model=model
                    ):
                        logger.info(
                            "Context insufficient – falling back to internal knowledge"
                        )
                        rag_context = ""
                        sources.clear()  # hide irrelevant citations
            else:
                logger.info(
                    "No nodes passed the similarity threshold; skipping RAG."
                )

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
