# history_utils.py
#
# Description: Provides utilities to manage chat history and construct
#              prompts for the RAG pipeline. It defines structured types
#              for chat messages and includes logic for injecting RAG
#              context into the final LLM prompt.
#

# --------------------------------------------------------------------------- #
# imports
# --------------------------------------------------------------------------- #
from __future__ import annotations
from typing import TypedDict, List, Optional, cast

# --------------------------------------------------------------------------- #
# constants and type definitions
# --------------------------------------------------------------------------- #
# sentinel to detect whether rag_context was provided
_NO_RAG = object()
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Use the conversation history "
    "and provided context to answer user questions."
)
DEFAULT_MAX_TURNS = 20

class ChatMessage(TypedDict):
    """A dictionary representing one turn in a conversation."""
    role: str
    content: str
    timestamp: Optional[str]

History = List[ChatMessage]

# --------------------------------------------------------------------------- #
# prompt formatting
# --------------------------------------------------------------------------- #
def format_prompt(
    history: History,
    next_user_message: str,
    system_prompt: Optional[str] = None,
    rag_context: Optional[str] = cast(Optional[str], _NO_RAG),
    max_turns: int = DEFAULT_MAX_TURNS,
) -> str:
    """
    Builds a single, comprehensive prompt string for the LLM.
    It combines:
    - A system prompt.
    - Recent conversation history.
    - Retrieved RAG context (if available).
    - An instruction for how to use the context.
    - The user's latest question (safely wrapped).
    Returns:
    A fully formatted prompt string ready for the LLM.
    """
    system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    parts: list[str] = [f"{ROLE_SYSTEM}: {system_prompt}"]

    # detect whether rag_context was explicitly passed
    provided_rag = (rag_context is not _NO_RAG)
    use_context = (
        provided_rag
        and rag_context
        and rag_context.strip()
        and rag_context != "No context found."
    )

    # truncate or drop history
    if provided_rag and not use_context:
        truncated_history: list[ChatMessage] = []
    else:
        start_index = max(0, len(history) - max_turns * 2)
        truncated_history = history[start_index:]

    for msg in truncated_history:
        parts.append(f"{msg['role']}: {msg['content']}")

    # sanitize the user message
    safe_user_message = f"<user_question>{next_user_message}</user_question>"

    # inject RAG context only when meaningful
    if use_context:
        instruction = (
            "Use the following context to answer the user's question. "
            "The context is sourced from local documents. "
            "If the answer is not in the context, state that you "
            "could not find an answer in the provided documents."
        )
        combined = (
            f"{instruction}\n\n### Context:\n{rag_context}\n\n"
            f"### User Question:\n{safe_user_message}"
        )
        parts.append(f"{ROLE_USER}: {combined}")
    else:
        parts.append(f"{ROLE_USER}: {safe_user_message}")


    # Final part tells the LLM to begin its response
    parts.append(f"{ROLE_ASSISTANT}:")
    return "\n".join(parts)