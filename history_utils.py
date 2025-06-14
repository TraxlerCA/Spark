# history_utils.py
#
# Description: Utilities for formatting and trimming conversation history into LLM prompts.
#

from typing import TypedDict, List, Optional

# --------------------------------------------------------------------------- #
# Constants and Type Definitions
# --------------------------------------------------------------------------- #

ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"


class ChatMessage(TypedDict):
    """A structured dictionary representing one turn in a conversation."""
    role: str   # ROLE_SYSTEM, ROLE_USER, or ROLE_ASSISTANT
    content: str


History = List[ChatMessage]

# Default system instruction to guide the assistant
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Use the conversation history to answer user questions."
)

# --------------------------------------------------------------------------- #
# Prompt Formatting
# --------------------------------------------------------------------------- #


def format_prompt(
    history: History,
    next_user_message: str,
    system_prompt: Optional[str] = None,
    max_turns: int = 20
) -> str:
    """
    Build a single prompt string by concatenating a system message,
    the most recent conversation turns, and the incoming user message.

    Args:
        history: List of past ChatMessage entries.
        next_user_message: The new message from the user.
        system_prompt: Optional top-level system instruction.
        max_turns: Maximum number of past user–assistant pairs to include.

    Returns:
        A formatted prompt string ready to send to ask_llm().
    """
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    parts: list[str] = []
    # add the system instruction
    parts.append(f"{ROLE_SYSTEM}: {system_prompt}")

    # include only the last max_turns * 2 messages
    start_index = max(0, len(history) - max_turns * 2)
    for msg in history[start_index:]:
        parts.append(f"{msg['role']}: {msg['content']}")

    # add the new user message
    parts.append(f"{ROLE_USER}: {next_user_message}")
    # signal assistant to reply
    parts.append(f"{ROLE_ASSISTANT}:")

    return "\n".join(parts)
