# history_utils.py
#
# Description: Utilities for formatting and trimming conversation history into LLM prompts.
#

from typing import TypedDict     # to define structured chat message types
from typing import List          # for list type hints
from typing import Optional      # for optional function parameters

# --------------------------------------------------------------------------- #
# constants and type definitions
# --------------------------------------------------------------------------- #

ROLE_SYSTEM = "system"           # system role identifier
ROLE_USER = "user"               # user role identifier
ROLE_ASSISTANT = "assistant"     # assistant role identifier

class ChatMessage(TypedDict):
    """
    Represents a single conversation turn.

    Attributes:
        role: one of ROLE_SYSTEM, ROLE_USER, or ROLE_ASSISTANT.
        content: textual content of the message.
    """
    role: str   # role of the message sender
    content: str  # content of the message

History = List[ChatMessage]       # alias for a list of chat messages

DEFAULT_SYSTEM_PROMPT = (        # default system instruction for the assistant
    "You are a helpful AI assistant. Use the conversation history to answer user questions."
)

# --------------------------------------------------------------------------- #
# prompt formatting
# --------------------------------------------------------------------------- #

def format_prompt(
    history: History,
    next_user_message: str,
    system_prompt: Optional[str] = None,
    max_turns: int = 20
) -> str:
    """
    Build a single prompt string by concatenating a system message,
    recent conversation turns, and the incoming user message.

    Args:
        history: List of past ChatMessage entries.
        next_user_message: The new message from the user.
        system_prompt: Optional top-level system instruction.
        max_turns: Maximum number of past user–assistant pairs to include.

    Returns:
        A formatted prompt string ready to send to the LLM.
    """
    # set default system instruction if none provided
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    parts: List[str] = []  # collect each segment of the final prompt

    # add the system instruction at the top of the prompt
    parts.append(f"{ROLE_SYSTEM}: {system_prompt}")

    # determine which history messages to include
    # * limit history to most recent pairs to control prompt length
    # * max_turns counts user–assistant exchanges
    start_index = max(0, len(history) - max_turns * 2)

    # append each historical message in order
    for msg in history[start_index:]:
        # * format as "role: content" for LLM consumption
        parts.append(f"{msg['role']}: {msg['content']}")

    # add the incoming user message
    parts.append(f"{ROLE_USER}: {next_user_message}")
    # signal that the assistant should reply next
    parts.append(f"{ROLE_ASSISTANT}:")

    # combine all parts into one string with newline separators
    return "\n".join(parts)
