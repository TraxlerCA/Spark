#
# chat.py
#
# Description: A multi-turn chat client for a local Ollama LLM.
#
"""
This script provides a simple, stateful command-line chat interface.

It manages a conversation history in memory and allows special commands
for interacting with the chat session (e.g., :help, :history, :exit).
It uses the refactored `ask_llm` function for model interaction.
"""

from __future__ import annotations

import logging
import sys
from typing import Callable, Dict, List, TypedDict

try:
    from main import OllamaClientError, ask_llm
except ImportError:
    print("Error: Could not import from 'main.py'. Make sure it's in the same directory.")
    sys.exit(1)

# --------------------------------------------------------------------------- #
# Constants and Type Definitions
# --------------------------------------------------------------------------- #

logger = logging.getLogger(__name__)

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"


class ChatMessage(TypedDict):
    """A structured dictionary representing one turn in a conversation."""

    role: str  # Typically ROLE_USER or ROLE_ASSISTANT
    content: str


History = List[ChatMessage]
CommandHandler = Callable[[History], None]

# --------------------------------------------------------------------------- #
# Command Handlers
# --------------------------------------------------------------------------- #


def handle_exit(history: History) -> None:
    """Exits the chat application."""
    print("Goodbye!")
    sys.exit(0)


def handle_help(history: History) -> None:
    """Displays this help message."""
    print("Available commands:")
    for cmd, (_, description) in COMMANDS.items():
        print(f"  {cmd:<10} - {description}")


def handle_history(history: History) -> None:
    """Displays the conversation history."""
    if not history:
        print("No messages in history yet.")
        return

    print("\n--- Chat History ---")
    for turn in history:
        speaker = "You" if turn["role"] == ROLE_USER else "Assistant"
        print(f"{speaker}: {turn['content']}")
    print("--- End History ---\n")


def handle_clear(history: History) -> None:
    """Clears the current chat history."""
    history.clear()
    print("Chat history has been cleared.")


# --- Command Routing Table ---
COMMANDS: Dict[str, tuple[CommandHandler, str]] = {
    ":exit": (handle_exit, "Exits the chat application."),
    ":help": (handle_help, "Displays this help message."),
    ":history": (handle_history, "Displays the conversation history."),
    ":clear": (handle_clear, "Clears the current chat history."),
}


# --------------------------------------------------------------------------- #
# Main Chat Application
# --------------------------------------------------------------------------- #


class ChatApplication:
    """Encapsulates the state and logic of the chat loop."""

    def __init__(self) -> None:
        """Initializes the chat application with an empty history."""
        self.history: History = []
        logger.info("Chat application initialized.")

    def run(self) -> None:
        """Starts and manages the main chat loop."""
        print("Welcome to Chat! Type ':help' for commands, or ':exit' to quit.\n")

        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue

                command = user_input.lower()
                entry = COMMANDS.get(command)
                if entry:
                    handler_fn, _ = entry
                    # call the current handler function to respect patches
                    handler = getattr(sys.modules[__name__], handler_fn.__name__)
                    handler(self.history)
                else:
                    self.process_message(user_input)

            except (KeyboardInterrupt, EOFError):
                handle_exit(self.history)

    def process_message(self, text: str) -> None:
        """
        Processes a regular user message, gets a reply, and updates history.

        Args:
            text: The user's message content.
        """
        # Add user message to history
        self.history.append({"role": ROLE_USER, "content": text})

        try:
            print("Assistant: ", end="", flush=True)
            reply = ask_llm(text)
            print(reply)

            # Add assistant response to history
            self.history.append({"role": ROLE_ASSISTANT, "content": reply})

        except Exception as e:
            error_message = f"Error: Could not get a response from the LLM. {e}"
            print(f"\n[SYSTEM] {error_message}")
            logger.error("LLM error during chat loop.", extra={"error": str(e)})
            # Remove the user's last message if the call failed
            self.history.pop()


def main() -> None:
    """Entry point for the chat application."""
    app = ChatApplication()
    app.run()


if __name__ == "__main__":
    main()