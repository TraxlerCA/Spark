# chat_history.py
#
# Description: A multi-turn chat client with history-aware prompts for a local Ollama LLM.
#
"""
This script provides a stateful command-line chat interface that threads
conversation history into each prompt sent to the LLM. It supports special
commands (e.g., :help, :history, :clear) and handles errors gracefully.
"""

from __future__ import annotations

import logging
import sys
from typing import Callable, Dict, List, TypedDict

# import main after setting up imports so its dictConfig runs here
from main import OllamaClientError, ask_llm  
from history_utils import (
    format_prompt,
    History,
    ChatMessage,
    ROLE_USER,
    ROLE_ASSISTANT,
)

# --------------------------------------------------------------------------- #
# logger setup
# --------------------------------------------------------------------------- #

# get this module's logger (when run as script, __name__ == "__main__")
logger = logging.getLogger(__name__)
# suppress info logs from this module
logger.setLevel(logging.WARNING)

# --------------------------------------------------------------------------- #
# Command Handlers
# --------------------------------------------------------------------------- #

CommandHandler = Callable[[History], None]


def handle_exit(history: History) -> None:
    """Exit the chat application."""
    print("Goodbye!")
    sys.exit(0)


def handle_help(history: History) -> None:
    """Display available commands."""
    print("Available commands:")
    for cmd, (_, description) in COMMANDS.items():
        print(f"  {cmd:<10} - {description}")


def handle_history(history: History) -> None:
    """Print the full conversation history."""
    if not history:
        print("No messages in history yet.")
        return

    print("\n--- Chat History ---")
    for turn in history:
        speaker = "You" if turn["role"] == ROLE_USER else "Assistant"
        print(f"{speaker}: {turn['content']}")
    print("--- End History ---\n")


def handle_clear(history: History) -> None:
    """Clear the current chat history."""
    history.clear()
    print("Chat history has been cleared.")


# routing table for special “:” commands
COMMANDS: Dict[str, tuple[CommandHandler, str]] = {
    ":exit":    (handle_exit,    "Exit the chat"),
    ":help":    (handle_help,    "Show this help message"),
    ":history": (handle_history, "Display conversation history"),
    ":clear":   (handle_clear,   "Clear all messages"),
}


# --------------------------------------------------------------------------- #
# Main Chat Application
# --------------------------------------------------------------------------- #


class ChatApplication:
    """Encapsulates the state and logic of the multi-turn chat loop."""

    def __init__(self) -> None:
        """Initialize with an empty history."""
        self.history: History = []
        logger.info("chat_history initialized")

    def run(self) -> None:
        """Start the REPL loop, handle commands or pass messages to the LLM."""
        print("Welcome to Chat! Type ':help' for commands, or ':exit' to quit.\n")

        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue

                cmd = user_input.lower()
                if cmd in COMMANDS:
                    handler, _ = COMMANDS[cmd]
                    handler(self.history)
                else:
                    self.process_message(user_input)

            except (KeyboardInterrupt, EOFError):
                handle_exit(self.history)

    def process_message(self, text: str) -> None:
        """
        Add the user message to history, format a context-aware prompt,
        call the LLM, and record the assistant’s reply.
        """
        # record user turn
        self.history.append({"role": ROLE_USER, "content": text})

        try:
            # build the combined prompt
            prompt = format_prompt(self.history, text)
            print("Assistant: ", end="", flush=True)

            # send to LLM
            reply = ask_llm(prompt)
            print(reply)

            # record assistant turn
            self.history.append({"role": ROLE_ASSISTANT, "content": reply})

        except Exception as e:
            # on failure, remove the last user turn
            self.history.pop()
            err = f"Error: could not get response from LLM. {e}"
            print(f"\n[SYSTEM] {err}")
            logger.error("LLM call failed", extra={"error": str(e)})


def main() -> None:
    """Entry point for the chat application."""
    app = ChatApplication()
    app.run()


if __name__ == "__main__":
    main()
