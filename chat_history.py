# chat_history.py
#
# Description: A multi-turn chat client with history-aware prompts for a local Ollama LLM.
# This script provides a stateful command-line chat interface that threads
# conversation history into each prompt sent to the LLM. It supports special
# commands (e.g., :help, :history, :clear) and handles errors gracefully.

from __future__ import annotations  # allow postponed evaluation of annotations

import logging  # structured logging
import sys  # for system exit and input/output operations

from typing import Callable, Dict, List, TypedDict  # type hints for handlers and history entries

# import main after setting up imports so its dictConfig runs here
from main import OllamaClientError, ask_llm  # Ollama client error and function to query LLM
from history_utils import (  # utilities for prompt formatting and history management
    format_prompt,
    History,
    ChatMessage,
    ROLE_USER,
    ROLE_ASSISTANT,
)

# --------------------------------------------------------------------------- #
# logger setup
# --------------------------------------------------------------------------- #

logger = logging.getLogger(__name__)  # get this module's logger (when run as script, __name__ == "__main__")
logger.setLevel(logging.WARNING)  # suppress info logs from this module

# --------------------------------------------------------------------------- #
# command handlers
# --------------------------------------------------------------------------- #

CommandHandler = Callable[[History], None]  # type alias for command handler functions

def handle_exit(history: History) -> None:
    """exit the chat application.

    Args:
        history (History): current conversation history.
    Returns:
        None.
    """
    print("Goodbye!")  # inform user of exit
    sys.exit(0)  # terminate application

def handle_help(history: History) -> None:
    """display available commands to the user.

    Args:
        history (History): current conversation history.
    Returns:
        None.
    """
    print("Available commands:")
    for cmd, (_, description) in COMMANDS.items():
        print(f"  {cmd:<10} - {description}")  # list each command and its description

def handle_history(history: History) -> None:
    """print the full conversation history.

    Args:
        history (History): current conversation history.
    Returns:
        None.
    """
    if not history:
        print("No messages in history yet.")  # no history to display
        return

    print("\n--- Chat History ---")
    for turn in history:
        speaker = "You" if turn["role"] == ROLE_USER else "Assistant"
        print(f"{speaker}: {turn['content']}")  # show each turn with speaker
    print("--- End History ---\n")

def handle_clear(history: History) -> None:
    """clear the current chat history.

    Args:
        history (History): current conversation history.
    Returns:
        None.
    """
    history.clear()  # remove all history entries
    print("Chat history has been cleared.")  # notify user

# --------------------------------------------------------------------------- #
# command mapping
# --------------------------------------------------------------------------- #

COMMANDS: Dict[str, tuple[CommandHandler, str]] = {
    ":exit":    (handle_exit,    "Exit the chat"),
    ":help":    (handle_help,    "Show this help message"),
    ":history": (handle_history, "Display conversation history"),
    ":clear":   (handle_clear,   "Clear all messages"),
}

# --------------------------------------------------------------------------- #
# main chat application class
# --------------------------------------------------------------------------- #

class ChatApplication:
    """encapsulates the state and logic of the multi-turn chat loop."""

    def __init__(self) -> None:
        """initialize with an empty history."""
        self.history: History = []  # conversation history storage
        logger.info("chat_history initialized")  # log initialization

    def run(self) -> None:
        """start the REPL loop, handle commands or pass messages to the LLM.

        Returns:
            None.
        """
        print("Welcome to Chat! Type ':help' for commands, or ':exit' to quit.\n")  # greet user

        while True:
            try:
                user_input = input("You: ").strip()  # prompt for user input
                if not user_input:
                    continue  # ignore empty input

                cmd = user_input.lower()  # normalize for command lookup
                if cmd in COMMANDS:
                    handler, _ = COMMANDS[cmd]  # find handler
                    handler(self.history)  # execute command
                else:
                    self.process_message(user_input)  # handle as chat message

            except (KeyboardInterrupt, EOFError):
                handle_exit(self.history)  # exit gracefully on interrupt

    def process_message(self, text: str) -> None:
        """
        process a user message by:
          - recording the user message in history
          - building a context-aware prompt
          - sending the prompt to the LLM and displaying the response
          - recording the assistant's reply or handling errors

        Args:
            text (str): user input message.
        Returns:
            None.
        """
        # record user turn
        self.history.append({"role": ROLE_USER, "content": text})

        try:
            # - build the combined prompt to include full history and latest input
            prompt = format_prompt(self.history, text)
            # - display assistant prefix and ensure immediate output
            print("Assistant: ", end="", flush=True)

            # - send prompt to LLM and capture reply
            reply = ask_llm(prompt)
            print(reply)  # display assistant response

            # - record assistant turn in history
            self.history.append({"role": ROLE_ASSISTANT, "content": reply})

        except Exception as e:
            # - rollback state by removing last user turn on failure
            self.history.pop()
            err = f"Error: could not get response from LLM. {e}"
            print(f"\n[SYSTEM] {err}")  # show error to user
            logger.error("LLM call failed", extra={"error": str(e)})  # log error details

# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    """entry point for the chat application.

    Returns:
        None.
    """
    app = ChatApplication()  # instantiate application
    app.run()  # start event loop

if __name__ == "__main__":
    main()  # run application when executed as script
