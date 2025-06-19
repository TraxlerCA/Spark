# chat.py
#
# Description: a multi-turn chat client for a local Ollama LLM.

"""
This script provides a simple, stateful command-line chat interface.

It manages a conversation history in memory and allows special commands
for interacting with the chat session (e.g., :help, :history, :exit).
It uses the refactored `ask_llm` function for model interaction.
"""

# --------------------------------------------------------------------------- #
# imports
# --------------------------------------------------------------------------- #
from __future__ import annotations  # allow postponed evaluation of annotations
import logging                       # for logging messages
import sys                           # for system exit and module introspection
from typing import Callable, Dict, List, TypedDict  # for type definitions

try:
    from main import OllamaClientError, ask_llm  # import LLM client error and API call
except ImportError:
    print("Error: Could not import from 'main.py'. Make sure it's in the same directory.")  # notify missing dependency
    sys.exit(1)  # exit if import fails

# --------------------------------------------------------------------------- #
# constants and type definitions
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)  # module-level logger

ROLE_USER = "user"            # identifier for user role in messages
ROLE_ASSISTANT = "assistant"  # identifier for assistant role in messages

class ChatMessage(TypedDict):
    """A structured dictionary representing one turn in a conversation."""
    role: str     # typically ROLE_USER or ROLE_ASSISTANT
    content: str  # the text content of the message

History = List[ChatMessage]               # alias for conversation history type
CommandHandler = Callable[[History], None]  # alias for command handler signature

# --------------------------------------------------------------------------- #
# command handlers
# --------------------------------------------------------------------------- #
def handle_exit(history: History) -> None:
    """Exit the chat application immediately.
    
    Args:
        history: the conversation history (unused).
    """
    print("Goodbye!")  # inform user
    sys.exit(0)        # terminate program

def handle_help(history: History) -> None:
    """Display available commands and their descriptions."""
    print("Available commands:")
    for cmd, (_, description) in COMMANDS.items():
        print(f"  {cmd:<10} - {description}")  # list each command

def handle_history(history: History) -> None:
    """Show the conversation history, if any."""
    if not history:
        print("No messages in history yet.")  # nothing to show
        return

    print("\n--- Chat History ---")
    for turn in history:
        speaker = "You" if turn["role"] == ROLE_USER else "Assistant"
        print(f"{speaker}: {turn['content']}")  # display each turn
    print("--- End History ---\n")

def handle_clear(history: History) -> None:
    """Clear the current chat history."""
    history.clear()  # remove all entries
    print("Chat history has been cleared.")  # confirm action

# --------------------------------------------------------------------------- #
# command routing table
# --------------------------------------------------------------------------- #
COMMANDS: Dict[str, tuple[CommandHandler, str]] = {
    ":exit":    (handle_exit,    "Exits the chat application."),
    ":help":    (handle_help,    "Displays this help message."),
    ":history": (handle_history, "Displays the conversation history."),
    ":clear":   (handle_clear,   "Clears the current chat history."),
}

# --------------------------------------------------------------------------- #
# main chat application class
# --------------------------------------------------------------------------- #
class ChatApplication:
    """Encapsulates the state and logic of the chat loop."""

    def __init__(self) -> None:
        """Initialize with an empty history."""
        self.history: History = []                      # store message sequence
        logger.info("Chat application initialized.")     # log startup

    def run(self) -> None:
        """Start and manage the main chat loop."""
        print("Welcome to Chat! Type ':help' for commands, or ':exit' to quit.\n")  # greeting

        while True:
            try:
                user_input = input("You: ").strip()  # get and trim user input
                if not user_input:
                    continue  # skip empty entries

                command = user_input.lower()          # normalize for lookup
                entry = COMMANDS.get(command)         # check for special command
                if entry:
                    handler_fn, _ = entry             # extract handler function
                    # call handler dynamically to allow patching:
                    # - retrieve current function object
                    # - execute with conversation history
                    handler = getattr(sys.modules[__name__], handler_fn.__name__)
                    handler(self.history)
                else:
                    self.process_message(user_input)  # handle regular message

            except (KeyboardInterrupt, EOFError):
                handle_exit(self.history)  # exit cleanly on interrupt

    def process_message(self, text: str) -> None:
        """
        Process a user message: send to LLM, handle response, update history.

        Args:
            text: the user's message content.
        """
        # add user message to history
        self.history.append({"role": ROLE_USER, "content": text})

        try:
            print("Assistant: ", end="", flush=True)  # prepare reply prompt
            reply = ask_llm(text)                     # call the LLM API
            print(reply)                              # display the assistant response

            # record assistant response in history
            self.history.append({"role": ROLE_ASSISTANT, "content": reply})

        except Exception as e:
            # - build error message including exception details
            # - notify user of failure
            # - log full error for debugging
            # - revert last history entry to maintain consistency
            error_message = f"Error: Could not get a response from the LLM. {e}"
            print(f"\n[SYSTEM] {error_message}")  # inform user
            logger.error("LLM error during chat loop.", extra={"error": str(e)})  # log error
            self.history.pop()  # remove failed user entry

# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    """Entry point for the chat application."""
    app = ChatApplication()  # create chat app instance
    app.run()                # start chat loop

if __name__ == "__main__":
    main()  # run when executed as a script
