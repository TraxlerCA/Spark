# chat.py

"""
A simple chat loop that:
 - Accepts user input
 - Routes special commands (e.g. :help, :exit, :history)
 - Sends regular messages to the LLM via ask_llm()
 - Remembers the full conversation in 'history'
"""

import sys
from main import ask_llm  # your existing single-turn function

# ------------------------------------------------------------------------------
# Command handlers
# ------------------------------------------------------------------------------

def handle_exit(history):
    """Exit the program immediately."""
    print("Goodbye!")
    sys.exit(0)


def handle_help(history):
    """List all available commands."""
    print("Available commands:")
    for cmd in COMMANDS:
        print(f"  {cmd}")


def handle_history(history):
    """Display the full chat history with roles."""
    if not history:
        print("No messages yet.")
        return

    print("\nChat history:")
    for turn in history:
        speaker = "You" if turn["role"] == "user" else "Assistant"
        print(f"{speaker}: {turn['content']}")
    print()  # blank line after history


# Map textual commands to their handler functions
COMMANDS = {
    ":exit": handle_exit,
    ":help": handle_help,
    ":history": handle_history,
}


# ------------------------------------------------------------------------------
# Main chat loop
# ------------------------------------------------------------------------------

def main():
    # Keep a list of all turns, each as {'role': 'user'|'llm', 'content': str}
    history = []

    # Welcoming message
    print("Welcome to chat! Type ':help' for a list of commands, ':exit' to quit.\n")

    # Loop forever until the user calls :exit
    while True:
        try:
            # Prompt the user for input
            text = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            # Handle Ctrl+C / Ctrl+D gracefully
            print("\nGoodbye!")
            break

        # If the input matches a command, dispatch it
        if text in COMMANDS:
            COMMANDS[text](history)
            continue

        # Otherwise, treat it as normal chat
        if text:
            # Record user turn
            history.append({"role": "user", "content": text})

            # Send to your LLM function and catch any errors
            try:
                reply = ask_llm(text)
            except Exception as e:
                # If something goes wrong with the LLM call, show an error
                print(f"[Error] failed to get response: {e}")
                continue

            # Record and display the assistant's reply
            history.append({"role": "llm", "content": reply})
            print(f"Assistant: {reply}\n")
        else:
            # Ignore empty input
            continue


# Only run main() if this script is executed directly
if __name__ == "__main__":
    main()
