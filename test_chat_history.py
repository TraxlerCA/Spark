# test_chat_history.py

import unittest
from unittest.mock import patch
import logging
import sys
import os

# allow importing from the same directory
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from chat_history import (
    handle_clear,
    handle_history,
    ChatApplication,
    ROLE_USER,
    ROLE_ASSISTANT,
)


class TestCommandHandlers(unittest.TestCase):
    """Unit tests for the command handler functions in chat_history.py."""

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_handle_clear(self):
        """Clearing history should empty the list and print a confirmation."""
        history = [{"role": ROLE_USER, "content": "hi"}]
        with patch("builtins.print") as mock_print:
            handle_clear(history)
            self.assertEqual(history, [])
            mock_print.assert_called_once_with("Chat history has been cleared.")

    def test_handle_history_empty(self):
        """Printing history when empty should notify the user."""
        history = []
        with patch("builtins.print") as mock_print:
            handle_history(history)
            mock_print.assert_called_once_with("No messages in history yet.")

    def test_handle_history_nonempty(self):
        """Printing nonempty history should list all turns in order."""
        history = [
            {"role": ROLE_USER, "content": "hi"},
            {"role": ROLE_ASSISTANT, "content": "hello"},
        ]
        with patch("builtins.print") as mock_print:
            handle_history(history)

            expected_calls = [
                unittest.mock.call("\n--- Chat History ---"),
                unittest.mock.call("You: hi"),
                unittest.mock.call("Assistant: hello"),
                unittest.mock.call("--- End History ---\n"),
            ]
            mock_print.assert_has_calls(expected_calls, any_order=False)


class TestChatApplication(unittest.TestCase):
    """Tests for ChatApplication.process_message in chat_history.py."""

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    @patch("chat_history.ask_llm", return_value="reply")
    def test_process_message_success(self, mock_ask_llm):
        """
        Calling process_message should append both user and assistant messages
        to history and print the assistant’s reply.
        """
        app = ChatApplication()
        with patch("builtins.print") as mock_print:
            app.process_message("hello")

            # history should now have two entries
            self.assertEqual(len(app.history), 2)
            self.assertEqual(app.history[0], {"role": ROLE_USER, "content": "hello"})
            self.assertEqual(app.history[1], {"role": ROLE_ASSISTANT, "content": "reply"})

            # verify printing of the assistant prompt and reply
            mock_print.assert_any_call("Assistant: ", end="", flush=True)
            mock_print.assert_any_call("reply")


if __name__ == "__main__":
    unittest.main()
