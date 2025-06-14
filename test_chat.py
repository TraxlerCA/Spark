"""
Unit tests for the multi-turn chat client (chat.py).

This test suite verifies the functionality of the ChatApplication, including
state management (history), command handling, and interaction with the
underlying LLM client function. The `ask_llm` function is mocked to
isolate the chat logic from the client and network layers.
"""
from __future__ import annotations

import logging
import os
import sys
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

# In order to import from main.py and chat.py, we need to add their
# directory to the Python path. We also create a dummy main module
# to satisfy the import dependency in chat.py
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.modules["main"] = MagicMock()

# pylint: disable=wrong-import-position,-reimported
from chat import (
    COMMANDS,
    ROLE_ASSISTANT,
    ROLE_USER,
    ChatApplication,
    handle_clear,
    handle_exit,
    handle_help,
    handle_history,
)


# --------------------------------------------------------------------------- #
# Test Cases
# --------------------------------------------------------------------------- #


class TestCommandHandlers(unittest.TestCase):
    """Tests the standalone command handler functions."""

    def setUp(self) -> None:
        """Redirect stdout to capture print statements."""
        self.held_stdout = sys.stdout
        sys.stdout = StringIO()

    def tearDown(self) -> None:
        """Restore stdout."""
        sys.stdout = self.held_stdout

    def test_handle_exit(self) -> None:
        """Tests that handle_exit attempts to exit the program."""
        with self.assertRaises(SystemExit) as cm:
            handle_exit([])
        self.assertEqual(cm.exception.code, 0)
        self.assertIn("Goodbye!", sys.stdout.getvalue())

    def test_handle_help(self) -> None:
        """Tests that handle_help displays all commands."""
        handle_help([])
        output = sys.stdout.getvalue()
        # Check that every command is present in the help text
        for cmd, (_, description) in COMMANDS.items():
            self.assertIn(cmd, output)
            self.assertIn(description, output)

    def test_handle_history_empty(self) -> None:
        """Tests handle_history with no messages."""
        handle_history([])
        self.assertIn("No messages in history", sys.stdout.getvalue())

    def test_handle_history_with_messages(self) -> None:
        """Tests handle_history with a populated history."""
        history = [
            {"role": ROLE_USER, "content": "Hello"},
            {"role": ROLE_ASSISTANT, "content": "Hi there!"},
        ]
        handle_history(history)
        output = sys.stdout.getvalue()
        self.assertIn("You: Hello", output)
        self.assertIn("Assistant: Hi there!", output)

    def test_handle_clear(self) -> None:
        """Tests that handle_clear empties the history list."""
        history = [
            {"role": ROLE_USER, "content": "This will be cleared."},
        ]
        handle_clear(history)
        self.assertEqual(len(history), 0)
        self.assertIn("Chat history has been cleared", sys.stdout.getvalue())


class TestChatApplication(unittest.TestCase):
    """Tests the ChatApplication class and its methods."""

    def setUp(self) -> None:
        """Initializes a ChatApplication instance for each test."""
        # Mock the dependency from main.py
        self.ask_llm_patcher = patch("chat.ask_llm")
        self.mock_ask_llm = self.ask_llm_patcher.start()
        self.addCleanup(self.ask_llm_patcher.stop)

        self.app = ChatApplication()
        # Suppress logging to keep test output clean
        logging.disable(logging.CRITICAL)

    def tearDown(self) -> None:
        """Re-enables logging after tests."""
        logging.disable(logging.NOTSET)

    def test_initialization(self) -> None:
        """Tests the initial state of the application."""
        self.assertEqual(self.app.history, [])

    def test_process_message_success(self) -> None:
        """
        Tests processing a user message that gets a successful reply.
        """
        # --- Arrange ---
        user_message = "What is Python?"
        assistant_reply = "An amazing programming language."
        self.mock_ask_llm.return_value = assistant_reply

        # --- Act ---
        # We redirect stdout to suppress the "Assistant: " print output
        with patch("sys.stdout", new_callable=StringIO):
            self.app.process_message(user_message)

        # --- Assert ---
        # Check that history was updated correctly
        self.assertEqual(len(self.app.history), 2)
        self.assertEqual(self.app.history[0]["role"], ROLE_USER)
        self.assertEqual(self.app.history[0]["content"], user_message)
        self.assertEqual(self.app.history[1]["role"], ROLE_ASSISTANT)
        self.assertEqual(self.app.history[1]["content"], assistant_reply)

        # Check that the LLM was called correctly
        self.mock_ask_llm.assert_called_once_with(user_message)

    def test_process_message_llm_error(self) -> None:
        """
        Tests how a client error from the LLM is handled during a chat.
        """
        # --- Arrange ---
        user_message = "This will fail."
        # The main.py module is mocked, so we need to mock the exception too.
        mock_ollama_error = type("OllamaClientError", (Exception,), {})
        sys.modules["main"].OllamaClientError = mock_ollama_error
        self.mock_ask_llm.side_effect = mock_ollama_error("Connection failed")

        # --- Act ---
        # Redirect stdout to capture the error message printed to the user
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            self.app.process_message(user_message)
            output = mock_stdout.getvalue()

        # --- Assert ---
        # History should be empty because the user's turn is popped on failure
        self.assertEqual(len(self.app.history), 0)
        self.mock_ask_llm.assert_called_once_with(user_message)

        # The user should see a system error message
        self.assertIn("[SYSTEM] Error:", output)
        self.assertIn("Connection failed", output)

    @patch("chat.input")
    def test_run_loop_command_dispatch(self, mock_input: MagicMock) -> None:
        """
        Tests that the main run loop correctly dispatches a command.
        """
        # Simulate user typing a command and then raising an exception to exit the loop
        mock_input.side_effect = [":history", KeyboardInterrupt]
        # Mock the handler to verify it was called
        with patch("chat.handle_history") as mock_handler:
            # We expect a SystemExit or KeyboardInterrupt to be caught
            with self.assertRaises((SystemExit, KeyboardInterrupt)):
                self.app.run()
            # Assert that the correct command handler was called with the history
            mock_handler.assert_called_once_with(self.app.history)

    @patch("chat.input")
    def test_run_loop_message_processing(self, mock_input: MagicMock) -> None:
        """
        Tests that the main run loop correctly calls process_message for user input.
        """
        user_message = "A regular message"
        mock_input.side_effect = [user_message, KeyboardInterrupt]

        with patch.object(
            self.app, "process_message"
        ) as mock_process_message:
            with self.assertRaises((SystemExit, KeyboardInterrupt)):
                self.app.run()
            # Assert that process_message was called with the user's text
            mock_process_message.assert_called_once_with(user_message)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)