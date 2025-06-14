"""
Unit tests for the Ollama single-turn client (main.py).

This test suite verifies the functionality of the Ollama client, including
configuration management, request generation, response handling, and error
conditions. The tests use mocking to isolate the client from the network
and the actual Ollama API.
"""
from __future__ import annotations

import json
import logging
import os
import unittest
from unittest.mock import MagicMock, mock_open, patch

import requests

# Temporarily add the script's directory to the path to allow direct import
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# pylint: disable=wrong-import-position
from main import (
    MAX_PROMPT_LENGTH,
    OllamaClientError,
    OllamaConnectionError,
    OllamaResponseError,
    OllamaSettings,
    OllamaTimeoutError,
    ask_llm,
    stream_llm_response,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

TEST_PROMPT = "Why is the sky blue?"
DEFAULT_HOST = "http://localhost"
DEFAULT_PORT = 11434
DEFAULT_MODEL = "gemma3:4b"
DEFAULT_URL = f"{DEFAULT_HOST}:{DEFAULT_PORT}/api/generate"


# --------------------------------------------------------------------------- #
# Test Cases
# --------------------------------------------------------------------------- #


class TestOllamaSettings(unittest.TestCase):
    """Tests the OllamaSettings configuration class."""

    def test_default_settings(self) -> None:
        """
        Tests that default settings are loaded correctly when no environment
        variables are set.
        """
        settings = OllamaSettings()
        self.assertEqual(settings.host, DEFAULT_HOST)
        self.assertEqual(settings.port, DEFAULT_PORT)
        self.assertEqual(settings.model, DEFAULT_MODEL)
        self.assertEqual(settings.timeout, 60)

    @patch.dict(
        os.environ,
        {
            "OLLAMA_HOST": "http://testhost",
            "OLLAMA_PORT": "9999",
            "OLLAMA_MODEL": "test-model:latest",
            "OLLAMA_TIMEOUT": "120",
        },
    )
    def test_settings_from_environment_variables(self) -> None:
        """
        Tests that settings are correctly overridden by environment variables.
        """
        settings = OllamaSettings()
        self.assertEqual(settings.host, "http://testhost")
        self.assertEqual(settings.port, 9999)
        self.assertEqual(settings.model, "test-model:latest")
        self.assertEqual(settings.timeout, 120)


class TestStreamLlmResponse(unittest.TestCase):
    """Tests the stream_llm_response generator function."""

    def setUp(self) -> None:
        """Initializes common test settings."""
        self.settings = OllamaSettings()
        # Suppress logging to keep test output clean
        logging.disable(logging.CRITICAL)

    def tearDown(self) -> None:
        """Re-enables logging after tests."""
        logging.disable(logging.NOTSET)

    @patch("main.requests.post")
    def test_successful_stream(self, mock_post: MagicMock) -> None:
        """
        Tests a successful API call that streams back a valid response.
        """
        # --- Arrange ---
        # Mock the streaming response from requests
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        # Simulate the line-by-line streaming from the API
        mock_response_lines = [
            json.dumps({"response": "The sky is blue ", "done": False}),
            json.dumps({"response": "due to Rayleigh scattering.", "done": False}),
            json.dumps({"response": "", "done": True}),
        ]
        mock_response.iter_lines.return_value = mock_response_lines
        mock_post.return_value.__enter__.return_value = mock_response

        # --- Act ---
        response_generator = stream_llm_response(TEST_PROMPT, self.settings)
        response_parts = list(response_generator)

        # --- Assert ---
        # Ensure the response is correctly assembled
        self.assertEqual(len(response_parts), 2)
        self.assertEqual(response_parts[0], "The sky is blue ")
        self.assertEqual(response_parts[1], "due to Rayleigh scattering.")

        # Ensure requests.post was called correctly
        expected_payload = {
            "model": self.settings.model,
            "prompt": TEST_PROMPT,
            "stream": True,
        }
        mock_post.assert_called_once_with(
            DEFAULT_URL,
            json=expected_payload,
            stream=True,
            timeout=self.settings.timeout,
        )

    def test_prompt_validation(self) -> None:
        """
        Tests that the function raises ValueError for invalid prompts.
        """
        with self.assertRaisesRegex(ValueError, "Prompt must be a non-empty string."):
            list(stream_llm_response("", self.settings))

        with self.assertRaisesRegex(ValueError, "Prompt must be a non-empty string."):
            list(stream_llm_response("   ", self.settings))

        long_prompt = "a" * (MAX_PROMPT_LENGTH + 1)
        with self.assertRaisesRegex(ValueError, "Prompt exceeds maximum length"):
            list(stream_llm_response(long_prompt, self.settings))

    @patch("main.requests.post", side_effect=requests.exceptions.ConnectionError)
    def test_connection_error(self, mock_post: MagicMock) -> None:
        """
        Tests that OllamaConnectionError is raised on connection failure.
        """
        with self.assertRaises(OllamaConnectionError):
            list(stream_llm_response(TEST_PROMPT, self.settings))
        mock_post.assert_called_once()

    @patch("main.requests.post", side_effect=requests.exceptions.Timeout)
    def test_timeout_error(self, mock_post: MagicMock) -> None:
        """
        Tests that OllamaTimeoutError is raised on a request timeout.
        """
        with self.assertRaises(OllamaTimeoutError):
            list(stream_llm_response(TEST_PROMPT, self.settings))
        mock_post.assert_called_once()

    @patch("main.requests.post")
    def test_http_error(self, mock_post: MagicMock) -> None:
        """
        Tests that OllamaResponseError is raised on a non-2xx HTTP status.
        """
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=MagicMock(status_code=404)
        )
        mock_post.return_value.__enter__.return_value = mock_response

        with self.assertRaisesRegex(OllamaResponseError, "Ollama returned HTTP 404"):
            list(stream_llm_response(TEST_PROMPT, self.settings))
        mock_post.assert_called_once()

    @patch("main.requests.post", side_effect=requests.exceptions.RequestException)
    def test_generic_request_exception(self, mock_post: MagicMock) -> None:
        """
        Tests that a generic RequestException is caught and wrapped.
        """
        with self.assertRaises(OllamaClientError):
            list(stream_llm_response(TEST_PROMPT, self.settings))
        mock_post.assert_called_once()

    @patch("main.requests.post")
    def test_api_error_in_stream(self, mock_post: MagicMock) -> None:
        """
        Tests that an error message from the Ollama API itself is handled.
        """
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        # Simulate the API returning an error after starting the stream
        error_message = "the model 'mistake' is not found"
        mock_response_lines = [json.dumps({"error": error_message, "done": True})]
        mock_response.iter_lines.return_value = mock_response_lines
        mock_post.return_value.__enter__.return_value = mock_response

        with self.assertRaisesRegex(OllamaResponseError, error_message):
            list(stream_llm_response(TEST_PROMPT, self.settings))
        mock_post.assert_called_once()


class TestAskLlm(unittest.TestCase):
    """Tests the ask_llm function which aggregates the stream."""

    @patch("main.stream_llm_response")
    def test_successful_aggregation(self, mock_stream: MagicMock) -> None:
        """
        Tests that the streamed response is correctly joined into a single string.
        """
        # --- Arrange ---
        # Mock the generator to yield response parts
        mock_stream.return_value = iter(
            ["This ", "is ", "a ", "complete ", "sentence."]
        )
        settings = OllamaSettings()

        # --- Act ---
        full_response = ask_llm(TEST_PROMPT, settings)

        # --- Assert ---
        self.assertEqual(full_response, "This is a complete sentence.")
        mock_stream.assert_called_once_with(TEST_PROMPT, settings)

    @patch("main.stream_llm_response")
    @patch("main.OllamaSettings")
    def test_default_settings_initialization(
        self, mock_settings: MagicMock, mock_stream: MagicMock
    ) -> None:
        """
        Tests that OllamaSettings are initialized if none are provided.
        """
        mock_stream.return_value = iter([])  # Not testing the output here
        ask_llm(TEST_PROMPT)  # Call without settings argument
        mock_settings.assert_called_once()
        # Verify that the created settings instance was passed to the stream
        self.assertIsInstance(mock_stream.call_args[0][1], MagicMock)


class TestMainExecution(unittest.TestCase):
    """Tests the main entry point function."""

    def setUp(self) -> None:
        """Suppresses logging to keep test output clean."""
        logging.disable(logging.CRITICAL)

    def tearDown(self) -> None:
        """Re-enables logging after tests."""
        logging.disable(logging.NOTSET)

    @patch("main.input", return_value=TEST_PROMPT)
    @patch("main.stream_llm_response")
    @patch("builtins.print")
    def test_main_success(
        self, mock_print: MagicMock, mock_stream: MagicMock, mock_input: MagicMock
    ) -> None:
        """
        Tests the main function's successful execution path.
        """
        # --- Arrange ---
        from main import main as main_func

        mock_stream.return_value = iter(["Hello", " World"])

        # --- Act ---
        main_func()

        # --- Assert ---
        mock_input.assert_called_once_with("Ask the LLM a question: ")
        mock_stream.assert_called_once()
        # Check that print was called to stream the output
        self.assertIn(unittest.mock.call("Hello", end="", flush=True), mock_print.call_args_list)
        self.assertIn(unittest.mock.call(" World", end="", flush=True), mock_print.call_args_list)

    @patch("main.input", return_value=TEST_PROMPT)
    @patch("main.stream_llm_response", side_effect=OllamaConnectionError("Test error"))
    @patch("sys.exit")
    @patch("main.logger.error")
    def test_main_client_error(
        self,
        mock_logger_error: MagicMock,
        mock_exit: MagicMock,
        mock_stream: MagicMock,
        mock_input: MagicMock,
    ) -> None:
        """
        Tests that the main function catches OllamaClientError and exits.
        """
        from main import main as main_func

        main_func()

        mock_stream.assert_called_once()
        # Check that the error was logged
        mock_logger_error.assert_called_once_with(
            "A client-side error occurred.", extra={"error": "Test error"}
        )
        # Check that the program exited with an error code
        mock_exit.assert_called_once_with(1)

    @patch("main.input", side_effect=KeyboardInterrupt)
    @patch("builtins.print")
    def test_main_keyboard_interrupt(
        self, mock_print: MagicMock, mock_input: MagicMock
    ) -> None:
        """
        Tests the main function's handling of KeyboardInterrupt.
        """
        from main import main as main_func

        main_func()
        mock_print.assert_any_call("\nBye!")


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)