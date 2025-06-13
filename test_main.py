"""
test_main.py – basic tests for main.py.

We mock the network call so that tests run instantly and do not depend
on a live Ollama server.  Beginners can still run the real system test
by commenting out the @patch line if they like.
"""

import unittest
from unittest.mock import patch

import main


class AskLLMTests(unittest.TestCase):
    """Verify that ask_llm behaves and returns a string."""

    @patch("main.requests.Session.post")
    def test_ask_llm_returns_string(self, mock_post):
        # Arrange: craft a fake streaming response object
        class FakeResponse:
            def __init__(self):
                self.status_code = 200

            def raise_for_status(self):
                pass  # always OK

            # yield a couple of JSON chunks, then an empty line
            def iter_lines(self, decode_unicode=True):
                yield '{"response": "Hello"}'
                yield '{"response": ", world!"}'
                yield ""

        mock_post.return_value = FakeResponse()

        # Act
        reply = main.ask_llm("Say hello!")

        # Assert
        self.assertIsInstance(reply, str)
        self.assertEqual(reply, "Hello, world!")

    def test_rejects_empty_prompt(self):
        self.assertIn("Error:", main.ask_llm(""))


if __name__ == "__main__":
    unittest.main()
