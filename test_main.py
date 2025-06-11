
"""
Unit tests for the ask_llm function from the main module.
This script uses Python's built-in unittest framework and follows best practices
for clarity and maintainability, making it accessible for beginners.
"""

import unittest
from main import ask_llm


class AskLLMTests(unittest.TestCase):
    """Test suite for the ask_llm function."""

    def test_ask_llm_returns_string(self):
        """
        Test that ask_llm returns a non-empty string when given a simple prompt.

        Steps:
        1. Define a simple prompt.
        2. Call ask_llm with the prompt.
        3. Assert that the response is a string.
        4. Assert that the string is not empty or just whitespace.
        """
        prompt = "Say hello!"
        response = ask_llm(prompt)

        # Check that the response is of type str
        self.assertIsInstance(response, str, "Response should be a string.")

        # Check that the response is not empty or only whitespace
        self.assertTrue(response.strip(), "Response should not be empty or whitespace.")

    def test_ask_llm_handles_empty_prompt(self):
        """
        Test that ask_llm handles an empty prompt gracefully.

        Even if the prompt is empty, the function should still return a valid string response.
        """
        prompt = ""
        response = ask_llm(prompt)

        # We still expect a string response, even for empty prompts
        self.assertIsInstance(response, str, "Response should be a string even for an empty prompt.")
        self.assertTrue(len(response) > 0, "Response should not be empty.")


if __name__ == "__main__":
    # Run the tests with increased verbosity to see detailed output
    unittest.main(verbosity=2)
