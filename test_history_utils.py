# test_history_utils.py

import unittest
from history_utils import (
    format_prompt,
    DEFAULT_SYSTEM_PROMPT,
    ROLE_SYSTEM,
    ROLE_USER,
    ROLE_ASSISTANT,
)


class TestFormatPrompt(unittest.TestCase):
    """Tests for the format_prompt utility in history_utils.py."""

    def test_empty_history_default_system(self):
        """When history is empty, prompt should include only system, user, and assistant tags."""
        prompt = format_prompt([], "Hello")
        # format_prompt uses literal "\n" as separator, so split on that
        lines = prompt.split("\\n")
        self.assertEqual(lines[0], f"{ROLE_SYSTEM}: {DEFAULT_SYSTEM_PROMPT}")
        self.assertEqual(lines[1], f"{ROLE_USER}: Hello")
        self.assertEqual(lines[2], f"{ROLE_ASSISTANT}:")

    def test_custom_system_and_truncation(self):
        """
        When providing a custom system prompt and a long history,
        only the last max_turns * 2 entries should be included.
        """
        # build a history of 50 alternating messages
        hist = [
            {"role": ROLE_USER if i % 2 == 0 else ROLE_ASSISTANT, "content": f"msg{i}"}
            for i in range(50)
        ]
        # use max_turns=5 → include only last 10 history entries
        prompt = format_prompt(hist, "New", system_prompt="SYS", max_turns=5)
        # history is joined with literal "\n", so split on that
        lines = prompt.split("\\n")

        # first line is our custom system prompt
        self.assertEqual(lines[0], "system: SYS")
        # total lines = 1 system + 10 history + 1 new user + 1 assistant tag
        self.assertEqual(len(lines), 1 + 10 + 2)

        # history should start at index 40 → msg40 is a user turn
        self.assertEqual(lines[1], f"{ROLE_USER}: msg40")

        # final two lines must be the new user and assistant prompt
        self.assertEqual(lines[-2], f"{ROLE_USER}: New")
        self.assertEqual(lines[-1], f"{ROLE_ASSISTANT}:")


if __name__ == "__main__":
    unittest.main()
