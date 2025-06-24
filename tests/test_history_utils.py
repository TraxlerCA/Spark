import pytest

from history_utils import (
    format_prompt,
    ROLE_SYSTEM,
    ROLE_USER,
    ROLE_ASSISTANT,
    DEFAULT_SYSTEM_PROMPT,
)


class TestFormatPrompt:
    def test_default_format_with_empty_history(self):
        """
        Test that format_prompt with empty history uses the default system prompt
        and wraps the user message correctly.
        """
        # Arrange
        history = []
        next_user_message = "Hello"
        # Act
        prompt = format_prompt(history, next_user_message)
        # Assert
        expected = (
            f"{ROLE_SYSTEM}: {DEFAULT_SYSTEM_PROMPT}\n"
            f"{ROLE_USER}: <user_question>{next_user_message}</user_question>\n"
            f"{ROLE_ASSISTANT}:"
        )
        assert prompt == expected

    def test_custom_system_prompt(self):
        """
        Test that providing a custom system_prompt overrides the default.
        """
        # Arrange
        history = []
        next_user_message = "Test override"
        custom_system = "Custom system prompt."
        # Act
        prompt = format_prompt(
            history,
            next_user_message,
            system_prompt=custom_system
        )
        # Assert
        assert prompt.startswith(f"{ROLE_SYSTEM}: {custom_system}")

    def test_history_truncation_with_max_turns(self):
        """
        Test that when history is longer than max_turns * 2, only the most
        recent messages are included.
        """
        # Arrange
        history = []
        # create 10 alternating messages
        for i in range(10):
            role = ROLE_USER if i % 2 == 0 else ROLE_ASSISTANT
            history.append({"role": role, "content": f"msg{i}", "timestamp": f"t{i}"})
        next_user_message = "Next"
        # Act
        # max_turns=2 => include only the last 4 messages
        prompt = format_prompt(history, next_user_message, max_turns=2)
        lines = prompt.splitlines()
        # Assert
        # skip system line
        included = history[-4:]
        for idx, msg in enumerate(included, start=1):
            assert lines[idx] == f"{msg['role']}: {msg['content']}"
        # next line after history is the user message
        assert lines[1 + len(included)] == (
            f"{ROLE_USER}: <user_question>{next_user_message}</user_question>"
        )

    @pytest.mark.parametrize("rag_context", [None, "", "   ", "No context found."])
    def test_no_rag_context_branch(self, rag_context):
        """
        Test that blank or default 'No context found.' rag_context values
        do not trigger context injection.
        """
        # Arrange
        history = [{"role": ROLE_USER, "content": "hi", "timestamp": None}]
        next_user_message = "Hello"
        # Act
        prompt = format_prompt(
            history,
            next_user_message,
            rag_context=rag_context
        )
        # Assert
        assert "Use the following context" not in prompt
        # only one user-tagged message should appear (the wrapped one)
        assert prompt.count(f"{ROLE_USER}:") == 1
        assert f"<user_question>{next_user_message}</user_question>" in prompt

    def test_rag_context_injection(self):
        """
        Test that providing a meaningful rag_context injects both the
        instruction and the context block before the user question.
        """
        # Arrange
        history = [{"role": ROLE_USER, "content": "hello", "timestamp": None}]
        context = "This is important context."
        next_user_message = "What is the answer?"
        # Act
        prompt = format_prompt(
            history,
            next_user_message,
            rag_context=context
        )
        # Assert
        assert "Use the following context to answer the user's question." in prompt
        assert f"### Context:\n{context}" in prompt
        assert f"### User Question:\n<user_question>{next_user_message}</user_question>" in prompt
        # history line still present
        assert f"{ROLE_USER}: hello" in prompt
        # ends with assistant prompt
        assert prompt.endswith(f"{ROLE_ASSISTANT}:")

    def test_zero_max_turns_skips_history(self):
        """
        Test that setting max_turns to zero omits all history messages.
        """
        # Arrange
        history = [{"role": ROLE_USER, "content": "old message", "timestamp": None}]
        next_user_message = "New message"
        # Act
        prompt = format_prompt(
            history,
            next_user_message,
            max_turns=0
        )
        # Assert
        assert "old message" not in prompt
        assert f"<user_question>{next_user_message}</user_question>" in prompt

    def test_safe_user_message_wrapping(self):
        """
        Test that special characters in next_user_message are safely wrapped
        without alteration.
        """
        # Arrange
        history = []
        special_msg = "Hello <World> & other"
        # Act
        prompt = format_prompt(history, special_msg)
        # Assert
        assert f"<user_question>{special_msg}</user_question>" in prompt
