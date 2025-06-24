# test_script.py

import pytest
from types import SimpleNamespace
import core

class DummyNode:
    """Simple node-like object for testing."""
    def __init__(self, metadata, score):
        self.metadata = metadata
        self.score = score

class DummyHistory:
    """Stub for the History object."""
    pass

@pytest.fixture
def dummy_history():
    """Provide a dummy History instance."""
    return DummyHistory()

@pytest.fixture
def default_mocks(monkeypatch):
    """
    Mock external dependencies: format_prompt and stream_llm_response.
    Captures call args for assertions.
    """
    call_args = {}
    def dummy_format_prompt(history, next_user_message, rag_context, system_prompt):
        call_args['history'] = history
        call_args['message'] = next_user_message
        call_args['context'] = rag_context
        call_args['system'] = system_prompt
        return f"PROMPT[{next_user_message}|{rag_context}|{system_prompt}]"
    monkeypatch.setattr(core, 'format_prompt', dummy_format_prompt)

    def dummy_stream(final_prompt, model=None):
        yield "chunk1"
        yield "chunk2"
    monkeypatch.setattr(core, 'stream_llm_response', dummy_stream)

    return call_args

class TestContextIsAnswerable:
    def test_no_context_returns_false(self):
        """Return False when context is empty or whitespace only."""
        # Arrange
        question = "What is the capital of France?"
        for ctx in ["", "   \n"]:
            # Act
            result = core.context_is_answerable(question, ctx)
            # Assert
            assert result is False

    def test_llm_returns_yes(self, monkeypatch):
        """Return True when LLM response starts with 'y' (case-insensitive)."""
        # Arrange
        monkeypatch.setattr(core, 'get_llm_completion', lambda prompt, model=None: "Yes, I can answer")
        # Act
        result = core.context_is_answerable("Q", "some context", model="gpt-test")
        # Assert
        assert result is True

    def test_llm_returns_no(self, monkeypatch):
        """Return False when LLM response does not start with 'y'."""
        # Arrange
        monkeypatch.setattr(core, 'get_llm_completion', lambda prompt, model=None: "nope")
        # Act
        result = core.context_is_answerable("Q", "ctx")
        # Assert
        assert result is False

    def test_llm_raises_exception(self, monkeypatch):
        """Return True (fail open) when get_llm_completion raises an exception."""
        # Arrange
        def raise_exc(prompt, model=None):
            raise RuntimeError("LLM error")
        monkeypatch.setattr(core, 'get_llm_completion', raise_exc)
        # Act
        result = core.context_is_answerable("Q", "ctx")
        # Assert
        assert result is True

class TestExtractSources:
    def test_extracts_unique_filenames(self):
        """Collect unique file_name metadata, defaulting to 'unknown' if missing."""
        # Arrange
        nodes = [
            DummyNode({'file_name': 'file1.txt'}, 0.9),
            DummyNode({'file_name': 'file2.txt'}, 0.8),
            DummyNode({'file_name': 'file1.txt'}, 0.5),
            DummyNode({}, 0.7),
        ]
        # Act
        result = core.extract_sources(nodes)
        # Assert
        assert result == {'file1.txt', 'file2.txt', 'unknown'}

class TestProcessQuery:
    def test_process_query_without_rag(self, dummy_history, default_mocks):
        """Skip RAG when use_rag is False."""
        # Arrange
        user_prompt = "hello"
        # Act
        gen, sources = core.process_query(user_prompt, dummy_history, use_rag=False, system_prompt="sys", model="m")
        output = list(gen)
        # Assert
        assert output == ["chunk1", "chunk2"]
        assert sources == set()
        assert default_mocks['history'] is dummy_history
        assert default_mocks['message'] == user_prompt
        assert default_mocks['context'] == ""
        assert default_mocks['system'] == "sys"

    def test_process_query_rag_disabled(self, dummy_history, default_mocks, monkeypatch):
        """Skip RAG when RAG_ENABLED is False even if use_rag is True."""
        # Arrange
        monkeypatch.setattr(core, 'RAG_ENABLED', False)
        user_prompt = "hello2"
        # Act
        gen, sources = core.process_query(user_prompt, dummy_history, use_rag=True)
        output = list(gen)
        # Assert
        assert output == ["chunk1", "chunk2"]
        assert sources == set()
        assert default_mocks['message'] == user_prompt

    def test_process_query_rag_no_nodes(self, dummy_history, default_mocks, monkeypatch):
        """Skip RAG when retrieve returns no nodes."""
        # Arrange
        monkeypatch.setattr(core, 'RAG_ENABLED', True)
        settings = SimpleNamespace(similarity_top_k=1, abs_min_score=0.5, rel_window=0.1, enable_answerability_check=False)
        monkeypatch.setattr(core, 'settings', settings)
        monkeypatch.setattr(core, 'retrieve', lambda prompt, k: [])
        # Act
        gen, sources = core.process_query("q3", dummy_history, use_rag=True)
        output = list(gen)
        # Assert
        assert output == ["chunk1", "chunk2"]
        assert sources == set()
        assert default_mocks['context'] == ""

    def test_process_query_with_rag_and_context(self, dummy_history, default_mocks, monkeypatch):
        """Include rag_context and sources when nodes meet thresholds and answerability is disabled."""
        # Arrange
        monkeypatch.setattr(core, 'RAG_ENABLED', True)
        settings = SimpleNamespace(similarity_top_k=3, abs_min_score=0.5, rel_window=0.1, enable_answerability_check=False)
        monkeypatch.setattr(core, 'settings', settings)
        nodes = [
            DummyNode({'file_name': 'a.txt'}, 0.8),
            DummyNode({'file_name': 'b.txt'}, 0.75),
            DummyNode({'file_name': 'c.txt'}, 0.6),
        ]
        monkeypatch.setattr(core, 'retrieve', lambda prompt, k: nodes)
        monkeypatch.setattr(core, 'format_context', lambda good_nodes: "CTX")
        # Act
        gen, sources = core.process_query("q", dummy_history, use_rag=True, system_prompt=None)
        output = list(gen)
        # Assert
        assert output == ["chunk1", "chunk2"]
        assert sources == {'a.txt', 'b.txt'}
        assert default_mocks['context'] == "CTX"

    def test_process_query_answerability_filters_context(self, dummy_history, default_mocks, monkeypatch):
        """Clear rag_context and sources when answerability check fails."""
        # Arrange
        monkeypatch.setattr(core, 'RAG_ENABLED', True)
        settings = SimpleNamespace(similarity_top_k=1, abs_min_score=0.1, rel_window=0.05, enable_answerability_check=True)
        monkeypatch.setattr(core, 'settings', settings)
        node = DummyNode({'file_name': 'x.txt'}, 0.2)
        monkeypatch.setattr(core, 'retrieve', lambda prompt, k: [node])
        monkeypatch.setattr(core, 'format_context', lambda nodes: "CTX2")
        monkeypatch.setattr(core, 'context_is_answerable', lambda q, c, model=None: False)
        # Act
        gen, sources = core.process_query("q2", dummy_history, use_rag=True)
        output = list(gen)
        # Assert
        assert output == ["chunk1", "chunk2"]
        assert sources == set()
        assert default_mocks['context'] == ""

    def test_process_query_retrieval_exception(self, dummy_history, default_mocks, monkeypatch):
        """Proceed without context when retrieve raises an exception."""
        # Arrange
        monkeypatch.setattr(core, 'RAG_ENABLED', True)
        settings = SimpleNamespace(similarity_top_k=2, abs_min_score=0.1, rel_window=0.05, enable_answerability_check=True)
        monkeypatch.setattr(core, 'settings', settings)
        def raise_exc(prompt, k):
            raise ValueError("fail")
        monkeypatch.setattr(core, 'retrieve', raise_exc)
        # Act
        gen, sources = core.process_query("error", dummy_history)
        output = list(gen)
        # Assert
        assert output == ["chunk1", "chunk2"]
        assert sources == set()
