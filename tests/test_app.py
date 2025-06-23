import pytest
from unittest.mock import MagicMock

import app


class TestCheckVectorStoreExists:
    """Tests for the check_vector_store_exists helper function."""

    def test_returns_false_if_persist_dir_not_exist(self, monkeypatch):
        """Test that it returns False when the persist directory does not exist."""
        # Arrange
        fake_path = MagicMock()
        fake_path.exists.return_value = False
        monkeypatch.setattr(app.settings, "persist_dir", fake_path)

        # Act
        result = app.check_vector_store_exists()

        # Assert
        assert result is False

    def test_returns_true_when_collection_has_items(self, monkeypatch):
        """Test that it returns True when the directory exists and collection count > 0."""
        # Arrange
        fake_path = MagicMock()
        fake_path.exists.return_value = True
        monkeypatch.setattr(app.settings, "persist_dir", fake_path)

        class DummyCollection:
            def __init__(self, count):
                self._count = count

            def count(self):
                return self._count

        class DummyClient:
            def __init__(self, path):
                self.path = path

            def get_collection(self, name):
                return DummyCollection(5)

        monkeypatch.setattr(app.chromadb, "PersistentClient", DummyClient)

        # Act
        result = app.check_vector_store_exists()

        # Assert
        assert result is True

    def test_returns_false_when_collection_empty(self, monkeypatch):
        """Test that it returns False when the collection exists but is empty."""
        # Arrange
        fake_path = MagicMock()
        fake_path.exists.return_value = True
        monkeypatch.setattr(app.settings, "persist_dir", fake_path)

        class DummyCollection:
            def __init__(self, count):
                self._count = count

            def count(self):
                return self._count

        class DummyClient:
            def __init__(self, path):
                pass

            def get_collection(self, name):
                return DummyCollection(0)

        monkeypatch.setattr(app.chromadb, "PersistentClient", DummyClient)

        # Act
        result = app.check_vector_store_exists()

        # Assert
        assert result is False

    def test_returns_false_on_exception(self, monkeypatch):
        """Test that it returns False if PersistentClient or count() raises."""
        # Arrange
        fake_path = MagicMock()
        fake_path.exists.return_value = True
        monkeypatch.setattr(app.settings, "persist_dir", fake_path)

        # client init raises
        def bad_client_init(path):
            raise RuntimeError("oops")

        monkeypatch.setattr(app.chromadb, "PersistentClient", bad_client_init)

        # Act
        result = app.check_vector_store_exists()

        # Assert
        assert result is False


class TestInitSessionState:
    """Tests for the init_session_state function."""

    def test_initializes_session_state_keys(self, monkeypatch):
        """Test that missing keys are added with correct defaults."""
        # Arrange
        # create empty session_state
        fake_session = {}
        monkeypatch.setattr(app.st, "session_state", fake_session)
        # stub config defaults
        monkeypatch.setattr(app.settings, "ollama_model", "model-x")
        monkeypatch.setattr(app, "DEFAULT_SYSTEM_PROMPT", "prompt-x")
        monkeypatch.setattr(app, "RAG_AVAILABLE", True)

        # Act
        app.init_session_state()

        # Assert
        assert fake_session[app.SESSION_KEY_HISTORY] == []
        assert fake_session[app.SESSION_KEY_MODEL] == "model-x"
        assert fake_session[app.SESSION_KEY_SYSTEM_PROMPT] == "prompt-x"
        assert fake_session[app.SESSION_KEY_USE_RAG] is True

    def test_does_not_override_existing_keys(self, monkeypatch):
        """Test that existing session_state values are preserved."""
        # Arrange
        fake_session = {
            app.SESSION_KEY_HISTORY: ["already"],
            app.SESSION_KEY_MODEL: "old-model",
            app.SESSION_KEY_SYSTEM_PROMPT: "old-prompt",
            app.SESSION_KEY_USE_RAG: False,
        }
        monkeypatch.setattr(app.st, "session_state", fake_session)
        # change defaults to see if overridden
        monkeypatch.setattr(app.settings, "ollama_model", "new-model")
        monkeypatch.setattr(app, "DEFAULT_SYSTEM_PROMPT", "new-prompt")
        monkeypatch.setattr(app, "RAG_AVAILABLE", True)

        # Act
        app.init_session_state()

        # Assert
        assert fake_session[app.SESSION_KEY_HISTORY] == ["already"]
        assert fake_session[app.SESSION_KEY_MODEL] == "old-model"
        assert fake_session[app.SESSION_KEY_SYSTEM_PROMPT] == "old-prompt"
        assert fake_session[app.SESSION_KEY_USE_RAG] is False


class TestMain:
    """Tests for the main entry point behavior."""

    class StopCalled(Exception):
        """Custom exception to signal st.stop was called."""

    def test_main_warns_and_stops_when_vector_store_missing(self, monkeypatch):
        """Test that main shows a warning and stops if vector store is uninitialized."""
        # Arrange
        monkeypatch.setattr(app, "check_vector_store_exists", lambda: False)
        warnings = []
        monkeypatch.setattr(app.st, "warning", lambda msg, icon: warnings.append((msg, icon)))
        def fake_stop():
            raise TestMain.StopCalled()
        monkeypatch.setattr(app.st, "stop", fake_stop)
        monkeypatch.setattr(app.st, "set_page_config", lambda **kwargs: None)

        # Act / Assert
        with pytest.raises(TestMain.StopCalled):
            app.main()
        assert any("vector store is not initialised or empty" in msg for msg, _ in warnings)

    def test_main_initializes_and_renders_and_runs_chat_when_vector_store_exists(self, monkeypatch):
        """Test that main calls init_session_state, render_sidebar, and run_chat in normal startup."""
        # Arrange
        monkeypatch.setattr(app, "check_vector_store_exists", lambda: True)
        called = []
        monkeypatch.setattr(app, "init_session_state", lambda: called.append("init"))
        monkeypatch.setattr(app, "render_sidebar", lambda: called.append("sidebar"))
        monkeypatch.setattr(app, "run_chat", lambda: called.append("chat"))
        monkeypatch.setattr(app.st, "set_page_config", lambda **kwargs: called.append("config"))

        # Act
        app.main()

        # Assert
        assert called == ["config", "init", "sidebar", "chat"]
