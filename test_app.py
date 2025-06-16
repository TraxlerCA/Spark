# test_app.py
#
# Description: unit tests for app.py functions: sanitize_input, get_settings,
#              init_session_state.

import warnings
import pytest

import streamlit as st

import app  # so we can monkey-patch symbols on the real module
from app import (
    sanitize_input,
    get_settings,
    init_session_state,
    ChatError,
    AppSettings,
    SESSION_KEY_HISTORY,
    SESSION_KEY_MODEL,
    SESSION_KEY_SYSTEM_PROMPT,
)

# --------------------------------------------------------------------------- #
# test sanitize_input
# --------------------------------------------------------------------------- #

def test_sanitize_input_strips_whitespace():
    """inputs with leading and trailing whitespace are trimmed correctly."""
    assert sanitize_input("  hi there  ") == "hi there"


# --------------------------------------------------------------------------- #
# test get_settings
# --------------------------------------------------------------------------- #

def test_get_settings_default_model(monkeypatch):
    """when no APP_MODEL is set, default model from settings is returned."""
    get_settings.clear()
    monkeypatch.delenv("APP_MODEL", raising=False)

    settings = get_settings()

    assert settings.model == "gemma3:4b"


def test_get_settings_custom_model(monkeypatch):
    """when APP_MODEL is set, that value is loaded."""
    get_settings.clear()
    monkeypatch.setenv("APP_MODEL", "deepseek-r1:latest")

    assert get_settings().model == "deepseek-r1:latest"


def test_get_settings_invalid_settings_raises_chat_error(monkeypatch):
    """
    if AppSettings.__init__ raises a ValidationError-like exception,
    get_settings must wrap it in ChatError and not print the log.
    """
    get_settings.clear()

    # suppress the JSON log output by turning logger.error into a no-op
    monkeypatch.setattr(app.logger, "error", lambda *args, **kwargs: None)

    class FakeValidationError(Exception):
        """minimal stand-in for pydantic.ValidationError used only in this test."""

        def errors(self):
            return [{"loc": ["model"], "msg": "bad model", "type": "value_error"}]

    # override ValidationError in the app module
    monkeypatch.setattr(app, "ValidationError", FakeValidationError, raising=True)

    # make AppSettings.__init__ always raise our fake error
    def always_fail_init(self, *args, **kwargs):
        raise FakeValidationError("bad configuration")

    monkeypatch.setattr(app.AppSettings, "__init__", always_fail_init, raising=True)

    with pytest.raises(ChatError) as exc:
        get_settings()

    assert "invalid application settings" in str(exc.value)


# --------------------------------------------------------------------------- #
# test init_session_state
# --------------------------------------------------------------------------- #

def test_init_session_state_initializes_keys(monkeypatch):
    """history, model, and system_prompt keys are set with correct defaults."""
    get_settings.clear()
    monkeypatch.delenv("APP_MODEL", raising=False)

    fake_state = {}
    monkeypatch.setattr(st, "session_state", fake_state)

    init_session_state()

    assert fake_state[SESSION_KEY_HISTORY] == []
    assert fake_state[SESSION_KEY_MODEL] == get_settings().model
    assert fake_state[SESSION_KEY_SYSTEM_PROMPT].startswith("You are a helpful AI")


# --------------------------------------------------------------------------- #
# allow direct execution
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        message="No runtime found, using MemoryCacheStorageManager",
    )
    import pytest as _pytest
    _pytest.main([__file__, "-q"])
