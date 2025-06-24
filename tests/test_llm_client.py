import json
import pytest
import requests
from requests.exceptions import HTTPError, Timeout, ConnectionError, RequestException

import llm_client
from llm_client import (
    stream_llm_response,
    get_llm_completion,
    _validate_ollama_url,
    OllamaConnectionError,
    OllamaResponseError,
    OllamaTimeoutError,
    OllamaClientError,
    MAX_PROMPT_LENGTH,
)
import config


class TestValidateOllamaUrl:
    def test_accepts_https_hosts(self):
        """_validate_ollama_url should not raise for HTTPS URLs on non-local hosts."""
        url = "https://example.com/api"
        # Act / Assert: no exception
        _validate_ollama_url(url)

    def test_accepts_localhost_with_http(self):
        """_validate_ollama_url should not raise for localhost or 127.0.0.1 even if using HTTP."""
        for host in ("http://localhost/api", "http://127.0.0.1/api"):
            # Act / Assert: no exception
            _validate_ollama_url(host)

    def test_rejects_insecure_non_local_http(self):
        """_validate_ollama_url should raise ValueError for HTTP on non-local hosts."""
        url = "http://insecure.example.com/api"
        with pytest.raises(ValueError) as exc:
            _validate_ollama_url(url)
        assert "Insecure Ollama URL configured for non-local host" in str(exc.value)


class TestStreamLlmResponse:
    @pytest.fixture(autouse=True)
    def setup_settings(self, monkeypatch):
        """Monkeypatch settings to consistent, secure values."""
        monkeypatch.setattr(config.settings, "ollama_model", "test-model")
        monkeypatch.setattr(config.settings, "ollama_host", "https://api.test")
        monkeypatch.setattr(config.settings, "ollama_port", 1234)
        monkeypatch.setattr(config.settings, "ollama_timeout", 5)

    def make_response(self, lines, raise_exc=None):
        """Helper to build a fake streaming response."""
        class DummyResponse:
            def __init__(self, lines):
                self._lines = lines
                self.status_code = 200

            def raise_for_status(self):
                if raise_exc:
                    raise raise_exc

            def iter_lines(self, decode_unicode=True):
                for line in self._lines:
                    yield line

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, tb):
                return False

        return DummyResponse(lines)

    def test_streams_chunks_until_done(self, monkeypatch):
        """should yield all 'response' chunks and stop at 'done'."""
        lines = [
            b'{"response":"Hello"}',
            b"",
            b'{"response":"World","done":true}'
        ]
        monkeypatch.setattr(requests, "post", lambda *a, **k: self.make_response(lines))
        # Act
        chunks = list(stream_llm_response("hi"))
        # Assert
        assert chunks == ["Hello", "World"]

    def test_skips_invalid_json_lines(self, monkeypatch):
        """should skip lines that are not valid JSON."""
        lines = [
            b"not-a-json",
            b'{"response":"Data","done":true}'
        ]
        monkeypatch.setattr(requests, "post", lambda *a, **k: self.make_response(lines))
        chunks = list(stream_llm_response("prompt"))
        assert chunks == ["Data"]

    def test_error_field_raises_response_error(self, monkeypatch):
        """should raise OllamaResponseError if a stream message contains 'error'."""
        lines = [b'{"error":"oops"}']
        monkeypatch.setattr(requests, "post", lambda *a, **k: self.make_response(lines))
        with pytest.raises(OllamaResponseError) as exc:
            list(stream_llm_response("prompt"))
        assert "oops" in str(exc.value)

    def test_connection_error_wrapped(self, monkeypatch):
        """should raise OllamaConnectionError on connection failures."""
        monkeypatch.setattr(requests, "post", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("fail")))
        with pytest.raises(OllamaConnectionError):
            list(stream_llm_response("prompt"))

    def test_timeout_error_wrapped(self, monkeypatch):
        """should raise OllamaTimeoutError on timeouts."""
        monkeypatch.setattr(requests, "post", lambda *a, **k: (_ for _ in ()).throw(Timeout("tmi")))
        with pytest.raises(OllamaTimeoutError):
            list(stream_llm_response("prompt"))

    def test_http_error_wrapped_as_response_error(self, monkeypatch):
        """should wrap HTTPError from raise_for_status into OllamaResponseError."""
        http_err = HTTPError("bad", response=type("R", (), {"status_code": 418})())
        dummy = self.make_response([b'{}'], raise_exc=http_err)
        monkeypatch.setattr(requests, "post", lambda *a, **k: dummy)
        with pytest.raises(OllamaResponseError) as exc:
            list(stream_llm_response("prompt"))
        assert "418" in str(exc.value)

    def test_generic_request_exception_wrapped(self, monkeypatch):
        """should wrap other RequestException into OllamaClientError."""
        monkeypatch.setattr(requests, "post", lambda *a, **k: (_ for _ in ()).throw(RequestException("uh oh")))
        with pytest.raises(OllamaClientError):
            list(stream_llm_response("prompt"))

    @pytest.mark.parametrize("bad_prompt", ["", "   "])
    def test_empty_prompt_raises_value_error(self, bad_prompt):
        """should reject empty‐string prompts."""
        with pytest.raises(ValueError):
            next(stream_llm_response(bad_prompt))

    def test_too_long_prompt_raises_value_error(self):
        """should reject prompts exceeding MAX_PROMPT_LENGTH."""
        long_prompt = "x" * (MAX_PROMPT_LENGTH + 1)
        with pytest.raises(ValueError) as exc:
            next(stream_llm_response(long_prompt))
        assert f"Prompt exceeds max length of {MAX_PROMPT_LENGTH} chars" in str(exc.value)


class TestGetLlmCompletion:
    @pytest.fixture(autouse=True)
    def setup_settings(self, monkeypatch):
        """Monkeypatch settings to consistent, secure values."""
        monkeypatch.setattr(config.settings, "ollama_model", "sync-model")
        monkeypatch.setattr(config.settings, "ollama_host", "https://api.sync")
        monkeypatch.setattr(config.settings, "ollama_port", 4321)
        monkeypatch.setattr(config.settings, "ollama_timeout", 3)

    class DummySyncResponse:
        def __init__(self, data, status_code=200, raise_exc=None):
            self._data = data
            self.status_code = status_code
            self._raise = raise_exc

        def raise_for_status(self):
            if self._raise:
                raise self._raise

        def json(self):
            return self._data

    def test_returns_stripped_response(self, monkeypatch):
        """should return the trimmed 'response' string from JSON."""
        resp = self.DummySyncResponse({"response": "  answer  "})
        monkeypatch.setattr(requests, "post", lambda *a, **k: resp)
        result = get_llm_completion("hi")
        assert result == "answer"

    def test_missing_response_key_returns_empty(self, monkeypatch):
        """should return empty string if 'response' not present."""
        resp = self.DummySyncResponse({"foo": "bar"})
        monkeypatch.setattr(requests, "post", lambda *a, **k: resp)
        assert get_llm_completion("hi") == ""

    def test_request_exception_raises_connection_error(self, monkeypatch):
        """should wrap RequestException into OllamaConnectionError."""
        monkeypatch.setattr(requests, "post", lambda *a, **k: (_ for _ in ()).throw(RequestException("down")))
        with pytest.raises(OllamaConnectionError) as exc:
            get_llm_completion("hi")
        assert "down" in str(exc.value)

    @pytest.mark.parametrize("bad_prompt", ["", "   ", None, 123])
    def test_invalid_prompt_raises_value_error(self, bad_prompt):
        """should reject non‐string or blank prompts."""
        with pytest.raises(ValueError):
            get_llm_completion(bad_prompt)

    def test_too_long_prompt_raises_value_error(self):
        """should reject prompts exceeding MAX_PROMPT_LENGTH."""
        long_prompt = "y" * (MAX_PROMPT_LENGTH + 1)
        with pytest.raises(ValueError) as exc:
            get_llm_completion(long_prompt)
        assert f"Prompt exceeds max length of {MAX_PROMPT_LENGTH} characters" in str(exc.value)

    def test_insecure_host_rejected(self):
        """should reject insecure HTTP host in URL construction."""
        config.settings.ollama_host = "http://evil.com"
        with pytest.raises(ValueError):
            get_llm_completion("hello")
