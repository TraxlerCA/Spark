# test_script.py
import pytest
import logging
import sys
import json
import logging_config

from logging_config import JSONFormatter, setup_logging

class TestJSONFormatter:
    def test_format_without_exception_and_extra(self):
        """
        verify format outputs json with timestamp, level, logger, and message without exception or extra
        """
        # Arrange
        formatter = JSONFormatter()
        # stub formatTime to produce a predictable timestamp
        formatter.formatTime = lambda record, datefmt: "2025-06-23T12:00:00"
        record = logging.LogRecord(
            name="test_logger",
            level=logging.WARNING,
            pathname=__file__,
            lineno=10,
            msg="hello %s",
            args=("world",),
            exc_info=None
        )
        # Act
        result = formatter.format(record)
        # Assert
        payload = json.loads(result)
        assert payload["timestamp"] == "2025-06-23T12:00:00"
        assert payload["level"] == "WARNING"
        assert payload["logger"] == "test_logger"
        assert payload["message"] == "hello world"
        assert "exception" not in payload
        assert "foo" not in payload

    def test_format_with_exception(self):
        """
        verify format includes exception info when exc_info is provided
        """
        # Arrange
        formatter = JSONFormatter()
        formatter.formatTime = lambda record, datefmt: "2025-06-23T12:00:00"
        try:
            raise ValueError("oops")
        except ValueError:
            exc_info = sys.exc_info()
        record = logging.LogRecord(
            name="error_logger",
            level=logging.ERROR,
            pathname=__file__,
            lineno=20,
            msg="error happened",
            args=(),
            exc_info=exc_info
        )
        # Act
        result = formatter.format(record)
        # Assert
        payload = json.loads(result)
        assert payload["timestamp"] == "2025-06-23T12:00:00"
        assert payload["level"] == "ERROR"
        assert payload["logger"] == "error_logger"
        assert payload["message"] == "error happened"
        assert "exception" in payload
        # exception should mention the ValueError
        assert "ValueError: oops" in payload["exception"]

    def test_format_with_extra_fields(self):
        """
        verify format includes extra fields present in record.extra
        """
        # Arrange
        formatter = JSONFormatter()
        formatter.formatTime = lambda record, datefmt: "2025-06-23T12:00:00"
        record = logging.LogRecord(
            name="extra_logger",
            level=logging.INFO,
            pathname=__file__,
            lineno=30,
            msg="info with extra",
            args=(),
            exc_info=None
        )
        # attach extra attributes
        record.extra = {"foo": "bar", "baz": 123}
        # Act
        result = formatter.format(record)
        # Assert
        payload = json.loads(result)
        assert payload["timestamp"] == "2025-06-23T12:00:00"
        assert payload["level"] == "INFO"
        assert payload["logger"] == "extra_logger"
        assert payload["message"] == "info with extra"
        assert payload["foo"] == "bar"
        assert payload["baz"] == 123

class TestSetupLogging:
    def test_setup_logging_configures_logger(self):
        """
        verify setup_logging returns logger with INFO level and proper handler
        """
        # Arrange
        logger_name = logging_config.__name__
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        # Act
        configured = setup_logging()
        # Assert
        assert configured is logger
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1
        handler = logger.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        assert handler.stream is sys.stdout
        assert isinstance(handler.formatter, JSONFormatter)

    def test_setup_logging_is_idempotent(self):
        """
        verify calling setup_logging twice does not add duplicate handlers
        """
        # Arrange
        logger_name = logging_config.__name__
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        # Act
        first = setup_logging()
        second = setup_logging()
        # Assert
        assert first is second
        assert len(second.handlers) == 1

    def test_integration_logging_output(self, capsys, monkeypatch):
        """
        verify logger writes json output to stdout on logging an info message
        """
        # Arrange
        # stub formatTime for predictability
        monkeypatch.setattr(JSONFormatter, "formatTime", lambda self, record, datefmt: "2025-06-23T12:00:00")
        logger_name = logging_config.__name__
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger = setup_logging()
        # Act
        logger.info("integration test", extra={"extra": {"user": "tester"}})
        captured = capsys.readouterr().out.strip()
        # Assert
        payload = json.loads(captured)
        assert payload["timestamp"] == "2025-06-23T12:00:00"
        assert payload["level"] == "INFO"
        assert payload["logger"] == logger_name
        assert payload["message"] == "integration test"
        assert payload["user"] == "tester"
