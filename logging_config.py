# spark/logging_config.py
import logging
import sys
import json

class JSONFormatter(logging.Formatter):
    """
    Custom log formatter that emits events as single-line JSON.
    Includes timestamp, log level, logger name, message, and exception info.
    """
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Add any extra fields passed to the logger.
        if hasattr(record, 'extra'):
            payload.update(record.extra) # type: ignore
        return json.dumps(payload)

def setup_logging():
    """
    Configures and returns a logger with a JSON formatter.
    """
    logger = logging.getLogger(__name__)
    # Prevent duplicate handlers if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger