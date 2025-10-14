"""Tests for the structured logging helpers."""

from __future__ import annotations

import io
import json
import logging

from neva.utils.logging_utils import JsonLogFormatter, configure_logging


def test_json_log_formatter_includes_default_fields():
    logger = logging.getLogger("telemetry")
    record = logger.makeRecord(
        name="telemetry",
        level=logging.WARNING,
        fn=__file__,
        lno=42,
        msg="warning issued",
        args=(),
        exc_info=None,
        extra={"detail": "extra"},
    )
    formatter = JsonLogFormatter()
    rendered = formatter.format(record)
    payload = json.loads(rendered)

    assert payload["level"] == "WARNING"
    assert payload["logger"] == "telemetry"
    assert payload["message"] == "warning issued"
    assert payload["detail"] == "extra"


def test_configure_logging_sets_json_formatter_and_level():
    stream = io.StringIO()

    class CaptureHandler(logging.StreamHandler):
        def __init__(self) -> None:
            super().__init__(stream)
            self.formatted: list[str] = []

        def emit(
            self, record: logging.LogRecord
        ) -> None:  # pragma: no cover - uses base implementation.
            message = self.format(record)
            self.formatted.append(message)
            stream.write(message + "\n")

    handler = CaptureHandler()
    configure_logging(level=logging.DEBUG, handlers=[handler], include_timestamp=False)

    logger = logging.getLogger("neva.logging.test")
    logger.debug("structured message", extra={"context": "value"})

    assert handler.formatted, "Handler should capture at least one record"
    payload = json.loads(handler.formatted[0])
    assert payload["message"] == "structured message"
    assert payload["context"] == "value"
    assert "timestamp" not in payload

    # Ensure the root logger level respects the configuration.
    assert logging.getLogger().level == logging.DEBUG
