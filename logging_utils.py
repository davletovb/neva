"""Utilities for configuring application-wide logging behaviour.

The project previously relied on ad-hoc ``print`` statements sprinkled across
examples and tests.  While convenient during prototyping, the approach makes it
hard to integrate with observability stacks or correlate events emitted by
multiple agents.  This module exposes a small helper that configures a
structured logging setup backed by the standard :mod:`logging` package.  The
formatter emits JSON lines so downstream tooling (for example, ``jq`` or
ElasticSearch) can ingest experiment logs without additional processing.
"""

from __future__ import annotations

import json
import logging
from logging import Handler, LogRecord
from typing import Iterable, Mapping, MutableMapping, Optional


class JsonLogFormatter(logging.Formatter):
    """A tiny JSON formatter compatible with the stdlib logging package."""

    default_fields: Mapping[str, str] = {
        "level": "levelname",
        "logger": "name",
        "message": "message",
    }

    def __init__(
        self,
        *,
        fields: Optional[Mapping[str, str]] = None,
        ensure_ascii: bool = False,
    ) -> None:
        super().__init__()
        self._fields = dict(self.default_fields)
        if fields is not None:
            self._fields.update(fields)
        self._ensure_ascii = ensure_ascii

    def format(self, record: LogRecord) -> str:  # noqa: D401 - inherited docstring
        payload: MutableMapping[str, object] = {}
        for field_name, attribute in self._fields.items():
            value = getattr(record, attribute, None)
            if attribute == "created" and value is not None:
                value = self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ")
            payload[field_name] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = record.stack_info
        if isinstance(record.args, dict):
            payload.update(record.args)
        return json.dumps(payload, ensure_ascii=self._ensure_ascii)


def configure_logging(
    *,
    level: int = logging.INFO,
    handlers: Optional[Iterable[Handler]] = None,
    include_timestamp: bool = True,
) -> None:
    """Configure global logging using the :class:`JsonLogFormatter`.

    Parameters
    ----------
    level:
        Logging level applied to the root logger.
    handlers:
        Optional custom handlers.  When omitted a :class:`logging.StreamHandler`
        pointing to ``sys.stderr`` is installed.
    include_timestamp:
        When ``True`` (the default) the formatter injects the record creation
        time in ISO 8601 format under the ``timestamp`` key.
    """

    logging.shutdown()
    root_logger = logging.getLogger()
    for existing_handler in list(root_logger.handlers):
        root_logger.removeHandler(existing_handler)

    if handlers is None:
        stream_handler = logging.StreamHandler()
        handlers = [stream_handler]

    formatter_fields = dict(JsonLogFormatter.default_fields)
    if include_timestamp:
        formatter_fields["timestamp"] = "created"

    formatter = JsonLogFormatter(fields=formatter_fields)

    for handler in handlers:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    root_logger.setLevel(level)

