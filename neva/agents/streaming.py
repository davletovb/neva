"""Bounded delivery and explicit outcomes for synchronous model streams."""

from __future__ import annotations

import asyncio
import queue
import threading
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

from neva.utils.exceptions import BackendError, RateLimiterCancelledError


@dataclass(frozen=True)
class StreamEvent:
    """A text fragment or the terminal successful result of a model call."""

    kind: str  # "delta" or "complete"
    text: str
    first_token_seconds: Optional[float] = None
    completion_seconds: Optional[float] = None


class StreamInterruptedError(BackendError):
    """A failed stream; ``partial_text`` has been delivered but is not committed."""

    def __init__(self, message: str, partial_text: str, first_token_seconds: Optional[float]):
        super().__init__(message)
        self.partial_text = partial_text
        self.first_token_seconds = first_token_seconds


class StreamSession:
    """A single-use bounded stream. Close explicitly if consumption stops early.

    The worker can remain inside a synchronous SDK/socket read until its configured
    timeout; closing the session signals cancellation and stops queued delivery.
    """

    def __init__(
        self,
        producer: Callable[[Callable[[str], None], threading.Event], StreamEvent],
        *,
        max_queue_size: int = 8,
    ) -> None:
        if (
            isinstance(max_queue_size, bool)
            or not isinstance(max_queue_size, int)
            or max_queue_size < 1
        ):
            raise ValueError("max_queue_size must be a positive integer")
        self._queue: queue.Queue[object] = queue.Queue(maxsize=max_queue_size)
        self._cancel = threading.Event()
        self._producer = producer
        self._started = False
        self._lock = threading.Lock()

    def _put(self, item: object) -> None:
        while not self._cancel.is_set():
            try:
                self._queue.put(item, timeout=0.05)
                return
            except queue.Full:
                continue
        raise RateLimiterCancelledError("Stream delivery cancelled")

    def _start(self) -> None:
        with self._lock:
            if self._cancel.is_set():
                raise RateLimiterCancelledError("Stream session is closed")
            if self._started:
                raise RuntimeError("StreamSession can only be consumed once")
            self._started = True

        def run() -> None:
            try:
                result = self._producer(
                    lambda text: self._put(StreamEvent("delta", text)), self._cancel
                )
                self._put(result)
            except BaseException as exc:
                try:
                    self._put(exc)
                except RateLimiterCancelledError:
                    pass

        threading.Thread(target=run, daemon=True, name="neva-stream").start()

    def __iter__(self) -> Iterator[StreamEvent]:
        self._start()
        try:
            while True:
                if self._cancel.is_set():
                    raise RateLimiterCancelledError("Stream delivery cancelled")
                try:
                    item = self._queue.get(timeout=0.05)
                except queue.Empty:
                    if self._cancel.is_set():
                        raise RateLimiterCancelledError("Stream delivery cancelled")
                    continue
                if isinstance(item, BaseException):
                    raise item
                assert isinstance(item, StreamEvent)
                yield item
                if item.kind == "complete":
                    return
        finally:
            self.close()

    async def __aiter__(self):
        self._start()

        def receive() -> object:
            while not self._cancel.is_set():
                try:
                    return self._queue.get(timeout=0.05)
                except queue.Empty:
                    continue
            raise RateLimiterCancelledError("Stream delivery cancelled")

        try:
            while True:
                if self._cancel.is_set():
                    raise RateLimiterCancelledError("Stream delivery cancelled")
                item = await asyncio.to_thread(receive)
                if isinstance(item, BaseException):
                    raise item
                assert isinstance(item, StreamEvent)
                yield item
                if item.kind == "complete":
                    return
        finally:
            self.close()

    def close(self) -> None:
        self._cancel.set()

    async def aclose(self) -> None:
        self.close()
