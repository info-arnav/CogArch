"""The complete record of a run: every model call, grading verdict, code
execution and team decision, with its full input and output, as gzipped JSON
lines in `<run>/calls.jsonl.gz`.

Each record carries the tags in force when it was made (phase, cycle,
benchmark, system, task, sample, agent, role, step), so any answer can be
traced back to the exact prompts that produced it. The file is flushed often
enough that a crash loses at most a second of records, and a resumed run
appends a new gzip member that readers see as one stream.
"""

from __future__ import annotations

import contextvars
import gzip
import itertools
import json
import threading
import time
from collections.abc import Awaitable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")

FLUSH_SECONDS = 1.0

_tags: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "calllog_tags", default={}
)


@contextmanager
def tagged(**tags: Any) -> Iterator[None]:
    token = _tags.set({**_tags.get(), **tags})
    try:
        yield
    finally:
        _tags.reset(token)


def update(**tags: Any) -> None:
    """Add tags for the rest of the current task (and the tasks it starts)."""
    _tags.set({**_tags.get(), **tags})


async def within(tags: dict[str, Any], awaitable: Awaitable[T]) -> T:
    with tagged(**tags):
        return await awaitable


def tags() -> dict[str, Any]:
    return dict(_tags.get())


class CallLog:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._handle = None
        self._flushed = time.monotonic()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = gzip.open(path, "at", compresslevel=1, encoding="utf-8")

    def record(self, kind: str, **data: Any) -> None:
        if self._handle is None:
            return
        line = json.dumps(
            {
                "id": next(self._ids),
                "t": round(time.time(), 3),
                "kind": kind,
                **tags(),
                **data,
            },
            default=str,
        )
        with self._lock:
            self._handle.write(line + "\n")
            now = time.monotonic()
            if now - self._flushed >= FLUSH_SECONDS:
                self._handle.flush()
                self._flushed = now

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None


_log = CallLog(None)


def open_log(path: Path) -> CallLog:
    global _log
    _log.close()
    _log = CallLog(path)
    return _log


def record(kind: str, **data: Any) -> None:
    _log.record(kind, **data)


def close() -> None:
    _log.close()


def read(path: Path) -> Iterator[dict[str, Any]]:
    """Every complete record, including those before a crash cut the file short."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        try:
            for line in f:
                if line.endswith("\n") and line.strip():
                    yield json.loads(line)
        except EOFError:
            return
