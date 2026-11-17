"""The run's event log: everything that happens, in order, as JSON lines.

The live dashboard tails this file, so a run can be watched as it happens and
replayed afterwards. `stream_tokens` adds the text of every model reply as it
is generated; without it each reply is logged once, when it finishes.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Any


class EventLog:
    def __init__(self, path: Path | None, stream_tokens: bool = False) -> None:
        self.path = path
        self.stream_tokens = stream_tokens and path is not None
        self._ids = itertools.count(1)
        self._handle = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = open(path, "a", buffering=1)

    def new_id(self) -> int:
        return next(self._ids)

    def emit(self, kind: str, **data: Any) -> None:
        if self._handle is None:
            return
        self._handle.write(
            json.dumps({"t": round(time.time(), 3), "type": kind, **data}, default=str)
            + "\n"
        )

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


NULL_LOG = EventLog(None)
