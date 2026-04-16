"""Working memory — bounded in-memory buffer for the current session.

Mirrors the 7±2 capacity of human working memory. Items are evicted
oldest-first when the buffer is full. At session end the buffer is
summarized and cleared — the summary gets written to episodic storage
by the MemoryController.
"""

from __future__ import annotations

from collections import deque

from src.memory.models import WorkingMemoryItem


class WorkingMemory:
    def __init__(self, capacity: int = 10) -> None:
        self.capacity = capacity
        self._items: deque[WorkingMemoryItem] = deque(maxlen=capacity)

    def add(
        self,
        content: str,
        importance: float = 0.5,
        source: str = "",
    ) -> None:
        self._items.append(
            WorkingMemoryItem(content=content, importance=importance, source=source)
        )

    def get_items(self) -> list[WorkingMemoryItem]:
        return list(self._items)

    def get_context(self) -> str:
        """Formatted string for prompt injection."""
        if not self._items:
            return ""
        lines = ["Recent session context:"]
        for item in self._items:
            lines.append(f"  - {item.content}")
        return "\n".join(lines)

    def summarize(self) -> str:
        """Compact one-liner for episodic storage before clearing."""
        if not self._items:
            return ""
        high = [i for i in self._items if i.importance >= 0.7]
        items = high or list(self._items)
        return "; ".join(i.content for i in items[-5:])

    def clear(self) -> list[WorkingMemoryItem]:
        """Clear and return the evicted items."""
        items = list(self._items)
        self._items.clear()
        return items

    def __len__(self) -> int:
        return len(self._items)
