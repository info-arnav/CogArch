"""Data models for the memory system."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
    return str(uuid.uuid4())


class EpisodicEntry(BaseModel):
    """A single remembered experience."""

    id: str = Field(default_factory=_uuid)
    timestamp: str = Field(default_factory=_now)
    content: str  # what the agent experienced, in its own terms
    embedding: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    importance: float = 0.5  # 0–1; affects retention during pruning
    recall_count: int = 0
    last_recalled: str | None = None


class SemanticEntry(BaseModel):
    """A general lesson distilled from many episodes during sleep."""

    id: str = Field(default_factory=_uuid)
    concept: str
    category: str
    evidence_count: int = 1
    confidence: float = 0.5  # strengthens each time the lesson is re-derived
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)


class WorkingMemoryItem(BaseModel):
    """An item held in the current session's context."""

    content: str
    importance: float = 0.5
    source: str = ""
    timestamp: str = Field(default_factory=_now)


class MemoryContext(BaseModel):
    """What an agent recalls before working on a problem."""

    episodic: list[EpisodicEntry] = Field(default_factory=list)
    semantic: list[SemanticEntry] = Field(default_factory=list)
    working: list[WorkingMemoryItem] = Field(default_factory=list)

    def to_prompt_text(self) -> str:
        parts: list[str] = []
        if self.semantic:
            parts.append("## Lessons you have learned")
            parts.extend(f"- {s.concept}" for s in self.semantic)
        if self.episodic:
            parts.append("## Similar problems you have seen")
            parts.extend(e.content for e in self.episodic)
        if self.working:
            parts.append("## Recent events")
            parts.extend(f"- {w.content}" for w in self.working)
        return "\n".join(parts)

    def is_empty(self) -> bool:
        return not (self.working or self.semantic or self.episodic)
