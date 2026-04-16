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
    content: str  # natural-language description of the experience
    embedding: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    importance: float = 0.5  # 0–1; affects retention during pruning
    recall_count: int = 0  # how many times retrieved
    last_recalled: str | None = None


class SemanticEntry(BaseModel):
    """A fact or pattern extracted from multiple episodes."""

    id: str = Field(default_factory=_uuid)
    concept: str  # the fact / pattern in plain language
    category: str  # e.g. "error_pattern", "success_pattern", "algorithm"
    evidence_count: int = 1
    confidence: float = 0.5  # strengthens each time the pattern is observed
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)


class WorkingMemoryItem(BaseModel):
    """An item currently held in the active session context."""

    content: str
    importance: float = 0.5
    source: str = ""
    timestamp: str = Field(default_factory=_now)


class MemoryContext(BaseModel):
    """Assembled context injected into a specialist prompt."""

    episodic: list[EpisodicEntry] = Field(default_factory=list)
    semantic: list[SemanticEntry] = Field(default_factory=list)
    working: list[WorkingMemoryItem] = Field(default_factory=list)

    def to_prompt_text(self) -> str:
        """Format all memory tiers as a compact prompt section."""
        parts: list[str] = []

        if self.working:
            parts.append("## Recent session context")
            for item in self.working[-5:]:
                parts.append(f"- {item.content}")

        if self.semantic:
            parts.append("## Relevant knowledge")
            for sem_entry in self.semantic[:5]:
                parts.append(f"- {sem_entry.concept}")

        if self.episodic:
            parts.append("## Similar past experiences")
            for ep_entry in self.episodic[:3]:
                parts.append(f"- {ep_entry.content}")

        return "\n".join(parts)

    def is_empty(self) -> bool:
        return not (self.working or self.semantic or self.episodic)
