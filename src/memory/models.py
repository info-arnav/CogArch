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


class Candidate(BaseModel):
    """One stored episode considered during a recall."""

    id: str
    similarity: float
    kept: bool
    task_id: str = ""
    domain: str = ""
    solved: bool | None = None


class MemoryContext(BaseModel):
    """What an agent recalls before working on a problem, and how it chose it."""

    episodic: list[EpisodicEntry] = Field(default_factory=list)
    candidates: list[Candidate] = Field(
        default_factory=list, description="The closest episodes, kept or not"
    )
    min_similarity: float = 0.0

    def to_prompt_text(self) -> str:
        if not self.episodic:
            return ""
        return "\n".join(
            ["## Similar problems you have seen", *(e.content for e in self.episodic)]
        )
