"""One agent's memory: working, episodic and semantic stores.

Procedural memory is the agent's fine-tuned weights (see src/training).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from src.memory.embedder import Embedder
from src.memory.episodic import EpisodicStore
from src.memory.models import EpisodicEntry, MemoryContext, SemanticEntry
from src.memory.semantic import SemanticStore
from src.memory.working import WorkingMemory

LESSON = "lesson"


class AgentMemory:
    def __init__(
        self,
        directory: Path,
        embedder: Embedder,
        working_capacity: int = 10,
        retrieve_k: int = 3,
        min_similarity: float = 0.5,
        max_episodic: int = 2000,
        lessons_in_prompt: int = 5,
    ) -> None:
        self.episodic = EpisodicStore(directory / "episodic.jsonl")
        self.semantic = SemanticStore(directory / "semantic.db")
        self.working = WorkingMemory(capacity=working_capacity)
        self.embedder = embedder
        self.retrieve_k = retrieve_k
        self.min_similarity = min_similarity
        self.max_episodic = max_episodic
        self.lessons_in_prompt = lessons_in_prompt

    async def _embed(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embedder.embed, text)

    # ── waking ────────────────────────────────────────────────────────────────

    async def recall(self, query: str) -> MemoryContext:
        """Lessons, similar past episodes and recent events relevant to `query`."""
        episodes = self.episodic.retrieve(
            await self._embed(query),
            k=self.retrieve_k,
            min_similarity=self.min_similarity,
        )
        return MemoryContext(
            episodic=episodes,
            semantic=self.lessons()[: self.lessons_in_prompt],
            working=self.working.get_items(),
        )

    async def record(
        self,
        content: str,
        key: str,
        importance: float,
        metadata: dict[str, Any],
    ) -> None:
        """Store an episode, indexed by the embedding of `key`."""
        self.episodic.add(
            EpisodicEntry(
                content=content,
                embedding=await self._embed(key),
                importance=importance,
                metadata=metadata,
            )
        )

    def note(self, content: str, importance: float = 0.5) -> None:
        self.working.add(content, importance=importance)

    # ── sleep ─────────────────────────────────────────────────────────────────

    def episodes_from_cycle(self, cycle: int) -> list[EpisodicEntry]:
        return [
            e
            for e in self.episodic.get_recent(len(self.episodic))
            if e.metadata.get("cycle") == cycle
        ]

    def lessons(self) -> list[SemanticEntry]:
        """Current lessons, most trusted first."""
        return self.semantic.retrieve(category=LESSON, min_confidence=0.0, limit=50)

    def apply_reflection(
        self,
        ops: list[tuple[str, int | None, str]],
        numbered: list[SemanticEntry],
        banned_terms: set[str],
    ) -> list[str]:
        """Apply ADD / AGREE / DISAGREE / EDIT operations from a sleep reflection.

        `numbered` is the lesson list the reflection saw (1-based). Lessons that
        mention any banned term (e.g. a problem's function name) are rejected,
        so lessons stay general instead of memorising specific problems.
        """

        def allowed(text: str) -> bool:
            lowered = text.lower()
            return not any(term.lower() in lowered for term in banned_terms)

        for kind, number, text in ops:
            target = (
                numbered[number - 1]
                if number is not None and 1 <= number <= len(numbered)
                else None
            )
            if kind == "add" and allowed(text):
                self.semantic.add_or_update(text, LESSON, confidence_delta=0.1)
            elif kind == "agree" and target is not None:
                self.semantic.adjust(target.id, 0.1)
            elif kind == "disagree" and target is not None:
                self.semantic.adjust(target.id, -0.2)
            elif kind == "edit" and target is not None and allowed(text):
                self.semantic.edit(target.id, text)
        return [entry.concept for entry in self.lessons()]

    def prune(self) -> int:
        return self.episodic.prune(max_entries=self.max_episodic)

    def end_session(self) -> None:
        self.working.clear()

    def stats(self) -> dict[str, int]:
        return {
            "episodic": len(self.episodic),
            "semantic": len(self.semantic),
            "working": len(self.working),
        }
