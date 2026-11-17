"""One agent's episodic memory: specific past problems, recalled when a new
problem is unusually similar (the hippocampus's pattern completion).

General knowledge is not kept as text: it is learned slowly in the
fine-tuned weights by sleep replay, as complementary learning systems theory
places it in the neocortex (see src/training). Working memory is the current
problem's own attempts and feedback, carried within an attempt loop.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from src.memory.embedder import Embedder
from src.memory.episodic import EpisodicStore
from src.memory.models import Candidate, EpisodicEntry, MemoryContext

CANDIDATES_SHOWN = 8
DEFAULT_MIN_SIMILARITY = 0.8


class AgentMemory:
    def __init__(
        self,
        directory: Path,
        embedder: Embedder,
        retrieve_k: int = 3,
        min_similarity: dict[str, float] | None = None,
        max_episodic: int = 2000,
    ) -> None:
        self.episodic = EpisodicStore(directory / "episodic.jsonl")
        self.embedder = embedder
        self.retrieve_k = retrieve_k
        self.min_similarity = min_similarity or {}
        self.max_episodic = max_episodic

    async def _embed(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embedder.embed, text)

    async def recall(
        self, query: str, domain: str | None = None, touch: bool = True
    ) -> MemoryContext:
        """The past problems most similar to `query`, searched across all
        subjects, kept only when at least `min_similarity[domain]` alike. The
        cut-off is per subject because the embedding model rates problems that
        share a template (social questions) as close; it calibrates the model
        and is never shown to the agent. With `touch` off (evaluation),
        recalling leaves no trace in the store."""
        ranked = self.episodic.ranked(await self._embed(query))
        threshold = self.min_similarity.get(domain or "", DEFAULT_MIN_SIMILARITY)
        kept = [e for sim, e in ranked if sim >= threshold][: self.retrieve_k]
        if touch:
            self.episodic.touch(kept)
        kept_ids = {e.id for e in kept}
        return MemoryContext(
            episodic=kept,
            candidates=[
                Candidate(
                    id=e.id,
                    similarity=round(sim, 4),
                    kept=e.id in kept_ids,
                    task_id=str(e.metadata.get("task_id", "")),
                    domain=str(e.metadata.get("domain", "")),
                    solved=e.metadata.get("solved"),
                )
                for sim, e in ranked[: max(CANDIDATES_SHOWN, self.retrieve_k)]
            ],
            min_similarity=threshold,
        )

    async def search(self, query: str) -> list[tuple[float, EpisodicEntry]]:
        """Every episode by similarity to `query` (for inspection; no trace)."""
        return self.episodic.ranked(await self._embed(query))

    def annotate(self, task_id: str, cycle: int, note: str) -> EpisodicEntry | None:
        """Add what went wrong (found during sleep) to that problem's episode."""
        for entry in self.episodic.all():
            meta = entry.metadata
            if meta.get("task_id") == task_id and meta.get("cycle") == cycle:
                entry.content += f"\nWhat went wrong: {note}"
                entry.metadata["what_went_wrong"] = note
                self.episodic.update(entry)
                return entry
        return None

    async def record(
        self,
        content: str,
        key: str,
        importance: float,
        metadata: dict[str, Any],
    ) -> EpisodicEntry:
        """Store an episode, indexed by the embedding of `key`."""
        entry = EpisodicEntry(
            content=content,
            embedding=await self._embed(key),
            importance=importance,
            metadata=metadata,
        )
        self.episodic.add(entry)
        return entry

    def episodes_from_cycle(self, cycle: int) -> list[EpisodicEntry]:
        return [
            e
            for e in self.episodic.get_recent(len(self.episodic))
            if e.metadata.get("cycle") == cycle
        ]

    def prune(self) -> int:
        return self.episodic.prune(max_entries=self.max_episodic)

    def stats(self) -> dict[str, int]:
        return {"episodic": len(self.episodic)}
