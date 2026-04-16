"""Memory controller — coordinates all three memory tiers.

Usage
-----
    memory = MemoryController(memory_dir="data/memory", ollama_base_url=...)

    # Record an experience (writes to working + episodic)
    await memory.record("Solved two-sum using a hash map", importance=0.9,
                        metadata={"specialist": "logical", "passed_all": True})

    # Retrieve relevant context before solving a new problem
    ctx = await memory.retrieve_context("two-sum problem with O(n) constraint")
    prompt_section = ctx.to_prompt_text()

    # At end of experiment cycle (sleep phase)
    stats = memory.consolidate()

    # At end of session — summarise working memory and clear it
    memory.end_session()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.memory.consolidator import MemoryConsolidator
from src.memory.embedder import embed
from src.memory.episodic import EpisodicStore
from src.memory.models import EpisodicEntry, MemoryContext
from src.memory.semantic import SemanticStore
from src.memory.working import WorkingMemory


class MemoryController:
    """Coordinates working memory, episodic store, and semantic store."""

    def __init__(
        self,
        memory_dir: Path | str = "data/memory",
        ollama_base_url: str = "http://localhost:11434",
        working_capacity: int = 10,
    ) -> None:
        base = Path(memory_dir)
        self.episodic = EpisodicStore(base / "episodic.jsonl")
        self.semantic = SemanticStore(base / "semantic.db")
        self.working = WorkingMemory(capacity=working_capacity)
        self.consolidator = MemoryConsolidator(self.episodic, self.semantic)
        self.ollama_base_url = ollama_base_url

    # ── write ─────────────────────────────────────────────────────────────────

    async def record(
        self,
        content: str,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an experience to working memory and episodic store."""
        self.working.add(content, importance=importance)
        embedding = await embed(content, self.ollama_base_url)
        entry = EpisodicEntry(
            content=content,
            embedding=embedding,
            importance=importance,
            metadata=metadata or {},
        )
        self.episodic.add(entry)

    # ── read ──────────────────────────────────────────────────────────────────

    async def retrieve_context(
        self,
        query: str,
        semantic_category: str | None = None,
    ) -> MemoryContext:
        """Assemble memory context relevant to the given query."""
        query_embedding = await embed(query, self.ollama_base_url)
        episodic = self.episodic.retrieve(query_embedding, k=3)
        semantic = self.semantic.retrieve(category=semantic_category, limit=5)
        working = self.working.get_items()
        return MemoryContext(episodic=episodic, semantic=semantic, working=working)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def consolidate(self, max_episodic: int = 2000) -> dict[str, int]:
        """Run sleep-cycle consolidation. Call after each experiment cycle."""
        return self.consolidator.consolidate(max_episodic=max_episodic)

    async def end_session(self) -> None:
        """Summarise working memory into episodic store, then clear it."""
        summary = self.working.summarize()
        if summary:
            await self.record(
                f"Session summary: {summary}",
                importance=0.6,
                metadata={"type": "session_summary"},
            )
        self.working.clear()

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        return {
            "episodic": len(self.episodic),
            "semantic": len(self.semantic),
            "working": len(self.working),
        }
