"""Sleep-cycle memory consolidation.

Reads recent episodic entries, extracts recurring patterns, promotes
them to semantic memory, and prunes low-value episodic entries.

No LLM required — patterns are extracted via simple heuristics so the
consolidator works even when Ollama is offline.
"""

from __future__ import annotations

from collections import defaultdict

from src.memory.episodic import EpisodicStore
from src.memory.models import EpisodicEntry
from src.memory.semantic import SemanticStore


class MemoryConsolidator:
    """Runs at the end of each experiment cycle (the 'sleep' phase)."""

    # Minimum times a pattern must appear before it is promoted
    PROMOTION_THRESHOLD = 3

    def __init__(self, episodic: EpisodicStore, semantic: SemanticStore) -> None:
        self.episodic = episodic
        self.semantic = semantic

    def consolidate(
        self,
        recent_entries: list[EpisodicEntry] | None = None,
        max_episodic: int = 2000,
    ) -> dict[str, int]:
        """Run a full consolidation pass.

        Args:
            recent_entries: Entries to analyse (defaults to last 100 from store).
            max_episodic: Prune episodic store to this size after promotion.

        Returns:
            Stats dict with keys: promoted, pruned, total_semantic, total_episodic.
        """
        entries = recent_entries or self.episodic.get_recent(n=100)
        promoted = self._extract_patterns(entries)
        pruned = self.episodic.prune(max_entries=max_episodic)
        return {
            "promoted": promoted,
            "pruned": pruned,
            "total_semantic": len(self.semantic),
            "total_episodic": len(self.episodic),
        }

    # ── pattern extraction ────────────────────────────────────────────────────

    def _extract_patterns(self, entries: list[EpisodicEntry]) -> int:
        """Tally patterns across entries; promote frequent ones to semantic."""
        error_counts: dict[str, int] = defaultdict(int)
        success_counts: dict[str, int] = defaultdict(int)
        timeout_count = 0

        for entry in entries:
            meta = entry.metadata

            # Error patterns — key on the exception type / first line
            error = str(meta.get("error", "")).strip()
            if error:
                error_key = error.split("\n")[-1].split(":")[0].strip()[:60]
                if error_key:
                    error_counts[error_key] += 1

            # Success patterns — which specialist style solved which problem type
            if meta.get("passed_all"):
                specialist = str(meta.get("specialist", "")).strip()
                ptype = str(meta.get("problem_type", "general")).strip()
                if specialist:
                    success_counts[f"{specialist} succeeded on {ptype}"] += 1

            # Timeout patterns
            if meta.get("timed_out"):
                timeout_count += 1

        promoted = 0

        for error_key, count in error_counts.items():
            if count >= self.PROMOTION_THRESHOLD:
                self.semantic.add_or_update(
                    concept=f"Recurring error: {error_key}",
                    category="error_pattern",
                    confidence_delta=min(0.4, count * 0.05),
                )
                promoted += 1

        for pattern, count in success_counts.items():
            if count >= self.PROMOTION_THRESHOLD:
                self.semantic.add_or_update(
                    concept=pattern,
                    category="success_pattern",
                    confidence_delta=min(0.4, count * 0.05),
                )
                promoted += 1

        if timeout_count >= self.PROMOTION_THRESHOLD:
            self.semantic.add_or_update(
                concept="Solutions frequently time out — check for infinite loops",
                category="error_pattern",
                confidence_delta=min(0.4, timeout_count * 0.05),
            )
            promoted += 1

        return promoted
