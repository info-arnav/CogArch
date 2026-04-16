"""Episodic memory — JSONL persistence with cosine-similarity retrieval.

Every experience (problem solved, error encountered, approach tried) is
stored here permanently. Pruning removes the lowest-importance oldest
entries when the store exceeds max_entries.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.memory.embedder import cosine_similarity, is_zero_vector
from src.memory.models import EpisodicEntry


class EpisodicStore:
    def __init__(self, path: Path | str = "data/memory/episodic.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: list[EpisodicEntry] = self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> list[EpisodicEntry]:
        if not self.path.exists():
            return []
        entries: list[EpisodicEntry] = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(EpisodicEntry.model_validate_json(line))
                    except Exception:
                        continue
        return entries

    def _append(self, entry: EpisodicEntry) -> None:
        with open(self.path, "a") as f:
            f.write(entry.model_dump_json() + "\n")

    def _rewrite(self) -> None:
        with open(self.path, "w") as f:
            for entry in self._cache:
                f.write(entry.model_dump_json() + "\n")

    # ── public API ────────────────────────────────────────────────────────────

    def add(self, entry: EpisodicEntry) -> None:
        self._cache.append(entry)
        self._append(entry)

    def retrieve(
        self,
        query_embedding: list[float],
        k: int = 3,
        min_similarity: float = 0.4,
    ) -> list[EpisodicEntry]:
        """Return top-K most similar entries.

        Falls back to recency order if the query is a zero vector
        (i.e. embedding unavailable).
        """
        if is_zero_vector(query_embedding):
            return sorted(self._cache, key=lambda e: e.timestamp, reverse=True)[:k]

        scored = [
            (cosine_similarity(query_embedding, e.embedding), e)
            for e in self._cache
            if e.embedding and not is_zero_vector(e.embedding)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [e for sim, e in scored if sim >= min_similarity][:k]

        now = datetime.now(timezone.utc).isoformat()
        for entry in results:
            entry.recall_count += 1
            entry.last_recalled = now

        return results

    def get_recent(self, n: int = 50) -> list[EpisodicEntry]:
        return sorted(self._cache, key=lambda e: e.timestamp, reverse=True)[:n]

    def prune(self, max_entries: int = 2000) -> int:
        """Remove lowest-value entries down to max_entries. Returns count pruned."""
        if len(self._cache) <= max_entries:
            return 0

        def _score(e: EpisodicEntry) -> float:
            recency = 0.1 if e.last_recalled else 0.0
            return e.importance + recency + e.recall_count * 0.05

        self._cache.sort(key=_score, reverse=True)
        pruned = len(self._cache) - max_entries
        self._cache = self._cache[:max_entries]
        self._rewrite()
        return pruned

    def __len__(self) -> int:
        return len(self._cache)
