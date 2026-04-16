"""Tests for memory consolidator (sleep-cycle pattern extraction)."""

from src.memory.consolidator import MemoryConsolidator
from src.memory.episodic import EpisodicStore
from src.memory.models import EpisodicEntry
from src.memory.semantic import SemanticStore


def _entry(
    content: str = "test",
    importance: float = 0.5,
    error: str = "",
    specialist: str = "logical",
    passed_all: bool = False,
    timed_out: bool = False,
    problem_type: str = "general",
) -> EpisodicEntry:
    return EpisodicEntry(
        content=content,
        importance=importance,
        metadata={
            "error": error,
            "specialist": specialist,
            "passed_all": passed_all,
            "timed_out": timed_out,
            "problem_type": problem_type,
        },
    )


class TestMemoryConsolidator:
    def test_promotes_recurring_error_pattern(self, tmp_path):
        ep = EpisodicStore(tmp_path / "ep.jsonl")
        sem = SemanticStore(tmp_path / "sem.db")
        consolidator = MemoryConsolidator(ep, sem)

        # Same error type appears 4 times (above threshold of 3)
        entries = [
            _entry(error="IndexError: list index out of range") for _ in range(4)
        ]
        stats = consolidator.consolidate(recent_entries=entries)

        assert stats["promoted"] >= 1
        semantic_entries = sem.retrieve(category="error_pattern")
        assert len(semantic_entries) >= 1
        assert any("IndexError" in e.concept for e in semantic_entries)

    def test_promotes_recurring_success_pattern(self, tmp_path):
        ep = EpisodicStore(tmp_path / "ep.jsonl")
        sem = SemanticStore(tmp_path / "sem.db")
        consolidator = MemoryConsolidator(ep, sem)

        entries = [
            _entry(specialist="logical", passed_all=True, problem_type="sorting")
            for _ in range(4)
        ]
        stats = consolidator.consolidate(recent_entries=entries)

        assert stats["promoted"] >= 1
        semantic_entries = sem.retrieve(category="success_pattern")
        assert any("logical" in e.concept for e in semantic_entries)

    def test_below_threshold_not_promoted(self, tmp_path):
        ep = EpisodicStore(tmp_path / "ep.jsonl")
        sem = SemanticStore(tmp_path / "sem.db")
        consolidator = MemoryConsolidator(ep, sem)

        # Only 2 occurrences — below threshold of 3
        entries = [_entry(error="ValueError: invalid") for _ in range(2)]
        consolidator.consolidate(recent_entries=entries)

        assert len(sem) == 0

    def test_promotes_timeout_pattern(self, tmp_path):
        ep = EpisodicStore(tmp_path / "ep.jsonl")
        sem = SemanticStore(tmp_path / "sem.db")
        consolidator = MemoryConsolidator(ep, sem)

        entries = [_entry(timed_out=True) for _ in range(4)]
        consolidator.consolidate(recent_entries=entries)

        semantic = sem.retrieve(category="error_pattern")
        assert any(
            "time out" in e.concept.lower() or "loop" in e.concept.lower()
            for e in semantic
        )

    def test_prune_called_on_episodic(self, tmp_path):
        ep = EpisodicStore(tmp_path / "ep.jsonl")
        sem = SemanticStore(tmp_path / "sem.db")
        consolidator = MemoryConsolidator(ep, sem)

        for i in range(10):
            ep.add(EpisodicEntry(content=f"entry {i}", importance=0.1))

        stats = consolidator.consolidate(recent_entries=[], max_episodic=5)
        assert stats["pruned"] == 5
        assert len(ep) == 5

    def test_stats_returned(self, tmp_path):
        ep = EpisodicStore(tmp_path / "ep.jsonl")
        sem = SemanticStore(tmp_path / "sem.db")
        consolidator = MemoryConsolidator(ep, sem)

        stats = consolidator.consolidate(recent_entries=[])
        assert "promoted" in stats
        assert "pruned" in stats
        assert "total_semantic" in stats
        assert "total_episodic" in stats
