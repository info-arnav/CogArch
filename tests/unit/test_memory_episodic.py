"""Tests for episodic memory store."""

import pytest

from src.memory.episodic import EpisodicStore
from src.memory.models import EpisodicEntry


def _entry(
    content: str, importance: float = 0.5, embedding: list | None = None
) -> EpisodicEntry:
    return EpisodicEntry(
        content=content, importance=importance, embedding=embedding or []
    )


class TestEpisodicStore:
    def test_add_and_len(self, tmp_path):
        store = EpisodicStore(tmp_path / "ep.jsonl")
        store.add(_entry("experience one"))
        store.add(_entry("experience two"))
        assert len(store) == 2

    def test_persists_to_disk(self, tmp_path):
        path = tmp_path / "ep.jsonl"
        store = EpisodicStore(path)
        store.add(_entry("remembered"))

        store2 = EpisodicStore(path)
        assert len(store2) == 1
        assert store2.get_recent(1)[0].content == "remembered"

    def test_get_recent_order(self, tmp_path):
        store = EpisodicStore(tmp_path / "ep.jsonl")
        store.add(_entry("first"))
        store.add(_entry("second"))
        store.add(_entry("third"))
        recent = store.get_recent(2)
        # Most recent should be first
        assert recent[0].content == "third"
        assert recent[1].content == "second"

    def test_retrieve_by_similarity(self, tmp_path):
        store = EpisodicStore(tmp_path / "ep.jsonl")
        # Vectors: [1,0,0], [0,1,0], [0,0,1]
        store.add(_entry("apple", embedding=[1.0, 0.0, 0.0]))
        store.add(_entry("banana", embedding=[0.0, 1.0, 0.0]))
        store.add(_entry("cherry", embedding=[0.0, 0.0, 1.0]))

        # Query close to "apple"
        results = store.retrieve([0.99, 0.01, 0.0], k=1, min_similarity=0.5)
        assert len(results) == 1
        assert results[0].content == "apple"

    def test_retrieve_zero_vector_falls_back_to_recency(self, tmp_path):
        store = EpisodicStore(tmp_path / "ep.jsonl")
        store.add(_entry("older"))
        store.add(_entry("newer"))
        results = store.retrieve([0.0, 0.0, 0.0], k=1)
        assert results[0].content == "newer"

    def test_retrieve_increments_recall_count(self, tmp_path):
        store = EpisodicStore(tmp_path / "ep.jsonl")
        store.add(_entry("target", embedding=[1.0, 0.0]))
        results = store.retrieve([1.0, 0.0], k=1, min_similarity=0.5)
        assert results[0].recall_count == 1

    def test_prune_keeps_high_importance(self, tmp_path):
        store = EpisodicStore(tmp_path / "ep.jsonl")
        for i in range(10):
            importance = 0.9 if i < 3 else 0.1
            store.add(_entry(f"item-{i}", importance=importance))
        pruned = store.prune(max_entries=3)
        assert pruned == 7
        remaining = store.get_recent(10)
        # All retained entries should be the high-importance ones
        for entry in remaining:
            assert entry.importance == pytest.approx(0.9)

    def test_prune_no_op_when_under_limit(self, tmp_path):
        store = EpisodicStore(tmp_path / "ep.jsonl")
        store.add(_entry("a"))
        store.add(_entry("b"))
        pruned = store.prune(max_entries=100)
        assert pruned == 0
        assert len(store) == 2

    def test_corrupted_line_skipped(self, tmp_path):
        path = tmp_path / "ep.jsonl"
        path.write_text(
            '{"bad json"\n' + EpisodicEntry(content="good").model_dump_json() + "\n"
        )
        store = EpisodicStore(path)
        assert len(store) == 1
        assert store.get_recent(1)[0].content == "good"
