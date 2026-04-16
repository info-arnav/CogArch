"""Tests for semantic memory store."""

from src.memory.semantic import SemanticStore


class TestSemanticStore:
    def test_add_new_entry(self, tmp_path):
        store = SemanticStore(tmp_path / "sem.db")
        entry = store.add_or_update("recursion is slow for large n", "algorithm")
        assert entry.concept == "recursion is slow for large n"
        assert entry.category == "algorithm"
        assert entry.confidence == 0.5
        assert entry.evidence_count == 1
        assert len(store) == 1

    def test_update_existing_strengthens_confidence(self, tmp_path):
        store = SemanticStore(tmp_path / "sem.db")
        store.add_or_update("use hash map", "algorithm", confidence_delta=0.1)
        entry = store.add_or_update("use hash map", "algorithm", confidence_delta=0.1)
        assert entry.evidence_count == 2
        assert entry.confidence > 0.5

    def test_confidence_capped_at_1(self, tmp_path):
        store = SemanticStore(tmp_path / "sem.db")
        for _ in range(20):
            store.add_or_update("always true", "fact", confidence_delta=0.2)
        entry = store.add_or_update("always true", "fact", confidence_delta=0.2)
        assert entry.confidence <= 1.0

    def test_retrieve_by_category(self, tmp_path):
        store = SemanticStore(tmp_path / "sem.db")
        store.add_or_update("merge sort O(n log n)", "algorithm")
        store.add_or_update("IndexError means out of bounds", "error_pattern")
        results = store.retrieve(category="algorithm")
        assert len(results) == 1
        assert "sort" in results[0].concept

    def test_retrieve_all_categories(self, tmp_path):
        store = SemanticStore(tmp_path / "sem.db")
        store.add_or_update("fact A", "cat1")
        store.add_or_update("fact B", "cat2")
        results = store.retrieve()
        assert len(results) == 2

    def test_retrieve_respects_min_confidence(self, tmp_path):
        store = SemanticStore(tmp_path / "sem.db")
        store.add_or_update("low confidence fact", "cat", confidence_delta=0.0)
        # Default confidence is 0.5; retrieve with high threshold
        results = store.retrieve(min_confidence=0.9)
        assert len(results) == 0

    def test_get_all(self, tmp_path):
        store = SemanticStore(tmp_path / "sem.db")
        store.add_or_update("a", "c1")
        store.add_or_update("b", "c2")
        store.add_or_update("c", "c3")
        all_entries = store.get_all()
        assert len(all_entries) == 3

    def test_different_categories_same_concept_are_separate(self, tmp_path):
        store = SemanticStore(tmp_path / "sem.db")
        store.add_or_update("timeout", "error_pattern")
        store.add_or_update("timeout", "success_pattern")
        assert len(store) == 2

    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "sem.db"
        store1 = SemanticStore(path)
        store1.add_or_update("remembered fact", "general")

        store2 = SemanticStore(path)
        results = store2.retrieve()
        assert len(results) == 1
        assert results[0].concept == "remembered fact"
