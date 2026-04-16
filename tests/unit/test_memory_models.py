"""Tests for memory data models."""

from src.memory.models import (
    EpisodicEntry,
    MemoryContext,
    SemanticEntry,
    WorkingMemoryItem,
)


class TestEpisodicEntry:
    def test_defaults(self):
        e = EpisodicEntry(content="test experience")
        assert e.id != ""
        assert e.timestamp != ""
        assert e.importance == 0.5
        assert e.recall_count == 0
        assert e.last_recalled is None
        assert e.embedding == []
        assert e.metadata == {}

    def test_custom_fields(self):
        e = EpisodicEntry(content="x", importance=0.9, metadata={"k": "v"})
        assert e.importance == 0.9
        assert e.metadata["k"] == "v"

    def test_unique_ids(self):
        a = EpisodicEntry(content="a")
        b = EpisodicEntry(content="b")
        assert a.id != b.id

    def test_json_roundtrip(self):
        e = EpisodicEntry(content="hello", importance=0.8, embedding=[0.1, 0.2])
        restored = EpisodicEntry.model_validate_json(e.model_dump_json())
        assert restored.content == "hello"
        assert restored.importance == 0.8
        assert restored.embedding == [0.1, 0.2]


class TestSemanticEntry:
    def test_defaults(self):
        s = SemanticEntry(concept="recursion is slow for large n", category="algorithm")
        assert s.confidence == 0.5
        assert s.evidence_count == 1

    def test_unique_ids(self):
        a = SemanticEntry(concept="a", category="c")
        b = SemanticEntry(concept="b", category="c")
        assert a.id != b.id


class TestWorkingMemoryItem:
    def test_defaults(self):
        item = WorkingMemoryItem(content="currently solving sort problem")
        assert item.importance == 0.5
        assert item.source == ""
        assert item.timestamp != ""


class TestMemoryContext:
    def test_empty(self):
        ctx = MemoryContext()
        assert ctx.is_empty()
        assert ctx.to_prompt_text() == ""

    def test_with_working_only(self):
        ctx = MemoryContext(working=[WorkingMemoryItem(content="solved two-sum")])
        assert not ctx.is_empty()
        text = ctx.to_prompt_text()
        assert "solved two-sum" in text
        assert "Recent session context" in text

    def test_with_semantic(self):
        ctx = MemoryContext(
            semantic=[
                SemanticEntry(concept="use hash map for O(n)", category="algorithm")
            ]
        )
        text = ctx.to_prompt_text()
        assert "use hash map" in text
        assert "Relevant knowledge" in text

    def test_with_episodic(self):
        ctx = MemoryContext(
            episodic=[EpisodicEntry(content="solved two-sum using dict")]
        )
        text = ctx.to_prompt_text()
        assert "two-sum" in text
        assert "Similar past experiences" in text

    def test_all_tiers_combined(self):
        ctx = MemoryContext(
            working=[WorkingMemoryItem(content="working on sort")],
            semantic=[
                SemanticEntry(concept="quicksort is O(n log n)", category="algo")
            ],
            episodic=[EpisodicEntry(content="solved merge sort yesterday")],
        )
        text = ctx.to_prompt_text()
        assert "working on sort" in text
        assert "quicksort" in text
        assert "merge sort" in text
