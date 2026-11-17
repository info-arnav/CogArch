"""Tests for memory data models."""

from src.memory.models import EpisodicEntry, MemoryContext


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


class TestMemoryContext:
    def test_empty(self):
        assert MemoryContext().to_prompt_text() == ""

    def test_with_episodic(self):
        ctx = MemoryContext(
            episodic=[EpisodicEntry(content="solved two-sum using dict")]
        )
        text = ctx.to_prompt_text()
        assert "two-sum" in text
        assert "Similar problems you have seen" in text
