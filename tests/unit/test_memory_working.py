"""Tests for working memory."""

from src.memory.working import WorkingMemory


class TestWorkingMemory:
    def test_add_and_get(self):
        wm = WorkingMemory(capacity=5)
        wm.add("item one", importance=0.8)
        items = wm.get_items()
        assert len(items) == 1
        assert items[0].content == "item one"
        assert items[0].importance == 0.8

    def test_capacity_evicts_oldest(self):
        wm = WorkingMemory(capacity=3)
        wm.add("first")
        wm.add("second")
        wm.add("third")
        wm.add("fourth")  # evicts "first"
        items = wm.get_items()
        assert len(items) == 3
        assert all(i.content != "first" for i in items)
        assert items[-1].content == "fourth"

    def test_get_context_empty(self):
        wm = WorkingMemory()
        assert wm.get_context() == ""

    def test_get_context_nonempty(self):
        wm = WorkingMemory()
        wm.add("solved fibonacci")
        ctx = wm.get_context()
        assert "solved fibonacci" in ctx

    def test_summarize_empty(self):
        wm = WorkingMemory()
        assert wm.summarize() == ""

    def test_summarize_prefers_high_importance(self):
        wm = WorkingMemory(capacity=10)
        wm.add("low priority", importance=0.2)
        wm.add("critical insight", importance=0.9)
        summary = wm.summarize()
        assert "critical insight" in summary

    def test_summarize_fallback_to_all(self):
        wm = WorkingMemory()
        wm.add("only item", importance=0.3)
        summary = wm.summarize()
        assert "only item" in summary

    def test_clear_returns_items_and_empties(self):
        wm = WorkingMemory()
        wm.add("a")
        wm.add("b")
        evicted = wm.clear()
        assert len(evicted) == 2
        assert len(wm) == 0

    def test_len(self):
        wm = WorkingMemory(capacity=10)
        assert len(wm) == 0
        wm.add("x")
        wm.add("y")
        assert len(wm) == 2
