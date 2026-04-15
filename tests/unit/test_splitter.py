"""Tests for BenchmarkSplitter — train/test split with per-cycle partitioning."""

import pytest

from src.eval.splitter import BenchmarkSplitter
from src.models.benchmark import BenchmarkItem


def _make_items(n: int) -> list[BenchmarkItem]:
    """Create n dummy benchmark items."""
    return [
        BenchmarkItem(
            question=f"Question {i}",
            expected_answer=f"Answer {i}",
            category=f"cat_{i % 3}",
        )
        for i in range(n)
    ]


class TestBenchmarkSplitter:
    def test_basic_split(self) -> None:
        items = _make_items(100)
        splitter = BenchmarkSplitter(items, test_ratio=0.2, num_cycles=5, seed=42)
        assert len(splitter.test_items) == 20
        assert splitter.total_train_items == 80
        assert splitter.num_cycles == 5

    def test_no_overlap_between_test_and_train(self) -> None:
        items = _make_items(50)
        splitter = BenchmarkSplitter(items, test_ratio=0.2, num_cycles=3, seed=42)
        test_questions = {it.question for it in splitter.test_items}
        for i in range(splitter.num_cycles):
            cycle_questions = {it.question for it in splitter.get_cycle_items(i)}
            assert test_questions.isdisjoint(
                cycle_questions
            ), f"Cycle {i} overlaps with test set"

    def test_no_overlap_between_cycles(self) -> None:
        items = _make_items(100)
        splitter = BenchmarkSplitter(items, test_ratio=0.2, num_cycles=4, seed=42)
        seen: set[str] = set()
        for i in range(splitter.num_cycles):
            cycle_questions = {it.question for it in splitter.get_cycle_items(i)}
            assert seen.isdisjoint(
                cycle_questions
            ), f"Cycle {i} reuses questions from earlier cycles"
            seen.update(cycle_questions)

    def test_all_items_accounted_for(self) -> None:
        items = _make_items(73)  # odd number
        splitter = BenchmarkSplitter(items, test_ratio=0.2, num_cycles=5, seed=42)
        all_questions: set[str] = set()
        all_questions.update(it.question for it in splitter.test_items)
        for i in range(splitter.num_cycles):
            all_questions.update(it.question for it in splitter.get_cycle_items(i))
        assert len(all_questions) == 73

    def test_deterministic_with_same_seed(self) -> None:
        items = _make_items(50)
        s1 = BenchmarkSplitter(items, test_ratio=0.2, num_cycles=3, seed=99)
        s2 = BenchmarkSplitter(items, test_ratio=0.2, num_cycles=3, seed=99)
        assert [it.question for it in s1.test_items] == [
            it.question for it in s2.test_items
        ]

    def test_different_seed_different_split(self) -> None:
        items = _make_items(50)
        s1 = BenchmarkSplitter(items, test_ratio=0.2, num_cycles=3, seed=1)
        s2 = BenchmarkSplitter(items, test_ratio=0.2, num_cycles=3, seed=2)
        assert [it.question for it in s1.test_items] != [
            it.question for it in s2.test_items
        ]

    def test_cycle_index_out_of_range(self) -> None:
        items = _make_items(20)
        splitter = BenchmarkSplitter(items, test_ratio=0.2, num_cycles=3, seed=42)
        with pytest.raises(IndexError):
            splitter.get_cycle_items(10)
        with pytest.raises(IndexError):
            splitter.get_cycle_items(-1)

    def test_empty_items_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            BenchmarkSplitter([], test_ratio=0.2, num_cycles=3)

    def test_invalid_ratio_raises(self) -> None:
        items = _make_items(10)
        with pytest.raises(ValueError, match="test_ratio"):
            BenchmarkSplitter(items, test_ratio=0.0, num_cycles=3)
        with pytest.raises(ValueError, match="test_ratio"):
            BenchmarkSplitter(items, test_ratio=1.0, num_cycles=3)

    def test_summary(self) -> None:
        items = _make_items(100)
        splitter = BenchmarkSplitter(items, test_ratio=0.2, num_cycles=5, seed=42)
        summary = splitter.summary()
        assert summary["total_items"] == 100
        assert summary["test_items"] == 20
        assert summary["train_items"] == 80
        assert summary["num_cycles"] == 5
        ipc = summary["items_per_cycle"]
        assert isinstance(ipc, list)
        assert len(ipc) == 5
        assert sum(ipc) == 80

    def test_single_cycle(self) -> None:
        items = _make_items(20)
        splitter = BenchmarkSplitter(items, test_ratio=0.2, num_cycles=1, seed=42)
        assert splitter.num_cycles == 1
        assert splitter.total_train_items == 16
        assert len(splitter.get_cycle_items(0)) == 16

    def test_many_cycles_with_small_data(self) -> None:
        """If num_cycles > train items, actual cycles should be reduced."""
        items = _make_items(10)
        splitter = BenchmarkSplitter(items, test_ratio=0.2, num_cycles=50, seed=42)
        # Should reduce cycles to at most 8 (10 - 2 test = 8 train)
        assert splitter.num_cycles <= 8
        assert splitter.total_train_items == 8
