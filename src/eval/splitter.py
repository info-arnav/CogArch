"""Benchmark splitter — splits benchmark items into train/test and per-cycle chunks."""

from __future__ import annotations

import random

from src.models.benchmark import BenchmarkItem


class BenchmarkSplitter:
    """Deterministically splits benchmark items into train/test sets
    and partitions the train set into per-cycle chunks with no overlap.

    Usage::

        splitter = BenchmarkSplitter(items, test_ratio=0.2, num_cycles=5, seed=42)
        test_set = splitter.test_items
        cycle_3_items = splitter.get_cycle_items(3)  # 0-indexed
    """

    def __init__(
        self,
        items: list[BenchmarkItem],
        test_ratio: float = 0.2,
        num_cycles: int = 5,
        seed: int = 42,
    ) -> None:
        if not items:
            raise ValueError("Cannot split an empty item list")
        if not 0.0 < test_ratio < 1.0:
            raise ValueError(f"test_ratio must be in (0, 1), got {test_ratio}")
        if num_cycles < 1:
            raise ValueError(f"num_cycles must be >= 1, got {num_cycles}")

        self.num_cycles = num_cycles

        # Deterministic shuffle
        shuffled = list(items)
        rng = random.Random(seed)
        rng.shuffle(shuffled)

        # Split into test and train
        test_size = max(1, int(len(shuffled) * test_ratio))
        self.test_items: list[BenchmarkItem] = shuffled[:test_size]
        train_items: list[BenchmarkItem] = shuffled[test_size:]

        # Partition train items into num_cycles chunks (roughly equal)
        self._cycle_chunks: list[list[BenchmarkItem]] = []
        chunk_size = max(1, len(train_items) // num_cycles)
        for i in range(num_cycles):
            start = i * chunk_size
            # Last chunk takes any remainder
            if i == num_cycles - 1:
                chunk = train_items[start:]
            else:
                chunk = train_items[start : start + chunk_size]
            if chunk:
                self._cycle_chunks.append(chunk)

        # Adjust num_cycles if we have fewer chunks than requested
        self.num_cycles = len(self._cycle_chunks)

    def get_cycle_items(self, cycle_index: int) -> list[BenchmarkItem]:
        """Return the training items for a specific cycle (0-indexed)."""
        if cycle_index < 0 or cycle_index >= self.num_cycles:
            raise IndexError(
                f"cycle_index {cycle_index} out of range [0, {self.num_cycles})"
            )
        return self._cycle_chunks[cycle_index]

    @property
    def total_train_items(self) -> int:
        return sum(len(c) for c in self._cycle_chunks)

    def summary(self) -> dict[str, int | list[int]]:
        """Return a summary of the split."""
        return {
            "total_items": len(self.test_items) + self.total_train_items,
            "test_items": len(self.test_items),
            "train_items": self.total_train_items,
            "num_cycles": self.num_cycles,
            "items_per_cycle": [len(c) for c in self._cycle_chunks],
        }
