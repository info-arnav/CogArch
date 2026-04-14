"""Benchmark base class and JSONL loader."""

from abc import ABC, abstractmethod

from src.models.benchmark import BenchmarkItem


class Benchmark(ABC):
    """Abstract benchmark that provides test items with ground truth."""

    @abstractmethod
    async def load(self) -> list[BenchmarkItem]:
        """Load all test items."""

    @abstractmethod
    def score(self, predicted: str, ground_truth: str) -> float:
        """Score a single prediction against ground truth. Returns 0-1."""
