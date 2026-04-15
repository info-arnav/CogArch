"""TruthfulQA benchmark loader — tests for truthful, non-hallucinated answers (817 items)."""

from __future__ import annotations

from pathlib import Path

from src.eval.benchmarks.jsonl_benchmark import JsonlBenchmark
from src.eval.scorer import Scorer
from src.models.benchmark import BenchmarkItem

_DEFAULT_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "benchmarks" / "truthfulqa.jsonl"
)


class TruthfulQABenchmark(JsonlBenchmark):
    """TruthfulQA: 817-question benchmark for truthfulness.

    Reads pre-extracted items from ``data/benchmarks/truthfulqa.jsonl``.
    Scoring uses fuzzy matching against the best correct answer.
    """

    def __init__(
        self, path: str | Path = _DEFAULT_PATH, limit: int | None = None
    ) -> None:
        super().__init__(path=path, metric="fuzzy_match", name="truthfulqa")
        self.limit = limit
        self._scorer = Scorer()

    async def load(self) -> list[BenchmarkItem]:
        """Load TruthfulQA items from local JSONL."""
        items = await super().load()
        if self.limit:
            items = items[: self.limit]
        return items

    def score(self, predicted: str, ground_truth: str) -> float:
        """Fuzzy match — TruthfulQA answers are free-form text."""
        return self._scorer.fuzzy_match(predicted, ground_truth, threshold=0.6)
