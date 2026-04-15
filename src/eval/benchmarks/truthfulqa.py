"""TruthfulQA benchmark loader — tests for truthful, non-hallucinated answers."""

from __future__ import annotations

from src.eval.benchmarks.base import Benchmark
from src.eval.scorer import Scorer
from src.models.benchmark import BenchmarkItem


class TruthfulQABenchmark(Benchmark):
    """TruthfulQA: 817-question benchmark for truthfulness.

    Loads the ``generation`` config from HuggingFace ``truthfulqa/truthful_qa``.
    Scoring uses fuzzy matching against the best correct answer.
    """

    def __init__(self, limit: int | None = None) -> None:
        self.limit = limit
        self._scorer = Scorer()

    async def load(self) -> list[BenchmarkItem]:
        """Download / cache TruthfulQA and convert to BenchmarkItem list."""
        from datasets import load_dataset  # type: ignore[import-untyped]

        ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")

        items: list[BenchmarkItem] = []
        for row in ds:
            # best_answer is the top correct answer
            best = row.get("best_answer", "")
            if not best:
                correct_answers = row.get("correct_answers", [])
                best = correct_answers[0] if correct_answers else ""
            if not best:
                continue

            items.append(
                BenchmarkItem(
                    question=row["question"],
                    expected_answer=best,
                    category=row.get("category", "general"),
                    difficulty="hard",
                )
            )
        if self.limit:
            items = items[: self.limit]
        return items

    def score(self, predicted: str, ground_truth: str) -> float:
        """Fuzzy match — TruthfulQA answers are free-form text."""
        return self._scorer.fuzzy_match(predicted, ground_truth, threshold=0.6)
