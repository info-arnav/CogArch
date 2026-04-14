"""JSONL benchmark loader — loads benchmark items from a JSONL file.

Each line should be a JSON object with at least "question" and "expected_answer" fields.
Optional: "category", "difficulty".
"""

import json
from pathlib import Path

from src.eval.benchmarks.base import Benchmark
from src.eval.scorer import Scorer
from src.models.benchmark import BenchmarkItem


class JsonlBenchmark(Benchmark):
    """Benchmark loaded from a JSONL file on disk."""

    def __init__(
        self,
        path: str | Path,
        metric: str = "exact_match",
        name: str = "",
    ) -> None:
        self.path = Path(path)
        self.metric = metric
        self.name = name or self.path.stem
        self._scorer = Scorer()

    async def load(self) -> list[BenchmarkItem]:
        """Load benchmark items from JSONL file."""
        items: list[BenchmarkItem] = []
        if not self.path.exists():
            return items
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    items.append(BenchmarkItem(**data))
        return items

    def score(self, predicted: str, ground_truth: str) -> float:
        """Score using configured metric."""
        if self.metric == "fuzzy_match":
            return self._scorer.fuzzy_match(predicted, ground_truth)
        return self._scorer.exact_match(predicted, ground_truth)
