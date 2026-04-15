"""MMLU benchmark loader — Massive Multitask Language Understanding (14,042 items)."""

from __future__ import annotations

from pathlib import Path

from src.eval.benchmarks.jsonl_benchmark import JsonlBenchmark
from src.models.benchmark import BenchmarkItem

_DEFAULT_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "benchmarks" / "mmlu.jsonl"
)


class MMLUBenchmark(JsonlBenchmark):
    """MMLU: 57-subject multiple-choice benchmark.

    Reads pre-extracted items from ``data/benchmarks/mmlu.jsonl``.
    Questions already include formatted A/B/C/D choices.
    """

    def __init__(
        self,
        path: str | Path = _DEFAULT_PATH,
        subject: str | None = None,
        limit: int | None = None,
    ) -> None:
        super().__init__(path=path, metric="exact_match", name="mmlu")
        self.subject = subject
        self.limit = limit

    async def load(self) -> list[BenchmarkItem]:
        """Load MMLU items from local JSONL, optionally filtering by subject."""
        items = await super().load()
        if self.subject:
            items = [it for it in items if it.category == self.subject]
        if self.limit:
            items = items[: self.limit]
        return items

    def score(self, predicted: str, ground_truth: str) -> float:
        """Check if the predicted letter matches ground truth."""
        pred = predicted.strip().upper()
        gt = ground_truth.strip().upper()
        # Accept just the letter or "A)" or "(A)" formats
        for fmt in [pred, pred.rstrip(")"), pred.strip("()")]:
            if fmt and fmt[0] == gt:
                return 1.0
        return 0.0
