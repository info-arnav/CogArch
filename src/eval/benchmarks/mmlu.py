"""MMLU benchmark loader — Massive Multitask Language Understanding (14,042 items)."""

from __future__ import annotations

import re
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
        """Check if the predicted letter matches ground truth.

        Handles verbose model responses like "The answer is B" or "I'd choose option C".
        Priority order:
          1. Direct match at start of response ("B", "B)", "(B)", "B. because...")
          2. Explicit answer indicator ("the answer is B", "answer: B", "option B")
          3. Any standalone A/B/C/D letter at a word boundary (last occurrence wins)
        """
        pred = predicted.strip().upper()
        gt = ground_truth.strip().upper()

        # 1. Direct match at start
        for fmt in [pred, pred.rstrip(")"), pred.strip("()")]:
            if fmt and fmt[0] == gt:
                return 1.0

        # 2. Explicit answer indicator — most reliable for verbose responses
        explicit = re.findall(
            r"(?:answer(?:s| is| are)?|option|choice|select(?:ion)?|correct(?:ly)?)"
            r"[:\s]+([ABCD])\b",
            pred,
        )
        if explicit and explicit[-1] == gt:
            return 1.0

        # 3. Any standalone letter — last occurrence is usually the conclusion
        standalone = re.findall(r"\b([ABCD])\b", pred)
        if standalone and standalone[-1] == gt:
            return 1.0

        return 0.0
