"""GSM8K benchmark loader — grade school math (1,319 items)."""

from __future__ import annotations

import re
from pathlib import Path

from src.eval.benchmarks.jsonl_benchmark import JsonlBenchmark
from src.eval.scorer import Scorer
from src.models.benchmark import BenchmarkItem

_DEFAULT_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "benchmarks" / "gsm8k.jsonl"
)


class GSM8KBenchmark(JsonlBenchmark):
    """GSM8K: Grade School Math 8K.

    Reads pre-extracted items from ``data/benchmarks/gsm8k.jsonl``.
    Each item is a multi-step math word problem with a numeric answer.
    """

    def __init__(
        self, path: str | Path = _DEFAULT_PATH, limit: int | None = None
    ) -> None:
        super().__init__(path=path, metric="exact_match", name="gsm8k")
        self.limit = limit
        self._scorer = Scorer()

    async def load(self) -> list[BenchmarkItem]:
        """Load GSM8K items from local JSONL."""
        items = await super().load()
        if self.limit:
            items = items[: self.limit]
        return items

    def score(self, predicted: str, ground_truth: str) -> float:
        """Extract final number from predicted and compare to ground truth."""
        pred_num = self._extract_number(predicted)
        gt_num = self._extract_number(ground_truth)
        if pred_num is not None and gt_num is not None:
            return 1.0 if abs(pred_num - gt_num) < 1e-6 else 0.0
        # Fallback to fuzzy match on the raw strings
        return self._scorer.fuzzy_match(predicted, ground_truth)

    @staticmethod
    def _extract_number(text: str) -> float | None:
        """Pull the last number from a string (ignoring commas)."""
        # Look for numbers, possibly with commas and decimals
        matches = re.findall(r"-?[\d,]+\.?\d*", text.replace(",", ""))
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                return None
        return None
