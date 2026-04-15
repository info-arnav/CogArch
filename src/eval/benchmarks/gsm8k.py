"""GSM8K benchmark loader — grade school math via HuggingFace datasets."""

from __future__ import annotations

import re

from src.eval.benchmarks.base import Benchmark
from src.eval.scorer import Scorer
from src.models.benchmark import BenchmarkItem


class GSM8KBenchmark(Benchmark):
    """GSM8K: Grade School Math 8K.

    Loads from HuggingFace ``openai/gsm8k`` dataset. Each item has a
    multi-step math word problem and a numeric answer after ``####``.
    """

    def __init__(self, split: str = "test", limit: int | None = None) -> None:
        self.split = split
        self.limit = limit
        self._scorer = Scorer()

    async def load(self) -> list[BenchmarkItem]:
        """Download / cache GSM8K and convert to BenchmarkItem list."""
        from datasets import load_dataset  # type: ignore[import-untyped]

        ds = load_dataset("openai/gsm8k", "main", split=self.split)
        items: list[BenchmarkItem] = []
        for row in ds:
            answer_text: str = row["answer"]
            # Ground truth is the number after ####
            match = re.search(r"####\s*(.+)", answer_text)
            expected = match.group(1).strip() if match else answer_text.strip()
            items.append(
                BenchmarkItem(
                    question=row["question"],
                    expected_answer=expected,
                    category="math",
                    difficulty="medium",
                )
            )
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
