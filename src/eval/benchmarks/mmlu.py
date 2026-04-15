"""MMLU benchmark loader — Massive Multitask Language Understanding."""

from __future__ import annotations

from src.eval.benchmarks.base import Benchmark
from src.models.benchmark import BenchmarkItem

# Map index → letter for multiple-choice answers
_IDX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


class MMLUBenchmark(Benchmark):
    """MMLU: 57-subject multiple-choice benchmark.

    Loads from HuggingFace ``cais/mmlu`` (all subjects) or a specific subject.
    """

    def __init__(
        self,
        split: str = "test",
        subject: str | None = None,
        limit: int | None = None,
    ) -> None:
        self.split = split
        self.subject = subject
        self.limit = limit

    async def load(self) -> list[BenchmarkItem]:
        """Download / cache MMLU and convert to BenchmarkItem list."""
        from datasets import load_dataset  # type: ignore[import-untyped]

        config = self.subject if self.subject else "all"
        ds = load_dataset("cais/mmlu", config, split=self.split)

        items: list[BenchmarkItem] = []
        for row in ds:
            choices = row["choices"]
            answer_idx = int(row["answer"])
            expected_letter = _IDX_TO_LETTER.get(answer_idx, str(answer_idx))

            # Format question with multiple-choice options
            question_text = row["question"] + "\n"
            for i, choice in enumerate(choices):
                letter = _IDX_TO_LETTER.get(i, str(i))
                question_text += f"{letter}) {choice}\n"

            items.append(
                BenchmarkItem(
                    question=question_text.strip(),
                    expected_answer=expected_letter,
                    category=row.get("subject", self.subject or "general"),
                    difficulty="medium",
                )
            )
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
