"""Scorer — evaluates predicted answers against ground truth."""

from difflib import SequenceMatcher

from src.inference.backends.base import LLMBackend


class Scorer:
    """Scoring methods for benchmark evaluation."""

    def __init__(self, backend: LLMBackend | None = None) -> None:
        self.backend = backend

    def exact_match(self, predicted: str, ground_truth: str) -> float:
        """Return 1.0 if normalized strings match exactly, 0.0 otherwise."""
        return 1.0 if predicted.strip().lower() == ground_truth.strip().lower() else 0.0

    def fuzzy_match(
        self, predicted: str, ground_truth: str, threshold: float = 0.85
    ) -> float:
        """String similarity via SequenceMatcher. Returns 1.0 if above threshold."""
        ratio = SequenceMatcher(
            None, predicted.strip().lower(), ground_truth.strip().lower()
        ).ratio()
        return ratio if ratio >= threshold else 0.0

    async def llm_as_judge(
        self,
        question: str,
        predicted: str,
        ground_truth: str,
        model: str = "llama3:8b",
    ) -> float:
        """Use an LLM to judge if the predicted answer is semantically correct."""
        if self.backend is None:
            raise ValueError("LLM-as-judge requires an LLMBackend instance")

        prompt = (
            f"Question: {question}\n"
            f"Expected answer: {ground_truth}\n"
            f"Predicted answer: {predicted}\n\n"
            "Is the predicted answer semantically equivalent to the expected answer? "
            "Respond with a single number between 0.0 and 1.0 where 1.0 means "
            "completely correct and 0.0 means completely wrong."
        )
        messages = [
            {
                "role": "system",
                "content": "You are an objective evaluator. Respond with only a number.",
            },
            {"role": "user", "content": prompt},
        ]
        raw = await self.backend.generate(
            messages=messages, model=model, temperature=0.0, max_tokens=16
        )
        try:
            return max(0.0, min(1.0, float(raw.strip())))
        except ValueError:
            return 0.0
