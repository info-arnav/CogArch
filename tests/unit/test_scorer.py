"""Unit tests for the Scorer."""

import pytest

from src.eval.scorer import Scorer


class TestScorer:
    def setup_method(self) -> None:
        self.scorer = Scorer()

    def test_exact_match_identical(self) -> None:
        assert self.scorer.exact_match("Paris", "Paris") == 1.0

    def test_exact_match_case_insensitive(self) -> None:
        assert self.scorer.exact_match("paris", "Paris") == 1.0

    def test_exact_match_whitespace(self) -> None:
        assert self.scorer.exact_match("  Paris  ", "Paris") == 1.0

    def test_exact_match_mismatch(self) -> None:
        assert self.scorer.exact_match("London", "Paris") == 0.0

    def test_fuzzy_match_identical(self) -> None:
        assert self.scorer.fuzzy_match("Paris", "Paris") == 1.0

    def test_fuzzy_match_close(self) -> None:
        score = self.scorer.fuzzy_match("Parris", "Paris")
        assert score > 0.0  # Close enough for fuzzy

    def test_fuzzy_match_distant(self) -> None:
        assert self.scorer.fuzzy_match("Tokyo", "Paris") == 0.0

    def test_fuzzy_match_custom_threshold(self) -> None:
        score = self.scorer.fuzzy_match("Par", "Paris", threshold=0.5)
        assert score > 0.0

    def test_llm_as_judge_requires_backend(self) -> None:
        with pytest.raises(ValueError, match="requires an LLMBackend"):
            import asyncio

            asyncio.run(self.scorer.llm_as_judge("q", "a", "b"))
