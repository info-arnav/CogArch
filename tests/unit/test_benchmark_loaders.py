"""Tests for benchmark loaders — GSM8K, MMLU, TruthfulQA scoring logic."""

from src.eval.benchmarks.gsm8k import GSM8KBenchmark
from src.eval.benchmarks.mmlu import MMLUBenchmark
from src.eval.benchmarks.truthfulqa import TruthfulQABenchmark


class TestGSM8KScoring:
    """Test GSM8K number extraction and scoring."""

    def setup_method(self) -> None:
        self.bench = GSM8KBenchmark()

    def test_exact_number_match(self) -> None:
        assert self.bench.score("42", "42") == 1.0

    def test_number_in_sentence(self) -> None:
        assert self.bench.score("The answer is 42.", "42") == 1.0

    def test_number_with_commas(self) -> None:
        assert self.bench.score("1,234", "1234") == 1.0

    def test_number_mismatch(self) -> None:
        assert self.bench.score("41", "42") == 0.0

    def test_decimal_match(self) -> None:
        assert self.bench.score("3.14", "3.14") == 1.0

    def test_negative_number(self) -> None:
        assert self.bench.score("The result is -5", "-5") == 1.0

    def test_extract_last_number(self) -> None:
        # Should extract the last number (typical chain-of-thought)
        assert self.bench.score("Step 1: 10, Step 2: 20, Final: 30", "30") == 1.0

    def test_no_number_falls_back_to_fuzzy(self) -> None:
        score = self.bench.score("no numbers here", "42")
        assert score == 0.0  # fuzzy won't match


class TestMMLUScoring:
    """Test MMLU multiple-choice letter scoring."""

    def setup_method(self) -> None:
        self.bench = MMLUBenchmark()

    def test_exact_letter(self) -> None:
        assert self.bench.score("A", "A") == 1.0

    def test_lowercase_letter(self) -> None:
        assert self.bench.score("b", "B") == 1.0

    def test_letter_with_paren(self) -> None:
        assert self.bench.score("C)", "C") == 1.0

    def test_letter_in_parens(self) -> None:
        assert self.bench.score("(D)", "D") == 1.0

    def test_wrong_letter(self) -> None:
        assert self.bench.score("A", "B") == 0.0

    def test_answer_with_explanation(self) -> None:
        # First char after strip should be the letter
        assert self.bench.score("B) because of X", "B") == 1.0

    def test_empty_prediction(self) -> None:
        assert self.bench.score("", "A") == 0.0


class TestTruthfulQAScoring:
    """Test TruthfulQA fuzzy matching."""

    def setup_method(self) -> None:
        self.bench = TruthfulQABenchmark()

    def test_exact_match(self) -> None:
        assert self.bench.score("The earth is round", "The earth is round") == 1.0

    def test_close_match(self) -> None:
        score = self.bench.score("The Earth is round.", "The earth is round")
        assert score > 0.0  # fuzzy threshold is 0.6, very close strings

    def test_completely_different(self) -> None:
        assert self.bench.score("xyz abc", "The earth is round") == 0.0
