"""Answer extraction and judging for code, maths and multiple choice."""

from src.config import FeedbackLevel
from src.environment import (
    Environment,
    answers_match,
    extract_answer,
    extract_choice,
    extract_math,
)
from src.execution.sandbox import Sandbox


class TestExtraction:
    def test_math_boxed_with_nested_braces(self):
        assert extract_math("so \\boxed{\\frac{1}{2}} done") == "\\frac{1}{2}"

    def test_math_last_boxed_wins(self):
        assert extract_math("\\boxed{3} no wait \\boxed{4}") == "4"

    def test_math_final_answer_line(self):
        assert extract_math("work...\nFINAL ANSWER: $18$.") == "18"

    def test_math_nothing(self):
        assert extract_math("I am not sure") == ""

    def test_choice_final_answer(self):
        assert extract_choice("A looks wrong.\nFINAL ANSWER: (C)", 4) == "C"

    def test_choice_letter_outside_range_ignored(self):
        assert extract_choice("FINAL ANSWER: D", 3) == ""

    def test_choice_fallback_to_last_letter(self):
        assert extract_choice("I pick option B", 3) == "B"

    def test_code(self, add_task):
        assert extract_answer(add_task, "Here:\n```python\nx = 1\n```") == "x = 1"


class TestMatching:
    def test_math_equivalent_forms(self, math_task):
        assert answers_match(math_task, "42", "42.0")
        assert answers_match(math_task, "\\frac{1}{2}", "0.5")
        assert not answers_match(math_task, "42", "41")
        assert not answers_match(math_task, "42", "")

    def test_choice(self, choice_task):
        assert answers_match(choice_task, "B", "B")
        assert not answers_match(choice_task, "B", "")


class TestJudge:
    def env(self) -> Environment:
        return Environment(
            Sandbox(FeedbackLevel.BINARY, timeout_seconds=5), FeedbackLevel.BINARY
        )

    async def test_math_only_told_wrong(self, math_task):
        verdict = await self.env().judge(math_task, "41")
        assert not verdict.passed and verdict.feedback == "Your answer is wrong."
        assert "42" not in verdict.feedback

    async def test_choice_right(self, choice_task):
        assert (await self.env().judge(choice_task, "B")).passed

    async def test_code_runs_hidden_tests(self, add_task):
        assert (
            await self.env().judge(add_task, "def add(a, b):\n    return a + b")
        ).passed
        verdict = await self.env().judge(add_task, "def add(a, b):\n    return a - b")
        assert not verdict.passed and verdict.feedback == "Your solution is wrong."
