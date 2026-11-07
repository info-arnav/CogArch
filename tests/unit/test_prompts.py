"""Prompt building and output parsing."""

from src.inference.prompts import (
    coordinator_messages,
    coordinator_selection,
    extract_code,
    load_coordinator,
    load_specialist,
    parse_coordinator,
    parse_lessons,
    parse_probes,
    specialist_messages,
)
from src.models.code import CodeProblem, Decision, ExecutionResult, Proposal

PROBLEM = CodeProblem(
    task_id="t",
    prompt="Add two numbers.",
    entry_point="add",
    tests=["assert add(1, 2) == 3"],
)
NAMES = ["logical", "creative", "skeptical", "empathetic"]


def wrong_decision() -> Decision:
    return Decision(
        attempt_number=1,
        proposals=[],
        coordinator_prompt=[],
        coordinator_output="",
        code="def add(a, b):\n    return a - b",
        result=ExecutionResult(
            passed=False,
            tests_passed=0,
            tests_total=1,
            feedback="Your solution is wrong.",
        ),
    )


class TestPromptsNeverLeakTests:
    def test_specialist_prompt(self):
        msgs = specialist_messages(
            load_specialist("logical"), PROBLEM, previous=wrong_decision()
        )
        text = msgs[-1].content
        assert "assert" not in text
        assert "Your solution is wrong." in text and "return a - b" in text

    def test_coordinator_prompt_lists_every_proposal(self):
        proposals = [Proposal(specialist=n, code=f"# {n}") for n in NAMES]
        text = coordinator_messages(load_coordinator(), PROBLEM, proposals)[-1].content
        assert "assert" not in text
        assert all(f"### {n}" in text for n in NAMES)

    def test_memory_is_prepended(self):
        msgs = specialist_messages(
            load_specialist("logical"), PROBLEM, memory_text="## Lessons"
        )
        assert msgs[-1].content.startswith("## Lessons")


class TestParsing:
    def test_extract_code_block(self):
        assert extract_code("text\n```python\nx = 1\n```\nmore") == "x = 1"

    def test_extract_code_without_block(self):
        assert extract_code("x = 1") == "x = 1"

    def test_parse_coordinator(self):
        code, named = parse_coordinator(
            "BASED ON: Creative, logical\n```python\ndef add(a, b):\n    return a + b\n```",
            NAMES,
        )
        assert code.startswith("def add") and named == ["logical", "creative"]

    def test_coordinator_selection_roundtrip(self):
        reply = coordinator_selection(Proposal(specialist="skeptical", code="x = 1"))
        assert parse_coordinator(reply, NAMES) == ("x = 1", ["skeptical"])

    def test_parse_probes_keeps_only_calls(self):
        text = "```python\nadd(1, 2)\nadd(0, 0)\nprint('x')\nadd(1, 2)\n```"
        assert parse_probes(text, "add", 5) == ["add(1, 2)", "add(0, 0)"]

    def test_parse_lessons(self):
        text = (
            "Here:\n- Check empty input\n* Avoid recursion depth\n- Check empty input"
        )
        assert parse_lessons(text, 5) == ["Check empty input", "Avoid recursion depth"]
