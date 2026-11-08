"""Prompt building and output parsing."""

from src.inference.prompts import (
    coordinator_messages,
    coordinator_selection,
    extract_code,
    load_coordinator,
    load_specialist,
    parse_coordinator,
    parse_probes,
    parse_reflection,
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

    def test_choose_submits_proposal_unchanged(self):
        proposals = [Proposal(specialist=n, code=f"# {n}") for n in NAMES]
        assert parse_coordinator("CHOOSE: Skeptical", proposals) == (
            "# skeptical",
            "skeptical",
        )
        assert parse_coordinator("**CHOOSE:** creative", proposals) == (
            "# creative",
            "creative",
        )

    def test_combine_submits_own_code(self):
        proposals = [Proposal(specialist=n, code=f"# {n}") for n in NAMES]
        reply = "COMBINE:\n```python\nx = 1\n```"
        assert parse_coordinator(reply, proposals) == ("x = 1", "combined")

    def test_unformatted_reply_uses_its_code(self):
        proposals = [Proposal(specialist=n, code=f"# {n}") for n in NAMES]
        assert parse_coordinator("```python\ny = 2\n```", proposals) == ("y = 2", None)
        assert parse_coordinator("CHOOSE: nobody", proposals) == (
            "CHOOSE: nobody",
            None,
        )

    def test_coordinator_selection_roundtrip(self):
        proposal = Proposal(specialist="skeptical", code="x = 1")
        reply = coordinator_selection(proposal)
        assert parse_coordinator(reply, [proposal]) == ("x = 1", "skeptical")

    def test_parse_probes_keeps_only_calls(self):
        text = "```python\nadd(1, 2)\nadd(0, 0)\nprint('x')\nadd(1, 2)\n```"
        assert parse_probes(text, "add", 5) == ["add(1, 2)", "add(0, 0)"]

    def test_parse_probes_from_second_code_block(self):
        text = (
            "```python\ndef add(a, b):\n    return a + b\n```\nCalls:\n"
            "```python\nadd(3, 4)\nadd(0, 0)\n```"
        )
        assert parse_probes(text, "add", 5) == ["add(3, 4)", "add(0, 0)"]

    def test_parse_reflection_operations(self):
        text = (
            "ADD: Check for empty input before indexing\n"
            "- AGREE 2\n"
            "DISAGREE 7\n"
            "EDIT 1: Off-by-one errors happen at range ends\n"
            "random chatter"
        )
        assert parse_reflection(text, existing=3) == [
            ("add", None, "Check for empty input before indexing"),
            ("agree", 2, ""),
            ("edit", 1, "Off-by-one errors happen at range ends"),
        ]

    def test_task_text_uses_evalplus_instruction_and_fences_stubs(self):
        stub = PROBLEM.model_copy(
            update={"prompt": "def add(a, b):\n    pass", "prompt_is_code": True}
        )
        text = specialist_messages(load_specialist("logical"), stub)[-1].content
        assert text.startswith("Please provide a self-contained Python script")
        assert "```python\ndef add(a, b):" in text
