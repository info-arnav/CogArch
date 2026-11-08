"""Prompt building and parsing."""

from src.inference.prompts import (
    coordinator_endorsement,
    coordinator_messages,
    discussion_messages,
    load_coordinator,
    load_specialist,
    parse_probes,
    parse_reflection,
    specialist_messages,
    task_text,
)
from src.models.core import Decision, Proposal, Task, Verdict


def prop(name: str, text: str) -> Proposal:
    return Proposal(specialist=name, round=1, response=text, answer="")


def test_task_text_per_kind(add_task, math_task, choice_task):
    assert task_text(add_task).startswith(
        "Please provide a self-contained Python script"
    )
    assert "`add`" in task_text(add_task)
    assert "\\boxed{}" in task_text(math_task)
    text = task_text(choice_task)
    assert "A. happy\nB. sad\nC. bored" in text and "FINAL ANSWER: <letter>" in text


def test_code_stub_is_fenced():
    stub = Task(
        task_id="h",
        domain="code",
        kind="code",
        prompt="def f():\n    pass",
        prompt_is_code=True,
        entry_point="f",
    )
    assert "```python\ndef f():" in task_text(stub)


def test_prompts_never_contain_gold_or_tests(add_task, math_task):
    assert (
        "assert"
        not in specialist_messages(load_specialist("logical"), add_task)[-1].content
    )
    assert (
        "42"
        not in specialist_messages(load_specialist("logical"), math_task)[-1].content
    )


def test_retry_shows_previous_answer_and_wrong(math_task):
    prev = Decision(
        attempt_number=1,
        round1=[],
        round2=[],
        coordinator_prompt=[],
        coordinator_output="",
        answer="41",
        result=Verdict(passed=False, feedback="Your answer is wrong."),
    )
    text = specialist_messages(load_specialist("logical"), math_task, previous=prev)[
        -1
    ].content
    assert "41" in text and "Your answer is wrong." in text


def test_discussion_and_decision_prompts(math_task):
    own, mates = prop("logical", "I say 36"), [prop("creative", "I say 42")]
    text = discussion_messages(load_specialist("logical"), math_task, own, mates)[
        -1
    ].content
    assert "I say 36" in text and "### creative\nI say 42" in text
    decide = coordinator_messages(load_coordinator(), math_task, [own, *mates])[
        -1
    ].content
    assert "### logical" in decide and "### creative" in decide


def test_long_replies_are_truncated(math_task):
    own = prop("logical", "x" * 5000)
    text = discussion_messages(load_specialist("logical"), math_task, own, [])[
        -1
    ].content
    assert "x" * 1800 + " …" in text and "x" * 1801 not in text


def test_endorsement_contains_the_reasoning():
    assert coordinator_endorsement(prop("skeptical", "\\boxed{42}")).startswith(
        "skeptical's reasoning holds up"
    )


def test_parse_probes_from_any_block():
    text = "```python\ndef add(a, b): ...\n```\n```python\nadd(1, 2)\nadd(0, 0)\n```"
    assert parse_probes(text, "add", 5) == ["add(1, 2)", "add(0, 0)"]


def test_parse_reflection_operations():
    text = "ADD: Re-read the question\n- AGREE 2\nDISAGREE 9\nEDIT 1: Check units"
    assert parse_reflection(text, existing=3) == [
        ("add", None, "Re-read the question"),
        ("agree", 2, ""),
        ("edit", 1, "Check units"),
    ]
