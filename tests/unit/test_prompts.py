"""Prompt building and parsing."""

from src.inference.prompts import (
    choice_options,
    coordinator_messages,
    discussion_messages,
    evidence_text,
    load_coordinator,
    load_specialist,
    parse_choice,
    parse_probes,
    parse_reflection,
    specialist_messages,
    task_text,
)
from src.models.core import Decision, Evidence, Proposal, Task, Verdict


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
        chosen="logical",
        chosen_by="coordinator",
        submitted="\\boxed{41}",
        answer="41",
        result=Verdict(passed=False, feedback="Your answer is wrong."),
    )
    text = specialist_messages(load_specialist("logical"), math_task, previous=prev)[
        -1
    ].content
    assert "41" in text and "Your answer is wrong." in text


def test_discussion_and_choice_prompts(math_task):
    own = Proposal(
        specialist="logical", round=1, response="I say \\boxed{36}", answer="36"
    )
    mate = Proposal(
        specialist="creative", round=1, response="I say \\boxed{42}", answer="42"
    )
    evidence = Evidence(
        answers={"logical": "36", "creative": "42"}, groups=[["logical"], ["creative"]]
    )
    system, user = discussion_messages(
        load_specialist("logical"), math_task, own, [mate], evidence
    )
    text = user.content
    assert (
        "I say \\boxed{36}" in text
        and "### creative\nI say \\boxed{42}\nAnswer: 42" in text
    )
    assert "- creative: 42" in text and "DISAGREE <name>" in system.content
    assert "DISAGREE" not in text
    choose = coordinator_messages(load_coordinator(), math_task, [own, mate], evidence)[
        -1
    ].content
    assert "### logical" in choose and "Reply with only the letter" in choose
    assert "logical, creative" in choose


def test_long_reasoning_shows_its_end(math_task):
    own = Proposal(
        specialist="logical", round=1, response="a" * 2000 + "z" * 900, answer="1"
    )
    text = discussion_messages(
        load_specialist("logical"), math_task, own, [], Evidence()
    )[-1].content
    assert "… " + "z" * 900 in text and "a" * 10 not in text


def test_code_evidence_lists_examples_and_groups(add_task):
    evidence = Evidence(
        answers={"logical": "x", "creative": "y"},
        examples={"logical": [1, 1], "creative": [0, 1]},
        groups=[["logical"], ["creative"]],
    )
    text = evidence_text(add_task, evidence, ["logical", "creative"])
    assert "- creative: passes 0 of 1 shown examples" in text
    assert "identical outputs" in text


def test_choice_options_and_parse():
    options = choice_options(["W", "X", "Y", "Z"], may_reject=True)
    assert options == ["W", "X", "Y", "Z", "none"]
    assert choice_options(["W", "X"], may_reject=False) == ["W", "X"]
    assert parse_choice(" y\n", options) == "Y"
    assert parse_choice("none", options) == "none"
    assert parse_choice("FINAL CHOICE: Y", options) is None
    assert parse_choice("B", options) is None


def test_parse_probes_from_any_block():
    text = "```python\ndef add(a, b): ...\n```\n```python\nadd(1, 2)\nadd(0, 0)\n```"
    assert parse_probes(text, "add", 5) == ["add(1, 2)", "add(0, 0)"]


def test_parse_reflection_notes():
    text = "CASE 1: forgot to subtract\n- **CASE 2**: misread the units\nADD: ignored"
    assert parse_reflection(text) == [
        (1, "forgot to subtract"),
        (2, "misread the units"),
    ]
