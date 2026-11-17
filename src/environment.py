"""The environment: judges a submitted reply, and rule-based answer extraction.

Agents are only ever told "wrong" (optionally with the kind of error for code).
They never see tests, gold answers, or how close they were. Judging goes
through the Grader (src/grading.py): xVerify for maths and choice, EvalPlus's
sanitiser and the hidden tests for code.

The extractors below read an answer by rule. They group the team's positions
by answer and are the grader's fallback when no verifier is configured.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from src.config import FeedbackLevel
from src.execution.sandbox import Sandbox
from src.models.core import Task, Verdict

if TYPE_CHECKING:
    from src.grading import Grader

_FENCED = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
_FINAL = re.compile(r"FINAL ANSWER\W*[:\-]?\s*(.+)", re.IGNORECASE)
_STATED = re.compile(
    r"(?:the\s+)?(?:correct\s+|final\s+|best\s+)?(?:answer|option|choice)\s*"
    r"(?:is|would be|should be|:)\s*:?\s*\**\s*\(?([^\n]{1,80})",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_code(response: str) -> str:
    """The first fenced code block, or the whole response if there is none."""
    match = _FENCED.search(response)
    return (match.group(1) if match else response).strip()


def _last_boxed(text: str) -> str | None:
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth, i = 0, start + len("\\boxed{") - 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j].strip()
    return None


def _clean(value: str) -> str:
    return value.strip().rstrip(".").strip().strip("$*").strip()


def extract_math(response: str) -> str:
    """\\boxed{...}, else a 'FINAL ANSWER:' line, else 'the answer is ...', else
    the last number in the reply's final lines."""
    boxed = _last_boxed(response)
    if boxed:
        return boxed
    finals = _FINAL.findall(response)
    if finals:
        return _clean(str(finals[-1]))
    stated = _STATED.findall(response)
    if stated:
        number = _NUMBER.search(stated[-1])
        if number:
            return number.group(0).replace(",", "")
    numbers = _NUMBER.findall(response[-300:])
    return str(numbers[-1]).replace(",", "").rstrip(".") if numbers else ""


def _letter(text: str, letters: str, last: bool) -> str:
    found = re.findall(rf"(?<![A-Za-z])\(?([{letters}])\)?(?![A-Za-z'’])", text)
    if not found:
        return ""
    return str(found[-1] if last else found[0])


def extract_choice(
    response: str, options: int, choices: list[str] | None = None
) -> str:
    """The option letter after 'FINAL ANSWER:', else after 'the answer is', else
    the one option whose text the reply's ending names, else the last lone letter
    near the end."""
    letters = "ABCDEFGHIJ"[:options]
    finals = _FINAL.findall(response)
    if finals:
        return _letter(finals[-1], letters, last=False)
    for stated in reversed(_STATED.findall(response)):
        letter = _letter(stated, letters, last=False)
        if letter:
            return letter
    if choices:
        ending = response[-400:].lower()
        named = [
            i
            for i, c in enumerate(choices)
            if len(c) > 2 and c.lower().rstrip(".") in ending
        ]
        if len(named) == 1:
            return letters[named[0]]
    return _letter(response[-200:], letters, last=True)


def extract_answer(task: Task, response: str) -> str:
    if task.kind == "code":
        return extract_code(response)
    if task.kind == "math":
        return extract_math(response)
    return extract_choice(response, len(task.choices), task.choices)


def math_equal(gold: str, predicted: str) -> bool:
    if not predicted:
        return False
    from math_verify import parse, verify

    parsed_gold = parse(f"\\boxed{{{gold}}}")
    parsed_pred = parse(f"\\boxed{{{predicted}}}")
    if parsed_gold and parsed_pred:
        return bool(verify(parsed_gold, parsed_pred))
    return gold.strip().lower() == predicted.strip().lower()


def answers_match(task: Task, a: str, b: str) -> bool:
    """Same answer, for math and choice tasks (code uses behavioural probes)."""
    if task.kind == "math":
        return math_equal(a, b)
    return bool(a) and a == b


class Environment:
    def __init__(
        self, sandbox: Sandbox, feedback: FeedbackLevel, grader: Grader
    ) -> None:
        self.sandbox = sandbox
        self.feedback = feedback
        self.grader = grader

    async def judge(self, task: Task, reply: str) -> Verdict:
        """Judge a submitted reply."""
        if task.kind == "code":
            return await self.sandbox.run_tests(
                await self.grader.code(task, reply), task
            )
        passed = await self.grader.correct(task, reply, task.answer)
        return Verdict(
            passed=passed,
            tests_passed=int(passed),
            tests_total=1,
            feedback="" if passed else "Your answer is wrong.",
        )
