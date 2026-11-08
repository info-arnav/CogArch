"""The environment: extracts an answer from a reply and judges it.

Agents are only ever told "wrong" (optionally with the kind of error for code).
They never see tests, gold answers, or how close they were.

  code    the first fenced code block, run against hidden asserts in a sandbox
  math    \\boxed{...} or a 'FINAL ANSWER:' line, checked with math-verify
  choice  the option letter after 'FINAL ANSWER:', compared with the gold letter
"""

from __future__ import annotations

import re

from src.config import FeedbackLevel
from src.execution.sandbox import Sandbox
from src.models.core import Task, Verdict

_FENCED = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
_FINAL = re.compile(r"FINAL ANSWER\W*[:\-]?\s*(.+)", re.IGNORECASE)


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


def extract_math(response: str) -> str:
    boxed = _last_boxed(response)
    if boxed:
        return boxed
    finals = _FINAL.findall(response)
    if finals:
        return str(finals[-1]).strip().rstrip(".").strip().strip("$").strip()
    return ""


def extract_choice(response: str, options: int) -> str:
    letters = "ABCDEFGHIJ"[:options]
    finals = _FINAL.findall(response)
    candidates = [finals[-1]] if finals else []
    candidates.append(response[-200:])
    for text in candidates:
        found = re.findall(rf"(?<![A-Za-z])\(?([{letters}])\)?(?![A-Za-z])", text)
        if found:
            return str(found[0] if text is not candidates[-1] else found[-1])
    return ""


def extract_answer(task: Task, response: str) -> str:
    if task.kind == "code":
        return extract_code(response)
    if task.kind == "math":
        return extract_math(response)
    return extract_choice(response, len(task.choices))


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
    def __init__(self, sandbox: Sandbox, feedback: FeedbackLevel) -> None:
        self.sandbox = sandbox
        self.feedback = feedback

    async def judge(self, task: Task, answer: str) -> Verdict:
        """Judge an already-extracted answer."""
        if task.kind == "code":
            return await self.sandbox.run_tests(answer, task)
        passed = answers_match(task, task.answer, answer)
        return Verdict(
            passed=passed,
            tests_passed=int(passed),
            tests_total=1,
            feedback="" if passed else "Your answer is wrong.",
        )
