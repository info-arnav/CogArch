"""Code tasks: the MBPP training pool and the EvalPlus benchmarks.

Uses the full MBPP dataset (all splits) minus every task that appears in
MBPP+, which is reserved for evaluation. Agents see the task description and
the function signature; the tests stay with the environment.
"""

from __future__ import annotations

import random
import re
from typing import Any

from src.models.core import Task

_ASSERT_CALL = re.compile(r"assert\s+(?:not\s+)?\(?\s*(?:set\()?\s*(\w+)\s*\(")


def entry_point(code: str, tests: list[str]) -> str:
    """The function the tests call."""
    for test in tests:
        match = _ASSERT_CALL.search(test)
        if match and re.search(rf"^def\s+{match.group(1)}\s*\(", code, re.MULTILINE):
            return match.group(1)
    match = re.search(r"^def\s+(\w+)\s*\(", code, re.MULTILINE)
    if match is None:
        raise ValueError("No function definition in reference code")
    return match.group(1)


def signature(code: str, name: str) -> str:
    match = re.search(
        rf"^def\s+{name}\s*\(.*?\)\s*(?:->.*?)?:", code, re.MULTILINE | re.DOTALL
    )
    if match is None:
        raise ValueError(f"No signature for {name}")
    return match.group(0)


def to_problem(row: dict[str, Any]) -> Task:
    tests = [t.strip() for t in row["test_list"] if t.strip()]
    name = entry_point(row["code"], tests)
    prompt = (
        f"{row['text'].strip()}\n\n"
        f"Use this signature:\n```python\n{signature(row['code'], name)}\n```"
    )
    return Task(
        task_id=f"mbpp/{row['task_id']}",
        domain="code",
        kind="code",
        prompt=prompt,
        entry_point=name,
        tests=tests,
        setup_code=row.get("test_setup_code") or "",
    )


def split(
    problems: list[Task], validation_size: int, seed: int
) -> tuple[list[Task], list[Task]]:
    """Deterministic (train, validation) split."""
    shuffled = sorted(problems, key=lambda p: p.task_id)
    random.Random(seed).shuffle(shuffled)
    return shuffled[validation_size:], shuffled[:validation_size]


def mbpp_plus_task_numbers() -> set[int]:
    from evalplus.data import get_mbpp_plus

    return {int(task_id.split("/")[1]) for task_id in get_mbpp_plus()}


def load_pool() -> list[Task]:
    """All MBPP tasks not in MBPP+."""
    from datasets import load_dataset

    excluded = mbpp_plus_task_numbers()
    rows = [
        row
        for part in load_dataset("google-research-datasets/mbpp", "full").values()
        for row in part
        if row["task_id"] not in excluded
    ]
    return [to_problem(row) for row in rows]


def evalplus_tasks(benchmark: str) -> list[Task]:
    """HumanEval+ ('humaneval') or MBPP+ ('mbpp') as tasks; scored by EvalPlus."""
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    problems = get_human_eval_plus() if benchmark == "humaneval" else get_mbpp_plus()
    return [
        Task(
            task_id=tid,
            domain="code",
            kind="code",
            prompt=p["prompt"],
            prompt_is_code=True,
            entry_point=p["entry_point"],
        )
        for tid, p in problems.items()
    ]
