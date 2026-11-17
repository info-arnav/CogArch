"""Benchmark scoring for every kind of system we evaluate.

Three modes, each on the same benchmark tasks:

  agent       the whole brain (round 1, discussion, decision), no memory
  specialist  one specialist alone with its own persona (shows specialisation)
  specialist_neutral
              the same specialist, same weights and temperature, without its
              persona sentence (isolates what the persona costs or adds)
  plain       one model with no persona, greedy, using the standard prompt for
              that benchmark (EvalPlus's chat protocol for code) — comparable
              to officially published single-model scores

Code benchmarks are scored by EvalPlus (base and plus tests); all others by
the environment's answer checkers.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

from src import calllog
from src.environment import answers_match, extract_answer
from src.inference.client import InferenceClient
from src.inference.prompts import CODE_INSTRUCTION, task_text
from src.models.core import Message, Task

if TYPE_CHECKING:
    from src.grading import Grader

Answerer = Callable[[Task, int], Awaitable[str]]


def pass_at_1(results_path: Path) -> dict[str, float]:
    """Base and plus Pass@1 from an EvalPlus results file, averaged over each
    task's samples and then over tasks."""
    tasks = list(json.loads(results_path.read_text())["eval"].values())
    base = [sum(s["base_status"] == "pass" for s in t) / len(t) for t in tasks]
    plus = [
        sum(s["base_status"] == "pass" and s["plus_status"] == "pass" for s in t)
        / len(t)
        for t in tasks
    ]
    return {"base": sum(base) / len(base), "plus": sum(plus) / len(plus)}


async def _score_evalplus(
    benchmark: str,
    answers: list[tuple[Task, str]],
    samples_path: Path,
    grader: Grader,
) -> dict[str, float]:
    from evalplus.evaluate import evaluate as evalplus_evaluate

    samples_path.parent.mkdir(parents=True, exist_ok=True)
    results = samples_path.with_name(samples_path.stem + "_eval_results.json")
    results.unlink(missing_ok=True)
    solutions = await asyncio.gather(*(grader.code(t, reply) for t, reply in answers))
    with open(samples_path, "w") as f:
        for (task, _), solution in zip(answers, solutions):
            f.write(json.dumps({"task_id": task.task_id, "solution": solution}) + "\n")
    await asyncio.to_thread(
        evalplus_evaluate, dataset=benchmark, samples=str(samples_path)
    )
    return pass_at_1(results)


async def _score_answers(
    answers: list[tuple[Task, str]], samples_path: Path, grader: Grader
) -> dict[str, float]:
    """Graded by the verifier; the rule-based score is kept beside it, the way
    lm-eval reports strict and flexible matches."""
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    verdicts = await asyncio.gather(
        *(grader.correct(task, reply, task.answer) for task, reply in answers)
    )
    hits, rule_hits = [], []
    with open(samples_path, "w") as f:
        for (task, reply), ok in zip(answers, verdicts):
            extracted = extract_answer(task, reply)
            rule = answers_match(task, task.answer, extracted)
            hits.append(ok)
            rule_hits.append(rule)
            f.write(
                json.dumps(
                    {
                        "task_id": task.task_id,
                        "answer": extracted,
                        "correct": ok,
                        "rule_correct": rule,
                        "reply": reply,
                    }
                )
                + "\n"
            )
    return {
        "accuracy": sum(hits) / len(hits),
        "rule_accuracy": sum(rule_hits) / len(rule_hits),
    }


async def score(
    benchmark: str,
    tasks: list[Task],
    answerer: Answerer,
    samples: int,
    samples_path: Path,
    grader: Grader,
) -> dict[str, float]:
    """Pass@1 / accuracy averaged over `samples` replies per task."""
    jobs = [(t, i) for t in tasks for i in range(samples)]
    replies = await asyncio.gather(
        *(
            calllog.within({"task_id": t.task_id, "sample": i}, answerer(t, i))
            for t, i in jobs
        )
    )
    answers = [(t, a) for (t, _), a in zip(jobs, replies)]
    if tasks and tasks[0].kind == "code" and benchmark in {"humaneval", "mbpp"}:
        if len(tasks) < len(_all_evalplus(benchmark)):
            raise ValueError(f"EvalPlus needs every {benchmark} task; got a subset")
        return await _score_evalplus(benchmark, answers, samples_path, grader)
    return await _score_answers(answers, samples_path, grader)


def _all_evalplus(benchmark: str) -> list[Task]:
    from src.tasks.code import evalplus_tasks

    return evalplus_tasks(benchmark)


# ── plain single-model protocol ─────────────────────────────────────────────

_CODE_RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves the "
    "problem and passes corresponding tests:\n```python\n"
)


def plain_answerer(client: InferenceClient, model: str, max_tokens: int) -> Answerer:
    """A single model, no persona, greedy. Code follows EvalPlus's chat protocol
    exactly (instruction, assistant prefix, continuation)."""

    async def answer(task: Task, sample: int) -> str:
        if task.kind == "code":
            prompt = f"{CODE_INSTRUCTION}\n```python\n{task.prompt.strip()}\n```"
            reply = await client.complete_after(
                [Message(role="user", content=prompt)],
                _CODE_RESPONSE_PREFIX,
                model,
                max_tokens,
            )
            return f"```python\n{reply}"
        return await client.chat(
            [Message(role="user", content=task_text(task))],
            model=model,
            temperature=0.0,
            max_tokens=max_tokens,
            seed=sample,
        )

    return answer


def _same(task: Task, a: str, b: str) -> bool:
    if task.kind == "code":
        return " ".join(a.split()) == " ".join(b.split())
    return answers_match(task, a, b)


def majority_answerer(
    client: InferenceClient,
    model: str,
    max_tokens: int,
    samples: int,
    temperature: float = 0.7,
) -> Answerer:
    """Self-consistency (Wang et al., 2023): sample several answers from one
    model with no persona and return the most common one. Same prompts as the
    plain protocol; code answers are grouped by identical (whitespace-normalised)
    source, maths by math-verify equivalence, choices by letter. Ties go to the
    earliest sample."""

    async def answer(task: Task, sample: int) -> str:
        if task.kind == "code":
            prompt = f"{CODE_INSTRUCTION}\n```python\n{task.prompt.strip()}\n```"
            messages = [Message(role="user", content=prompt)]
        else:
            messages = [Message(role="user", content=task_text(task))]
        replies = await asyncio.gather(
            *(
                client.chat(
                    messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=sample * 1000 + i,
                )
                for i in range(samples)
            )
        )
        return most_common_reply(task, list(replies))

    return answer


def most_common_reply(task: Task, replies: list[str]) -> str:
    """A reply carrying the most common answer (ties go to the earliest);
    the first reply if none has a readable answer."""
    answered = [(extract_answer(task, r), r) for r in replies]
    answered = [(a, r) for a, r in answered if a]
    if not answered:
        return replies[0]
    groups: list[list[tuple[str, str]]] = []
    for a, r in answered:
        for g in groups:
            if _same(task, g[0][0], a):
                g.append((a, r))
                break
        else:
            groups.append([(a, r)])
    return max(groups, key=len)[0][1]
