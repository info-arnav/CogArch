"""Pass@1 on EvalPlus (HumanEval/HumanEval+ and MBPP/MBPP+).

The whole agent answers each problem once: specialists propose, the
coordinator decides. No memory, no feedback, no retries.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from src.inference.agent import Agent
from src.inference.client import InferenceClient
from src.inference.prompts import INSTRUCTION
from src.models.code import CodeProblem, Message


def load_problems(benchmark: str) -> dict[str, dict]:
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    problems: dict[str, dict] = (
        get_human_eval_plus() if benchmark == "humaneval" else get_mbpp_plus()
    )
    return problems


def pass_at_1(results_path: Path) -> dict[str, float]:
    """Base and plus Pass@1 from an EvalPlus results file, averaged over each
    task's samples and then over tasks."""
    tasks = json.loads(results_path.read_text())["eval"].values()
    base = [sum(s["base_status"] == "pass" for s in t) / len(t) for t in tasks]
    plus = [
        sum(s["base_status"] == "pass" and s["plus_status"] == "pass" for s in t)
        / len(t)
        for t in tasks
    ]
    return {"base": sum(base) / len(base), "plus": sum(plus) / len(plus)}


async def _score(
    benchmark: str, answers: list[tuple[CodeProblem, str]], samples_path: Path
) -> dict[str, float]:
    from evalplus.evaluate import evaluate as evalplus_evaluate
    from evalplus.sanitize import sanitize

    samples_path.parent.mkdir(parents=True, exist_ok=True)
    results = samples_path.with_name(samples_path.stem + "_eval_results.json")
    results.unlink(missing_ok=True)
    with open(samples_path, "w") as f:
        for task, code in answers:
            solution = sanitize(code, task.entry_point)
            f.write(json.dumps({"task_id": task.task_id, "solution": solution}) + "\n")
    await asyncio.to_thread(
        evalplus_evaluate, dataset=benchmark, samples=str(samples_path)
    )
    return pass_at_1(results)


def _tasks(benchmark: str) -> list[CodeProblem]:
    return [
        CodeProblem(
            task_id=tid,
            prompt=p["prompt"],
            prompt_is_code=True,
            entry_point=p["entry_point"],
            tests=[],
        )
        for tid, p in load_problems(benchmark).items()
    ]


async def evaluate(
    agent: Agent, benchmark: str, out_dir: Path, samples: int = 1
) -> dict[str, float]:
    """The whole agent (specialists + coordinator): Pass@1 averaged over
    `samples` independent answers per task."""
    jobs = [(task, i) for task in _tasks(benchmark) for i in range(samples)]
    codes = await asyncio.gather(*(agent.answer(task, i) for task, i in jobs))
    answers = [(task, code) for (task, _), code in zip(jobs, codes)]
    return await _score(benchmark, answers, out_dir / f"{benchmark}_{agent.name}.jsonl")


# EvalPlus's chat-model protocol: user instruction, assistant prefix, greedy.
_RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves the "
    "problem and passes corresponding tests:\n```python\n"
)


async def evaluate_plain(
    client: InferenceClient,
    model: str,
    benchmark: str,
    samples_path: Path,
    max_tokens: int,
) -> dict[str, float]:
    """A single model on its own, exactly as EvalPlus scores chat models: no
    persona, no coordinator, greedy. Comparable to officially published scores."""
    tasks = _tasks(benchmark)
    replies = await asyncio.gather(
        *(
            client.complete_after(
                [
                    Message(
                        role="user",
                        content=f"{INSTRUCTION}\n```python\n{t.prompt.strip()}\n```",
                    )
                ],
                _RESPONSE_PREFIX,
                model,
                max_tokens,
            )
            for t in tasks
        )
    )
    answers = [(t, f"```python\n{reply}") for t, reply in zip(tasks, replies)]
    return await _score(benchmark, answers, samples_path)
