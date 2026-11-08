"""Pass@1 on EvalPlus (HumanEval/HumanEval+ and MBPP/MBPP+).

The whole agent answers each problem once: specialists propose, the
coordinator decides. No memory, no feedback, no retries.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from src.inference.agent import Agent
from src.models.code import CodeProblem


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


async def evaluate(
    agent: Agent, benchmark: str, out_dir: Path, samples: int = 1
) -> dict[str, float]:
    """Pass@1 averaged over `samples` independent answers per task."""
    from evalplus.evaluate import evaluate as evalplus_evaluate
    from evalplus.sanitize import sanitize

    problems = load_problems(benchmark)
    tasks = [
        CodeProblem(
            task_id=tid,
            prompt=p["prompt"],
            prompt_is_code=True,
            entry_point=p["entry_point"],
            tests=[],
        )
        for tid, p in problems.items()
    ]
    jobs = [(task, i) for task in tasks for i in range(samples)]
    codes = await asyncio.gather(*(agent.answer(task, i) for task, i in jobs))

    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / f"{benchmark}_{agent.name}.jsonl"
    results = out_dir / f"{benchmark}_{agent.name}_eval_results.json"
    results.unlink(missing_ok=True)
    with open(samples_path, "w") as f:
        for (task, _), code in zip(jobs, codes):
            solution = sanitize(code, task.entry_point)
            f.write(json.dumps({"task_id": task.task_id, "solution": solution}) + "\n")

    await asyncio.to_thread(
        evalplus_evaluate, dataset=benchmark, samples=str(samples_path)
    )
    return pass_at_1(results)
