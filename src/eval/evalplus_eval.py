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
    """Base and plus Pass@1 from an EvalPlus results file (one sample per task)."""
    tasks = json.loads(results_path.read_text())["eval"]
    n = len(tasks)
    base = sum(r[0]["base_status"] == "pass" for r in tasks.values())
    plus = sum(
        r[0]["base_status"] == "pass" and r[0]["plus_status"] == "pass"
        for r in tasks.values()
    )
    return {"base": base / n, "plus": plus / n}


async def evaluate(agent: Agent, benchmark: str, out_dir: Path) -> dict[str, float]:
    from evalplus.evaluate import evaluate as evalplus_evaluate
    from evalplus.sanitize import sanitize

    problems = load_problems(benchmark)
    tasks = [
        CodeProblem(
            task_id=tid, prompt=p["prompt"], entry_point=p["entry_point"], tests=[]
        )
        for tid, p in problems.items()
    ]
    codes = await asyncio.gather(*(agent.answer(t) for t in tasks))

    out_dir.mkdir(parents=True, exist_ok=True)
    samples = out_dir / f"{benchmark}_{agent.name}.jsonl"
    results = out_dir / f"{benchmark}_{agent.name}_eval_results.json"
    results.unlink(missing_ok=True)
    with open(samples, "w") as f:
        for task, code in zip(tasks, codes):
            solution = sanitize(code, task.entry_point)
            f.write(json.dumps({"task_id": task.task_id, "solution": solution}) + "\n")

    await asyncio.to_thread(evalplus_evaluate, dataset=benchmark, samples=str(samples))
    return pass_at_1(results)
