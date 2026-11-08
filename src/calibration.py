"""Model calibration: how an untrained team does on each training domain.

The community only gets learning signal when agents sometimes fail and
sometimes succeed, so a good model solves roughly 40–70% per domain, follows
the answer formats, and leaves headroom on the benchmarks.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from rich.console import Console

from src.config import Config
from src.environment import Environment
from src.execution.sandbox import Sandbox
from src.inference.agent import Agent, Role
from src.inference.client import InferenceClient
from src.inference.prompts import load_coordinator, load_specialist
from src.tasks import Curriculum


async def calibrate(cfg: Config, per_domain: int, console: Console) -> Path:
    client = InferenceClient(
        cfg.inference.base_url,
        cfg.inference.request_timeout_seconds,
        cfg.inference.max_concurrent_requests,
    )
    sandbox = Sandbox(
        cfg.environment.feedback,
        cfg.environment.execution_timeout_seconds,
        cfg.environment.memory_limit_mb,
    )
    base = cfg.model.served_name
    agent = Agent(
        name="a",
        seed=cfg.agents.seeds[cfg.agents.names[0]],
        specialists={n: Role(load_specialist(n), base) for n in cfg.agents.specialists},
        coordinator=Role(load_coordinator(), base),
        client=client,
        environment=Environment(sandbox, cfg.environment.feedback),
        memory=None,
        max_attempts=cfg.environment.max_attempts,
        max_tokens=cfg.model.max_tokens,
        probes_per_problem=cfg.environment.probes_per_problem,
        discussion=cfg.agents.discussion,
    )
    curriculum = Curriculum(
        cfg.data.domains,
        cfg.data.validation_per_domain,
        len(cfg.data.domains) * per_domain,
        1,
        cfg.data.split_seed,
    )
    tasks = curriculum.cycle(1)

    start = time.perf_counter()
    results = await asyncio.gather(*(agent.solve(t) for t in tasks))
    seconds = time.perf_counter() - start

    stats: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for task, result in zip(tasks, results):
        first = result.decisions[0]
        s = stats[task.domain]
        s["first_attempt_solved"].append(float(first.result.passed))
        s["solved_within_attempts"].append(float(result.solved))
        s["final_answer_parsed"].append(float(bool(first.answer)))
        s["round1_parsed"].append(
            sum(bool(p.answer) for p in first.round1) / len(first.round1)
        )
        s["discussion_changed"].append(
            sum(a.answer != b.answer for a, b in zip(first.round1, first.round2))
            / len(first.round1)
        )
    report: dict[str, Any] = {
        "model": cfg.model.base,
        "tasks_per_domain": per_domain,
        "seconds_per_task": seconds / len(tasks),
        "domains": {
            d: {k: sum(v) / len(v) for k, v in by.items()} for d, by in stats.items()
        },
    }
    out = Path("data/calibration")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{cfg.model.base.replace('/', '__')}.json"
    path.write_text(json.dumps(report, indent=2))
    console.print_json(data=report)
    await client.aclose()
    return path
