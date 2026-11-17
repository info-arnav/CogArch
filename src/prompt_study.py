"""Prompt study: choose the prompts on practice problems before any run.

Untrained model, problems from the training pools that no run uses, every reply
graded against the gold answer (hidden tests for code). Three questions:

  personas      each specialist alone, with its persona vs a neutral team prompt
  deliberation  the whole team under each discussion style (critique, hold, du)
                and with no discussion; round 1 is identical across styles
                (same seeds), so differences come from the discussion alone
  coordinator   on problems where the team disagrees, is the coordinator's pick
                right more often than the largest agreeing group?

    cogarch prompt-study config/general.yaml --per-domain 50
"""

from __future__ import annotations

import asyncio
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from rich.console import Console

from src.config import Config
from src.environment import Environment
from src.execution.sandbox import Sandbox
from src.grading import build_grader
from src.inference.agent import Agent, Role
from src.inference.client import InferenceClient
from src.inference.prompts import (
    NEUTRAL_PROMPT,
    Persona,
    load_coordinator,
    load_specialist,
)
from src.models.core import Decision, Proposal, Task
from src.tasks import TRAIN_SOURCES, Curriculum

NEUTRAL = NEUTRAL_PROMPT
LIGHT = (
    "You are the {name} specialist in a team of four: {description}. Show your "
    "reasoning, then give your answer in exactly the format the problem asks for."
)
PERSONA_MODES = ["persona", "light", "neutral"]
CONFIGS = [
    ("du", "persona", "coordinator"),
    ("du", "light", "coordinator"),
    ("du", "neutral", "coordinator"),
    ("du", "neutral", "majority"),
    ("du", "light", "majority"),
    ("none", "neutral", "majority"),
    ("none", "neutral", "coordinator"),
]
BOOTSTRAP = 2000


def study_tasks(cfg: Config, per_domain: int) -> list[Task]:
    """Training-pool problems that no run's curriculum or validation uses."""
    used = Curriculum(
        cfg.data.domains,
        cfg.data.validation_per_domain,
        cfg.experiment.rounds_per_cycle,
        cfg.experiment.cycles,
        cfg.data.split_seed,
    )
    taken = {t.task_id for t in used.validation} | {
        t.task_id for c in range(1, cfg.experiment.cycles + 1) for t in used.cycle(c)
    }
    tasks = []
    for domain in cfg.data.domains:
        pool = sorted(
            (t for t in TRAIN_SOURCES[domain]() if t.task_id not in taken),
            key=lambda t: t.task_id,
        )
        random.Random(f"study/{domain}").shuffle(pool)
        tasks.extend(pool[:per_domain])
    return tasks


class Study:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.client = InferenceClient(
            cfg.inference.base_url,
            cfg.inference.request_timeout_seconds,
            cfg.inference.max_concurrent_requests,
        )
        self.sandbox = Sandbox(
            cfg.environment.feedback,
            cfg.environment.execution_timeout_seconds,
            cfg.environment.memory_limit_mb,
        )
        self.grader = build_grader(
            cfg,
            InferenceClient(
                cfg.inference.base_url,
                cfg.inference.request_timeout_seconds,
                cfg.inference.max_concurrent_requests,
            ),
        )
        self.environment = Environment(
            self.sandbox, cfg.environment.feedback, self.grader
        )

    def agent(self, style: str, persona: str, selector: str = "coordinator") -> Agent:
        base = self.cfg.model.served_name
        specialists = {}
        for name in self.cfg.agents.specialists:
            p = load_specialist(name)
            if persona != "persona":
                prompt = (
                    NEUTRAL
                    if persona == "neutral"
                    else LIGHT.format(
                        name=name.capitalize(),
                        description=p.description[0].lower() + p.description[1:],
                    )
                )
                p = Persona(
                    name=p.name,
                    description=p.description,
                    temperature=p.temperature,
                    system_prompt=prompt,
                )
            specialists[name] = Role(p, base)
        return Agent(
            name="study",
            seed=self.cfg.agents.seeds[self.cfg.agents.names[0]],
            specialists=specialists,
            coordinator=Role(load_coordinator(), base),
            client=self.client,
            environment=self.environment,
            memory=None,
            max_attempts=1,
            max_tokens=self.cfg.model.max_tokens,
            probes_per_problem=self.cfg.environment.probes_per_problem,
            discussion="never" if style == "none" else "disagreement",
            discussion_style="critique" if style == "none" else style,
            selector=selector,
        )

    async def right(self, task: Task, reply: str, answer: str) -> bool:
        if task.kind == "code":
            return (await self.sandbox.run_tests(answer, task)).passed
        return await self.grader.correct(task, reply, task.answer)

    async def personas(self, tasks: list[Task]) -> dict[str, Any]:
        jobs = []
        for persona in PERSONA_MODES:
            agent = self.agent("du", persona)
            for s in agent.specialists:
                for t in tasks:
                    jobs.append((persona, s, t, agent))

        async def one(persona: str, s: str, t: Task, agent: Agent) -> bool:
            reply = await agent.solo_answer(s, t, 0)
            answer = await self.grader.code(t, reply) if t.kind == "code" else ""
            return await self.right(t, reply, answer)

        results = await asyncio.gather(*(one(*j) for j in jobs))
        acc: dict[str, dict[str, dict[str, list[bool]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for (persona, s, t, _), ok in zip(jobs, results):
            acc[persona][s][t.domain].append(ok)
        means: dict[str, Any] = _means(acc)
        return means

    async def deliberation(
        self, tasks: list[Task], style: str, persona: str, selector: str
    ) -> list[dict[str, Any]]:
        agent = self.agent(style, persona, selector)

        async def one(t: Task) -> dict[str, Any]:
            probes = await agent._probes(t)
            d = await agent._deliberate(t, 1, "", None, "study", probes)
            return {"task_id": t.task_id, "domain": t.domain, **await self.score(t, d)}

        return list(await asyncio.gather(*(one(t) for t in tasks)))

    async def score(self, t: Task, d: Decision) -> dict[str, Any]:
        async def ok(p: Proposal) -> bool:
            return await self.right(t, p.response, p.answer)

        r1 = await asyncio.gather(*(ok(p) for p in d.round1))
        r2 = await asyncio.gather(*(ok(p) for p in d.round2))
        right2 = {p.specialist: v for p, v in zip(d.round2, r2)}
        largest = max(d.evidence.groups, key=len)
        tied = sum(len(g) == len(largest) for g in d.evidence.groups) > 1
        return {
            "final": right2[d.chosen],
            "chosen_by": d.chosen_by,
            "discussed": d.discussed,
            "round1": list(r1),
            "round2": list(r2),
            "majority": right2[largest[0]] if not tied else None,
            "any_right": any(r2),
            "calls": len(d.calls),
        }

    async def close(self) -> None:
        await self.client.aclose()


def _means(tree: Any) -> Any:
    if isinstance(tree, list):
        return round(sum(tree) / len(tree), 3) if tree else None
    return {k: _means(v) for k, v in tree.items()}


def _summarise(tasks: list[Task], rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t, r in zip(tasks, rows):
        by_domain[t.domain].append(r)
    for domain, rs in [*by_domain.items(), ("all", rows)]:
        disagreed = [r for r in rs if r["chosen_by"] != "agreement"]
        flips = [
            (a, b)
            for r in rs
            if r["discussed"]
            for a, b in zip(r["round1"], r["round2"])
        ]
        picks = [r for r in disagreed if r["majority"] is not None]
        out[domain] = {
            "accuracy": round(sum(r["final"] for r in rs) / len(rs), 3),
            "round1_specialist_accuracy": round(
                sum(sum(r["round1"]) for r in rs) / (4 * len(rs)), 3
            ),
            "oracle": round(sum(r["any_right"] for r in rs) / len(rs), 3),
            "discussed": round(sum(r["discussed"] for r in rs) / len(rs), 3),
            "wrong_to_right": sum(1 for a, b in flips if not a and b),
            "right_to_wrong": sum(1 for a, b in flips if a and not b),
            "coordinator_right_when_split": (
                round(sum(r["final"] for r in picks) / len(picks), 3) if picks else None
            ),
            "majority_right_when_split": (
                round(sum(bool(r["majority"]) for r in picks) / len(picks), 3)
                if picks
                else None
            ),
            "split_problems": len(picks),
            "calls_per_problem": round(sum(r["calls"] for r in rs) / len(rs), 2),
        }
    return out


def paired(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    """a minus b on the same problems: accuracy difference with a bootstrap 95%
    interval, and McNemar's exact p-value on the problems where they differ."""
    x = [int(r["final"]) for r in a]
    y = [int(r["final"]) for r in b]
    n = len(x)
    rng = random.Random(0)
    diffs = []
    for _ in range(BOOTSTRAP):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(x[i] - y[i] for i in idx) / n)
    diffs.sort()
    only_a = sum(1 for i in range(n) if x[i] and not y[i])
    only_b = sum(1 for i in range(n) if y[i] and not x[i])
    k, m = min(only_a, only_b), only_a + only_b
    p = min(1.0, 2 * sum(_comb(m, j) for j in range(k + 1)) / 2**m) if m else 1.0
    return {
        "difference": round((sum(x) - sum(y)) / n, 3),
        "ci95": [
            round(diffs[int(0.025 * BOOTSTRAP)], 3),
            round(diffs[int(0.975 * BOOTSTRAP)], 3),
        ],
        "only_first_right": only_a,
        "only_second_right": only_b,
        "mcnemar_p": round(p, 4),
    }


def _comb(n: int, k: int) -> int:
    from math import comb

    return comb(n, k)


async def run_study(cfg: Config, per_domain: int, console: Console) -> Path:
    study = Study(cfg)
    await study.client.wait_ready()
    tasks = study_tasks(cfg, per_domain)
    console.print(f"{len(tasks)} practice problems")
    names = ["/".join(c) for c in CONFIGS]
    persona_scores, *rows = await asyncio.gather(
        study.personas(tasks), *(study.deliberation(tasks, *c) for c in CONFIGS)
    )
    await study.close()
    teams = dict(zip(names, rows))
    reference = names[0]
    results = {
        "problems": len(tasks),
        "personas": persona_scores,
        "teams": {n: _summarise(tasks, r) for n, r in teams.items()},
        "versus_"
        + reference: {
            n: paired(r, teams[reference]) for n, r in teams.items() if n != reference
        },
        "pairs": {
            "neutral_vs_persona": paired(
                teams["du/neutral/coordinator"], teams["du/persona/coordinator"]
            ),
            "light_vs_neutral": paired(
                teams["du/light/coordinator"], teams["du/neutral/coordinator"]
            ),
            "majority_vs_coordinator": paired(
                teams["du/neutral/majority"], teams["du/neutral/coordinator"]
            ),
            "discussion_vs_none": paired(
                teams["du/neutral/majority"], teams["none/neutral/majority"]
            ),
        },
    }
    out = cfg.experiment.output_dir / "prompt_study"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"study2_{per_domain}.json"
    path.write_text(json.dumps(results, indent=2))
    (out / f"study2_{per_domain}_rows.json").write_text(json.dumps(teams))
    console.print_json(data={k: v for k, v in results.items() if k != "teams"})
    return path
