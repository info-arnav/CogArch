"""Training curricula and evaluation benchmarks across domains."""

from __future__ import annotations

import random
from collections.abc import Callable

from src.models.core import Task
from src.tasks import code, general

# Domains the community trains on, and where their problems come from.
TRAIN_SOURCES: dict[str, Callable[[], list[Task]]] = {
    "code": code.load_pool,
    "math": lambda: general.gsm8k("train"),
    "reasoning": lambda: general.arc_challenge("train"),
    "social": lambda: general.social_iqa("train"),
}

# Held-out benchmarks. EvalPlus code benchmarks are scored by EvalPlus itself.
BENCHMARKS: dict[str, Callable[[], list[Task]]] = {
    "humaneval": lambda: code.evalplus_tasks("humaneval"),
    "mbpp": lambda: code.evalplus_tasks("mbpp"),
    "gsm8k": lambda: general.gsm8k("test"),
    "math500": general.math500,
    "arc_challenge": lambda: general.arc_challenge("test"),
    "siqa": lambda: general.social_iqa("validation"),
    "emobench_eu": general.emobench_understanding,
    "emobench_ea": general.emobench_application,
}
EVALPLUS_BENCHMARKS = {"humaneval", "mbpp"}


class Curriculum:
    """A fixed, seeded split of each training domain into validation and
    per-cycle problem sets, with domains interleaved within each cycle."""

    def __init__(
        self,
        domains: list[str],
        validation_per_domain: int,
        rounds_per_cycle: int,
        cycles: int,
        seed: int,
    ) -> None:
        if rounds_per_cycle % len(domains):
            raise ValueError(
                "rounds_per_cycle must be divisible by the number of domains"
            )
        per_domain = rounds_per_cycle // len(domains)
        self.validation: list[Task] = []
        self._cycles: list[list[Task]] = [[] for _ in range(cycles)]
        for domain in domains:
            pool = sorted(TRAIN_SOURCES[domain](), key=lambda t: t.task_id)
            random.Random(f"{seed}/{domain}").shuffle(pool)
            needed = validation_per_domain + per_domain * cycles
            if needed > len(pool):
                raise ValueError(
                    f"{domain}: {needed} problems needed, {len(pool)} available"
                )
            self.validation.extend(pool[:validation_per_domain])
            rest = pool[validation_per_domain:]
            for c in range(cycles):
                self._cycles[c].extend(rest[c * per_domain : (c + 1) * per_domain])
        for c, tasks in enumerate(self._cycles):
            random.Random(f"{seed}/order/{c}").shuffle(tasks)

    def cycle(self, number: int) -> list[Task]:
        return self._cycles[number - 1]


def benchmark_subset(tasks: list[Task], limit: int | None, seed: int) -> list[Task]:
    """A fixed random subset (the same for every run), or all tasks."""
    if limit is None or limit >= len(tasks):
        return tasks
    chosen = sorted(tasks, key=lambda t: t.task_id)
    random.Random(seed).shuffle(chosen)
    return sorted(chosen[:limit], key=lambda t: t.task_id)
