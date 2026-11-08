"""The whole pipeline offline: run, baseline, analysis and dashboard data.

A scripted model replaces vLLM; tasks and benchmarks are small fake sets.
Training is disabled (it needs a GPU); everything else runs for real.
"""

from __future__ import annotations

import json

import pytest

import src.experiment as experiment_module
import src.tasks as tasks_module
from src.analysis import analyze
from src.config import load_config
from src.dashboard import RunStore
from src.experiment import Experiment
from src.models.core import Message, Task
from tests.conftest import role_of


def _math(i: int) -> Task:
    return Task(
        task_id=f"m/{i}",
        domain="math",
        kind="math",
        prompt=f"What is {i} + {i}?",
        answer=str(2 * i),
    )


def _choice(i: int) -> Task:
    return Task(
        task_id=f"s/{i}",
        domain="social",
        kind="choice",
        prompt=f"Case {i}",
        choices=["no", "yes"],
        answer="B",
        max_attempts=1,
    )


class FakeClient:
    """Specialists differ: logical is good at maths, empathetic at social tasks,
    and the coordinator follows whichever position it sees first."""

    def __init__(self) -> None:
        self.loaded: list[str] = []

    async def chat(
        self,
        messages: list[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> str:
        system, user = messages[0].content, messages[-1].content
        role = role_of(system)
        if role == "reflect":
            return "ADD: Re-read the question before answering"
        if "multiple-choice" in user:
            good = role in ("empathetic", "coordinator") or seed % 3 == 0
            return "FINAL ANSWER: B" if good else "FINAL ANSWER: A"
        n = int(user.split("What is ")[1].split(" +")[0]) if "What is " in user else 0
        good = role in ("logical", "coordinator") or seed % 2 == 0
        if role == "unknown":  # plain mode: no system persona
            good = seed % 2 == 0 or n % 2 == 0
        return f"\\boxed{{{2 * n if good else 2 * n + 1}}}"

    async def complete_after(
        self, messages, prefill, model, max_tokens
    ) -> str:  # noqa: ANN001
        return "def f():\n    return 0\n```"

    async def load_adapter(self, name, path) -> None:  # noqa: ANN001
        self.loaded.append(name)

    async def unload_adapter(self, name) -> None:  # noqa: ANN001
        return None

    async def aclose(self) -> None:
        return None


@pytest.fixture
def fake_world(monkeypatch, tmp_path):
    sources = {
        "math": lambda: [_math(i) for i in range(1, 40)],
        "social": lambda: [_choice(i) for i in range(1, 40)],
    }
    benchmarks = {
        "fake_math": lambda: [_math(i) for i in range(100, 120)],
        "fake_social": lambda: [_choice(i) for i in range(100, 120)],
    }
    monkeypatch.setattr(tasks_module, "TRAIN_SOURCES", sources)
    for name, loader in benchmarks.items():
        monkeypatch.setitem(tasks_module.BENCHMARKS, name, loader)
    assert experiment_module.BENCHMARKS is tasks_module.BENCHMARKS

    def config(path: str, name: str):
        cfg = load_config(path)
        cfg.data.domains = ["math", "social"]
        cfg.data.validation_per_domain = 4
        cfg.experiment.cycles = 2
        cfg.experiment.rounds_per_cycle = 6
        cfg.experiment.output_dir = tmp_path / "runs"
        cfg.experiment.name = name
        cfg.training.enabled = False
        cfg.memory.enabled = False
        cfg.evaluation.benchmarks = list(benchmarks)
        cfg.evaluation.max_items = 10
        return cfg

    return config, tmp_path / "runs"


def _experiment(cfg) -> Experiment:  # noqa: ANN001
    exp = Experiment(cfg)
    exp.client = FakeClient()  # type: ignore[assignment]
    return exp


async def test_full_pipeline_offline(fake_world):
    config, runs = fake_world
    await _experiment(config("config/general.yaml", "pair-r1")).run()
    await _experiment(config("config/general_solo.yaml", "solo-r1")).run()
    await _experiment(config("config/general.yaml", "baseline")).baseline()

    report = json.loads((runs / "pair-r1" / "report.json").read_text())
    assert len(report["cycles"]) == 2
    c = report["cycles"][0]
    assert set(c["validation"]["a"]) == {"math", "social"}
    assert set(c["validation_specialists"]["a"]) == {
        "logical",
        "creative",
        "skeptical",
        "empathetic",
    }
    assert c["preference_pairs"]["a"]["logical"] > 0
    assert (
        c["domain_records"]["a"]["logical"]["math"]
        > c["domain_records"]["a"]["creative"]["math"]
    )
    assert c["lessons"] == {}  # memory disabled
    modes = {e["mode"] for e in report["final_benchmarks"]}
    assert modes == {
        "agent",
        "specialist",
    }  # no adapters trained, so no plain fine-tuned
    rounds = [
        json.loads(line)
        for line in (runs / "pair-r1" / "rounds" / "cycle_1.jsonl").open()
    ]
    assert (
        len(rounds) == 6
        and len(rounds[0]["results"]["a"]["decisions"][0]["calls"]) == 9
    )

    base = json.loads((runs / "baseline" / "report.json").read_text())
    assert {e["mode"] for e in base["baseline_benchmarks"]} == {
        "agent",
        "specialist",
        "plain",
    }

    summary = analyze(runs, runs.parent / "analysis").read_text()
    assert (
        "Untrained · whole agent" in summary and "| pair | logical | math |" in summary
    )

    store = RunStore(runs)
    assert {r["kind"] for r in store.runs()} == {"baseline", "run"}
    assert store.rounds("pair-r1", 1)[0]["domain"] in {"math", "social"}
    assert store.round("pair-r1", 1, 0)["task"]["task_id"]
