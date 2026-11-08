"""The full run: (compete → sleep → validate) × cycles → final benchmarks.

Benchmarks of the untrained system are identical for every run, so they are
measured once with `baseline()` (cli: `cogarch baseline`) rather than in each run.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

from src.config import Config
from src.environment import Environment
from src.eval import scoring
from src.execution.sandbox import Sandbox
from src.inference.agent import Agent, Role, derive_seed
from src.inference.client import InferenceClient
from src.inference.prompts import load_coordinator, load_specialist
from src.memory.controller import AgentMemory
from src.memory.embedder import Embedder, SentenceTransformerEmbedder
from src.models.core import (
    BenchmarkScores,
    CycleMetrics,
    ExperimentReport,
    RoundResult,
    Task,
)
from src.tasks import BENCHMARKS, EVALPLUS_BENCHMARKS, Curriculum, benchmark_subset
from src.training.community import play_round, remember_round
from src.training.preferences import (
    COORDINATOR,
    DomainRecords,
    build_pairs,
    discussion_stats,
    specialist_accuracy,
)
from src.training.registry import AdapterRegistry, AdapterVersion
from src.training.trainer import make_job, run_jobs, training_summary, write_dataset

MAX_CASES_PER_REFLECTION = 8


@dataclass
class ValidationResult:
    whole: dict[str, dict[str, float]] = field(default_factory=dict)
    specialists: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    divergence: float | None = None
    disagreement: float | None = None


def _solo(agent: Agent, specialist: str) -> scoring.Answerer:
    async def answer(task: Task, sample: int) -> str:
        return await agent.solo_answer(specialist, task, sample)

    return answer


def _normalise(answer: str) -> str:
    return "\n".join(line.rstrip() for line in answer.strip().splitlines())


class Experiment:
    def __init__(self, cfg: Config, console: Console | None = None) -> None:
        self.cfg = cfg
        self.console = console or Console()
        self.run_dir = cfg.run_dir
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
        self.environment = Environment(self.sandbox, cfg.environment.feedback)
        self.registry = AdapterRegistry(self.run_dir / "adapters" / "registry.json")
        self.specialist_personas = {
            n: load_specialist(n) for n in cfg.agents.specialists
        }
        self.coordinator_persona = load_coordinator()
        self.agents: dict[str, Agent] = {}
        self.records: dict[str, DomainRecords] = {}
        self.session = format(int(time.time()), "x")

    # ── setup ────────────────────────────────────────────────────────────────

    def _build_agents(self, with_memory: bool = True) -> None:
        embedder: Embedder | None = None
        if with_memory and self.cfg.memory.enabled:
            embedder = SentenceTransformerEmbedder(self.cfg.memory.embedding_model)
        base = self.cfg.model.served_name
        for name in self.cfg.agents.names:
            memory = None
            if embedder is not None:
                m = self.cfg.memory
                memory = AgentMemory(
                    self.run_dir / "memory" / name,
                    embedder,
                    working_capacity=m.working_capacity,
                    retrieve_k=m.retrieve_k,
                    min_similarity=m.min_similarity,
                    max_episodic=m.max_episodic,
                    lessons_in_prompt=m.lessons_in_prompt,
                )
            self.agents[name] = Agent(
                name=name,
                seed=self.cfg.agents.seeds[name],
                specialists={
                    n: Role(p, base) for n, p in self.specialist_personas.items()
                },
                coordinator=Role(self.coordinator_persona, base),
                client=self.client,
                environment=self.environment,
                memory=memory,
                max_attempts=self.cfg.environment.max_attempts,
                max_tokens=self.cfg.model.max_tokens,
                probes_per_problem=self.cfg.environment.probes_per_problem,
                discussion=self.cfg.agents.discussion,
            )
            self.records[name] = DomainRecords(list(self.specialist_personas))

    def _curriculum(self) -> Curriculum:
        return Curriculum(
            self.cfg.data.domains,
            self.cfg.data.validation_per_domain,
            self.cfg.experiment.rounds_per_cycle,
            self.cfg.experiment.cycles,
            self.cfg.data.split_seed,
        )

    # ── measurement ──────────────────────────────────────────────────────────

    async def _accuracy(
        self, answerer: scoring.Answerer, tasks: list[Task], samples: int
    ) -> tuple[dict[str, float], dict[tuple[str, int], tuple[str, bool]]]:
        """Per-domain accuracy and (task, sample) -> (answer, correct)."""
        jobs = [(t, i) for t in tasks for i in range(samples)]
        answers = await asyncio.gather(*(answerer(t, i) for t, i in jobs))
        verdicts = await asyncio.gather(
            *(self.environment.judge(t, a) for (t, _), a in zip(jobs, answers))
        )
        by_domain: dict[str, list[bool]] = defaultdict(list)
        detail: dict[tuple[str, int], tuple[str, bool]] = {}
        for (t, i), a, v in zip(jobs, answers, verdicts):
            by_domain[t.domain].append(v.passed)
            detail[(t.task_id, i)] = (a, v.passed)
        return {d: sum(v) / len(v) for d, v in by_domain.items()}, detail

    async def _validate(self, tasks: list[Task]) -> ValidationResult:
        """Whole-agent and solo-specialist accuracy per domain, and how much the
        agents' answers differ."""
        k = self.cfg.evaluation.validation_samples
        result = ValidationResult()
        details = {}
        for name, agent in self.agents.items():
            result.whole[name], details[name] = await self._accuracy(
                agent.answer, tasks, k
            )
            result.specialists[name] = {}
            for s in agent.specialists:
                result.specialists[name][s], _ = await self._accuracy(
                    _solo(agent, s), tasks, 1
                )
        names = list(self.agents)
        if len(names) > 1:
            pairs = list(itertools.combinations(names, 2))
            keys = [(t.task_id, 0) for t in tasks]
            n = len(pairs) * len(keys)
            result.divergence = (
                sum(
                    _normalise(details[a][key][0]) != _normalise(details[b][key][0])
                    for a, b in pairs
                    for key in keys
                )
                / n
            )
            result.disagreement = (
                sum(
                    details[a][key][1] != details[b][key][1]
                    for a, b in pairs
                    for key in keys
                )
                / n
            )
        return result

    async def _benchmarks(self, stage: str, agents: list[str]) -> list[BenchmarkScores]:
        ev = self.cfg.evaluation
        out_dir = self.run_dir / "benchmarks" / stage
        results: list[BenchmarkScores] = []

        async def run(
            system: str, mode: str, answerer: scoring.Answerer, samples: int, full: bool
        ) -> None:
            entry = BenchmarkScores(system=system, mode=mode)
            for bench in ev.benchmarks:
                tasks = BENCHMARKS[bench]()
                if not full and bench not in EVALPLUS_BENCHMARKS:
                    tasks = benchmark_subset(tasks, ev.max_items, ev.subset_seed)
                self.console.print(
                    f"  {stage} · {mode} · {system} · {bench} ({len(tasks)})"
                )
                entry.scores[bench] = await scoring.score(
                    bench,
                    tasks,
                    answerer,
                    samples,
                    out_dir / mode / system.replace("/", "_") / f"{bench}.jsonl",
                )
            results.append(entry)

        for name in agents:
            agent = self.agents[name]
            await run(f"agent {name}", "agent", agent.answer, ev.samples, full=False)
            if ev.specialists:
                for s in agent.specialists:
                    await run(
                        f"{name}/{s}", "specialist", _solo(agent, s), 1, full=False
                    )
        if ev.plain:
            models = (
                {"base": self.cfg.model.served_name}
                if stage == "baseline"
                else {
                    f"{name}/{s}": role.model
                    for name in agents
                    for s, role in self.agents[name].specialists.items()
                    if role.model != self.cfg.model.served_name
                }
            )
            for system, model in models.items():
                await run(
                    system,
                    "plain",
                    scoring.plain_answerer(
                        self.client, model, self.cfg.model.max_tokens
                    ),
                    1,
                    full=True,
                )
        return results

    # ── waking ───────────────────────────────────────────────────────────────

    async def _compete(self, cycle: int, tasks: list[Task]) -> list[RoundResult]:
        rounds: list[RoundResult] = []
        log = self.run_dir / "rounds" / f"cycle_{cycle}.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        size = self.cfg.experiment.round_batch_size
        with open(log, "w") as f:
            for start in range(0, len(tasks), size):
                batch = tasks[start : start + size]
                results = await asyncio.gather(
                    *(
                        play_round(
                            self.agents,
                            t,
                            cycle,
                            self.sandbox,
                            self.cfg.agents.share_discoveries,
                        )
                        for t in batch
                    )
                )
                for r in results:
                    for agent in self.agents.values():
                        await remember_round(agent, r)
                    f.write(r.model_dump_json() + "\n")
                    f.flush()
                rounds.extend(results)
                self.console.print(f"    {len(rounds)}/{len(tasks)} problems")
        return rounds

    # ── sleep ────────────────────────────────────────────────────────────────

    async def _reflect(
        self, cycle: int, rounds: list[RoundResult], metrics: CycleMetrics
    ) -> None:
        banned = {r.task.entry_point for r in rounds if r.task.entry_point} | {
            r.task.task_id for r in rounds
        }
        for name, agent in self.agents.items():
            if agent.memory is None:
                continue
            cases: list[str] = []
            for r in rounds:
                known = r.known.get(name)
                wrong = [d for d in r.results[name].decisions if not d.result.passed]
                if known is None or not wrong:
                    continue
                source = (
                    "your team" if known.found_by == name else f"agent {known.found_by}"
                )
                cases.append(
                    f"### Case {len(cases) + 1} ({r.task.domain})\nProblem:\n{r.task.prompt}\n"
                    f"Your team's wrong answer:\n{wrong[-1].coordinator_output[:1500]}\n"
                    f"Correct answer (found by {source}):\n{known.response[:1500]}"
                )
            cases = cases[-MAX_CASES_PER_REFLECTION:]
            if cases:
                numbered = agent.memory.lessons()
                ops, call = await agent.reflect(
                    [e.concept for e in numbered],
                    cases,
                    self.cfg.memory.lessons_per_sleep,
                    cycle,
                )
                metrics.lessons[name] = agent.memory.apply_reflection(
                    ops, numbered, banned
                )
                metrics.reflections[name] = call
            agent.memory.prune()
            agent.memory.end_session()

    async def _train(
        self, cycle: int, rounds: list[RoundResult], metrics: CycleMetrics
    ) -> list[tuple[str, str]]:
        jobs = {}
        for name, agent in self.agents.items():
            pairs = build_pairs(rounds, name, self.records[name])
            metrics.domain_records[name] = self.records[name].snapshot()
            metrics.preference_pairs[name] = {role: len(p) for role, p in pairs.items()}
            metrics.pairs_by_domain[name] = {
                role: dict(Counter(pair.domain for pair in p))
                for role, p in pairs.items()
            }
            metrics.pair_sources[name] = dict(
                Counter(pair.source for p in pairs.values() for pair in p)
            )
            if not self.cfg.training.enabled:
                continue
            for role, role_pairs in pairs.items():
                if len(role_pairs) < self.cfg.training.min_pairs:
                    continue
                out = self.run_dir / "adapters" / name / role / f"cycle_{cycle}"
                dataset = out.with_name(f"cycle_{cycle}_pairs.jsonl")
                write_dataset(role_pairs, dataset)
                current = self.registry.current(name, role)
                jobs[f"{name}/{role}"] = make_job(
                    self.cfg.training,
                    self.cfg.model.base,
                    Path(current.path) if current else None,
                    dataset,
                    out,
                    seed=derive_seed(agent.seed, role, cycle),
                )
        if not jobs:
            return []
        self.console.print(f"  Training {len(jobs)} adapters")
        await run_jobs(
            jobs, self.cfg.training.slots_dir, self.cfg.training.global_slots
        )

        updated = []
        for key, job in jobs.items():
            name, role = key.split("/")
            metrics.training.setdefault(name, {})[role] = training_summary(
                Path(job.output_dir), metrics.preference_pairs[name][role]
            )
            served = f"{self.cfg.experiment.name}-{self.session}-{name}-{role}-c{cycle}"
            await self.client.load_adapter(served, Path(job.output_dir))
            self.registry.push(
                name,
                role,
                AdapterVersion(cycle=cycle, served_name=served, path=job.output_dir),
            )
            self.agents[name].roles[role].model = served
            updated.append((name, role))
            history = self.registry.history(name, role)
            if len(history) >= 3:
                await self.client.unload_adapter(history[-3].served_name)
        return updated

    async def _rollback(
        self, cycle: int, agent_name: str, updated: list[tuple[str, str]]
    ) -> None:
        for name, role in updated:
            if name != agent_name:
                continue
            bad = self.registry.current(name, role)
            previous = self.registry.rollback(name, role, cycle)
            self.agents[name].roles[role].model = (
                previous.served_name if previous else self.cfg.model.served_name
            )
            if bad is not None and bad.cycle == cycle:
                await self.client.unload_adapter(bad.served_name)

    # ── entry points ─────────────────────────────────────────────────────────

    async def baseline(self) -> Path:
        """Benchmarks of the untrained system, once: the whole brain, each
        specialist with its persona, and the plain base model."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._build_agents(with_memory=False)
        first = self.cfg.agents.names[0]
        report = ExperimentReport(config=self.cfg.model_dump(mode="json"))
        report.baseline_benchmarks = await self._benchmarks("baseline", [first])
        path = self.run_dir / "report.json"
        path.write_text(report.model_dump_json(indent=2))
        await self.client.aclose()
        return path

    async def run(self) -> ExperimentReport:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        report = ExperimentReport(config=self.cfg.model_dump(mode="json"))
        curriculum = self._curriculum()
        self._build_agents()
        self.console.print(
            f"[bold]{self.cfg.experiment.name}[/bold]: agents {list(self.agents)} · "
            f"domains {self.cfg.data.domains} · {len(curriculum.validation)} validation tasks"
        )

        self.console.print("[bold yellow]Baseline validation[/bold yellow]")
        start = await self._validate(curriculum.validation)
        report.baseline_validation = start.whole
        report.baseline_validation_specialists = start.specialists
        best_mean = {n: sum(v.values()) / len(v) for n, v in start.whole.items()}
        self._save(report)

        for cycle in range(1, self.cfg.experiment.cycles + 1):
            self.console.print(f"[bold cyan]Cycle {cycle}[/bold cyan]")
            metrics = CycleMetrics(cycle=cycle)
            rounds = await self._compete(cycle, curriculum.cycle(cycle))
            self._summarise(rounds, metrics)

            self.console.print("  Sleep")
            await self._reflect(cycle, rounds, metrics)
            updated = await self._train(cycle, rounds, metrics)

            validation = await self._validate(curriculum.validation)
            metrics.validation = validation.whole
            metrics.validation_specialists = validation.specialists
            metrics.answer_divergence = validation.divergence
            metrics.outcome_disagreement = validation.disagreement
            changed = {n for n, _ in updated}
            for name, by_domain in validation.whole.items():
                mean = sum(by_domain.values()) / len(by_domain)
                if (
                    name in changed
                    and mean < best_mean[name] - self.cfg.training.rollback_threshold
                ):
                    self.console.print(
                        f"  [yellow]Agent {name} regressed — rolling back[/yellow]"
                    )
                    await self._rollback(cycle, name, updated)
                    metrics.rolled_back.append(name)
                else:
                    best_mean[name] = max(best_mean[name], mean)
            report.cycles.append(metrics)
            self._save(report)
            self.console.print(f"  Validation: {validation.whole}")

        self.console.print("[bold green]Final benchmarks[/bold green]")
        report.final_benchmarks = await self._benchmarks("final", list(self.agents))
        self._save(report)
        await self.client.aclose()
        return report

    def _summarise(self, rounds: list[RoundResult], metrics: CycleMetrics) -> None:
        metrics.rounds = len(rounds)
        specialists = list(self.specialist_personas)
        for name in self.agents:
            by_domain: dict[str, list[bool]] = defaultdict(list)
            for r in rounds:
                by_domain[r.task.domain].append(r.results[name].solved)
            metrics.solved_rate[name] = {
                d: sum(v) / len(v) for d, v in by_domain.items()
            }
            metrics.learned_from_peer[name] = sum(
                1 for r in rounds if name in r.known and r.known[name].found_by != name
            )
            changed, fixed = discussion_stats(rounds, name)
            metrics.discussion_changed[name] = changed
            metrics.discussion_fixed[name] = fixed
            metrics.specialist_accuracy[name] = specialist_accuracy(
                rounds, name, specialists
            )

    def _save(self, report: ExperimentReport) -> None:
        (self.run_dir / "report.json").write_text(report.model_dump_json(indent=2))


__all__ = ["Experiment", "COORDINATOR"]
