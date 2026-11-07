"""The full run: baseline → (compete → sleep → validate) × cycles → final eval."""

from __future__ import annotations

import asyncio
import itertools
from collections import Counter
from pathlib import Path

from rich.console import Console

from src.config import Config
from src.eval import evalplus_eval
from src.eval.benchmarks import mbpp
from src.execution.sandbox import Sandbox
from src.inference.agent import Agent, Role, derive_seed
from src.inference.client import InferenceClient
from src.inference.prompts import load_coordinator, load_specialist
from src.memory.controller import AgentMemory
from src.memory.embedder import Embedder, SentenceTransformerEmbedder
from src.models.code import (
    CodeProblem,
    CycleMetrics,
    EvalScore,
    ExperimentReport,
    RoundResult,
)
from src.training.community import play_round, remember_round
from src.training.preferences import build_pairs
from src.training.registry import AdapterRegistry, AdapterVersion
from src.training.trainer import make_job, run_jobs, write_dataset

MAX_EXPERIENCES_PER_REFLECTION = 30


def _normalise(code: str) -> str:
    return "\n".join(line.rstrip() for line in code.strip().splitlines())


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
        self.registry = AdapterRegistry(self.run_dir / "adapters" / "registry.json")
        self.specialist_personas = {
            n: load_specialist(n) for n in cfg.agents.specialists
        }
        self.coordinator_persona = load_coordinator()
        self.agents: dict[str, Agent] = {}

    # ── setup ────────────────────────────────────────────────────────────────

    def _build_agents(self) -> None:
        embedder: Embedder | None = None
        if self.cfg.memory.enabled:
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
                sandbox=self.sandbox,
                memory=memory,
                max_attempts=self.cfg.environment.max_attempts,
                max_tokens=self.cfg.model.max_tokens,
                probes_per_problem=self.cfg.environment.probes_per_problem,
            )

    def _load_data(self) -> tuple[list[CodeProblem], list[CodeProblem]]:
        train, validation = mbpp.split(
            mbpp.load_pool(), self.cfg.data.validation_size, self.cfg.data.split_seed
        )
        needed = self.cfg.experiment.cycles * self.cfg.experiment.rounds_per_cycle
        if needed > len(train):
            raise ValueError(
                f"{needed} training problems needed but only {len(train)} available"
            )
        return train, validation

    # ── measurement ──────────────────────────────────────────────────────────

    async def _validate(
        self, problems: list[CodeProblem]
    ) -> tuple[dict[str, float], float | None, float | None]:
        """Pass@1 per agent, plus how much the agents' answers differ."""
        names = list(self.agents)
        answers = await asyncio.gather(
            *(
                asyncio.gather(*(self.agents[n].answer(p) for p in problems))
                for n in names
            )
        )
        verdicts = await asyncio.gather(
            *(
                asyncio.gather(
                    *(
                        self.sandbox.run_tests(code, p)
                        for code, p in zip(codes, problems)
                    )
                )
                for codes in answers
            )
        )
        passed = {n: [v.passed for v in vs] for n, vs in zip(names, verdicts)}
        scores = {n: sum(ok) / len(problems) for n, ok in passed.items()}
        if len(names) < 2:
            return scores, None, None
        code_by = dict(zip(names, answers))
        pairs = list(itertools.combinations(names, 2))
        divergence = sum(
            _normalise(x) != _normalise(y)
            for a, b in pairs
            for x, y in zip(code_by[a], code_by[b])
        ) / (len(pairs) * len(problems))
        disagreement = sum(
            x != y for a, b in pairs for x, y in zip(passed[a], passed[b])
        ) / (len(pairs) * len(problems))
        return scores, divergence, disagreement

    async def _evalplus(self, stage: str) -> list[EvalScore]:
        scores = []
        for name, agent in self.agents.items():
            score = EvalScore(agent=name)
            for benchmark in self.cfg.evaluation.benchmarks:
                self.console.print(f"  EvalPlus {benchmark} · agent {name} ({stage})")
                score.scores[benchmark] = await evalplus_eval.evaluate(
                    agent, benchmark, self.run_dir / "evalplus" / stage
                )
            scores.append(score)
        return scores

    # ── waking: competition ──────────────────────────────────────────────────

    async def _compete(
        self, cycle: int, problems: list[CodeProblem]
    ) -> list[RoundResult]:
        rounds: list[RoundResult] = []
        log = self.run_dir / "rounds" / f"cycle_{cycle}.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        size = self.cfg.experiment.round_batch_size
        with open(log, "w") as f:
            for start in range(0, len(problems), size):
                batch = problems[start : start + size]
                results = await asyncio.gather(
                    *(
                        play_round(
                            self.agents,
                            p,
                            cycle,
                            self.sandbox,
                            self.cfg.agents.share_discoveries,
                        )
                        for p in batch
                    )
                )
                for r in results:
                    for agent in self.agents.values():
                        await remember_round(agent, r)
                    f.write(r.model_dump_json() + "\n")
                rounds.extend(results)
                self.console.print(f"    {len(rounds)}/{len(problems)} problems")
        return rounds

    # ── sleep ────────────────────────────────────────────────────────────────

    async def _reflect(self, cycle: int, metrics: CycleMetrics) -> None:
        for name, agent in self.agents.items():
            if agent.memory is None:
                continue
            episodes = agent.memory.episodes_from_cycle(cycle)[
                :MAX_EXPERIENCES_PER_REFLECTION
            ]
            if episodes:
                lessons = await agent.reflect(
                    "\n\n".join(e.content for e in episodes),
                    self.cfg.memory.lessons_per_sleep,
                    cycle,
                )
                agent.memory.add_lessons(lessons)
                metrics.lessons[name] = lessons
            agent.memory.prune()
            agent.memory.end_session()

    async def _train(
        self, cycle: int, rounds: list[RoundResult], metrics: CycleMetrics
    ) -> list[tuple[str, str]]:
        """Fine-tune every role with enough pairs; returns the (agent, role)s updated."""
        jobs = {}
        for name, agent in self.agents.items():
            pairs = build_pairs(rounds, name, self.specialist_personas)
            metrics.preference_pairs[name] = {role: len(p) for role, p in pairs.items()}
            metrics.pair_sources[name] = dict(
                Counter(pair.source for p in pairs.values() for pair in p)
            )
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
        await run_jobs(jobs, self.cfg.training.parallel_jobs)

        updated = []
        for key, job in jobs.items():
            name, role = key.split("/")
            served = f"{self.cfg.experiment.name}-{name}-{role}-c{cycle}"
            await self.client.load_adapter(served, Path(job.output_dir))
            self.registry.push(
                name,
                role,
                AdapterVersion(cycle=cycle, served_name=served, path=job.output_dir),
            )
            self.agents[name].roles[role].model = served
            updated.append((name, role))
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

    # ── main loop ────────────────────────────────────────────────────────────

    async def run(self) -> ExperimentReport:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        report = ExperimentReport(config=self.cfg.model_dump(mode="json"))
        train, validation = self._load_data()
        self._build_agents()
        self.console.print(
            f"[bold]{self.cfg.experiment.name}[/bold]: agents {list(self.agents)} · "
            f"{len(train)} train / {len(validation)} validation problems"
        )

        self.console.print("[bold yellow]Baseline[/bold yellow]")
        best, _, _ = await self._validate(validation)
        report.baseline_validation = dict(best)
        report.baseline_eval = await self._evalplus("baseline")
        self._save(report)

        rounds_per_cycle = self.cfg.experiment.rounds_per_cycle
        for cycle in range(1, self.cfg.experiment.cycles + 1):
            self.console.print(f"[bold cyan]Cycle {cycle}[/bold cyan]")
            problems = train[(cycle - 1) * rounds_per_cycle : cycle * rounds_per_cycle]
            metrics = CycleMetrics(cycle=cycle)

            rounds = await self._compete(cycle, problems)
            self._summarise(rounds, metrics)

            self.console.print("  Sleep")
            await self._reflect(cycle, metrics)
            updated = (
                await self._train(cycle, rounds, metrics)
                if self.cfg.training.enabled
                else []
            )

            scores, divergence, disagreement = await self._validate(validation)
            for name, score in scores.items():
                if score < best[name] - self.cfg.training.rollback_threshold:
                    self.console.print(
                        f"  [yellow]Agent {name} regressed — rolling back[/yellow]"
                    )
                    await self._rollback(cycle, name, updated)
                    metrics.rolled_back.append(name)
                else:
                    best[name] = max(best[name], score)
            metrics.validation_pass_at_1 = scores
            metrics.code_divergence = divergence
            metrics.outcome_disagreement = disagreement
            report.cycles.append(metrics)
            self._save(report)
            self.console.print(f"  Validation Pass@1: {scores}")

        self.console.print("[bold green]Final evaluation[/bold green]")
        report.final_eval = await self._evalplus("final")
        self._save(report)
        await self.client.aclose()
        return report

    def _summarise(self, rounds: list[RoundResult], metrics: CycleMetrics) -> None:
        metrics.rounds = len(rounds)
        for name in self.agents:
            results = [r.results[name] for r in rounds]
            metrics.solved_rate[name] = sum(r.solved for r in results) / len(rounds)
            metrics.learned_from_peer[name] = sum(
                1
                for r in rounds
                if name in r.known_solution and r.known_solution[name].found_by != name
            )
            metrics.coordinator_choices[name] = dict(
                Counter(s for r in results for s in r.decisions[0].based_on)
            )

    def _save(self, report: ExperimentReport) -> None:
        (self.run_dir / "report.json").write_text(report.model_dump_json(indent=2))
