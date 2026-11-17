"""The full run: (compete → sleep → validate) × cycles → final benchmarks.

Benchmarks of the untrained system are identical for every run, so they are
measured once with `baseline()` (cli: `cogarch baseline`) rather than in each run.
"""

from __future__ import annotations

import asyncio
import itertools
import math
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

from src import calllog
from src.config import Config
from src.environment import Environment, extract_answer
from src.eval import scoring
from src.events import EventLog
from src.execution.sandbox import Sandbox, behaves_alike
from src.grading import build_grader
from src.inference.agent import Agent, Role, derive_seed
from src.inference.client import InferenceClient
from src.inference.prompts import load_coordinator, load_specialist, neutral_persona
from src.memory.controller import AgentMemory
from src.memory.embedder import Embedder, SentenceTransformerEmbedder
from src.models.core import (
    BenchmarkScores,
    CycleMetrics,
    ExperimentReport,
    PreferencePair,
    Proposal,
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
from src.training.trainer import (
    make_job,
    replay_rows,
    run_jobs,
    training_summary,
    write_dataset,
)

MAX_CASES_PER_REFLECTION = 8


@dataclass
class ValidationResult:
    whole: dict[str, dict[str, float]] = field(default_factory=dict)
    specialists: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    specialists_no_persona: dict[str, dict[str, dict[str, float]]] = field(
        default_factory=dict
    )
    divergence: float | None = None
    disagreement: float | None = None


def _solo(agent: Agent, specialist: str, persona: bool = True) -> scoring.Answerer:
    async def answer(task: Task, sample: int) -> str:
        return await agent.solo_answer(specialist, task, sample, persona)

    return answer


async def _extracted(task: Task, reply: str) -> str:
    return extract_answer(task, reply)


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
        self.registry = AdapterRegistry(self.run_dir / "adapters" / "registry.json")
        self.specialist_personas = {
            n: (
                load_specialist(n)
                if cfg.agents.personas == "full"
                else neutral_persona(load_specialist(n))
            )
            for n in cfg.agents.specialists
        }
        self.coordinator_persona = load_coordinator()
        self.agents: dict[str, Agent] = {}
        self.records: dict[str, DomainRecords] = {}
        self.session = format(int(time.time()), "x")
        self._measured = 0
        self.events = EventLog(self.run_dir / "events.jsonl", cfg.events.stream_tokens)
        calllog.open_log(self.run_dir / "calls.jsonl.gz")

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
                    retrieve_k=m.retrieve_k,
                    min_similarity=m.min_similarity,
                    max_episodic=m.max_episodic,
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
                discussion_style=self.cfg.agents.discussion_style,
                selector=self.cfg.agents.selector,
                allow_retry=self.cfg.agents.allow_retry,
                memory_at_eval=self.cfg.evaluation.memory,
                events=self.events,
            )
            self.records[name] = DomainRecords(
                list(self.specialist_personas), conscience=self.cfg.training.conscience
            )

    def _curriculum(self) -> Curriculum:
        return Curriculum(
            self.cfg.data.domains,
            self.cfg.data.validation_per_domain,
            self.cfg.experiment.rounds_per_cycle,
            self.cfg.experiment.cycles,
            self.cfg.data.split_seed,
        )

    def _phase(self, phase: str, cycle: int) -> None:
        self.events.emit("phase", phase=phase, cycle=cycle)
        calllog.update(phase=phase, cycle=cycle)
        calllog.record("phase")

    # ── measurement ──────────────────────────────────────────────────────────

    def _team(self, agent: Agent) -> scoring.Answerer:
        """The whole brain as an answerer that also logs its full decision with
        every specialist's answer graded, before and after any retry. Code
        answers on benchmarks without hidden tests here are graded afterwards
        with EvalPlus from the logged code."""

        async def answer(task: Task, sample: int) -> str:
            d = await agent.decide(task, sample)
            first, final = await asyncio.gather(
                self._grade_all(task, d.round1), self._grade_all(task, d.round2)
            )
            calllog.record(
                "team_decision",
                agent=agent.name,
                task_id=task.task_id,
                sample=sample,
                decision=d.model_dump(exclude={"calls", "specialist_prompts"}),
                first_correct=first,
                final_correct=final,
            )
            return d.submitted

        return answer

    async def _grade_all(
        self, task: Task, proposals: list[Proposal]
    ) -> dict[str, bool | None]:
        async def one(p: Proposal) -> bool | None:
            if task.kind != "code":
                return await self.grader.correct(task, p.response, task.answer)
            if not task.tests:
                return None
            return (await self.sandbox.run_tests(p.answer, task)).passed

        verdicts = await asyncio.gather(*(one(p) for p in proposals))
        return {p.specialist: v for p, v in zip(proposals, verdicts)}

    async def _accuracy(
        self, answerer: scoring.Answerer, tasks: list[Task], samples: int
    ) -> tuple[dict[str, float], dict[tuple[str, int], tuple[str, bool]]]:
        """Per-domain accuracy and (task, sample) -> (extracted answer, correct)."""
        jobs = [(t, i) for t in tasks for i in range(samples)]

        async def counted(t: Task, i: int) -> str:
            with calllog.tagged(task_id=t.task_id, sample=i):
                reply = await answerer(t, i)
            self._measured += 1
            self.events.emit("measure_progress", done=self._measured)
            return reply

        replies = await asyncio.gather(*(counted(t, i) for t, i in jobs))
        verdicts = await asyncio.gather(
            *(self.environment.judge(t, r) for (t, _), r in zip(jobs, replies))
        )
        answers = await asyncio.gather(
            *(
                self.grader.code(t, r) if t.kind == "code" else _extracted(t, r)
                for (t, _), r in zip(jobs, replies)
            )
        )
        by_domain: dict[str, list[bool]] = defaultdict(list)
        detail: dict[tuple[str, int], tuple[str, bool]] = {}
        for (t, i), r, a, v in zip(jobs, replies, answers, verdicts):
            by_domain[t.domain].append(v.passed)
            detail[(t.task_id, i)] = (a, v.passed)
            calllog.record(
                "validated",
                task_id=t.task_id,
                sample=i,
                domain=t.domain,
                reply=r,
                answer=a,
                verdict=v.model_dump(),
            )
        return {d: sum(v) / len(v) for d, v in by_domain.items()}, detail

    async def _validate(
        self, tasks: list[Task], specialists: bool = True
    ) -> ValidationResult:
        """Whole-agent accuracy per domain and how much the agents' answers
        differ; with `specialists`, also each specialist alone, with and without
        its persona."""
        k = self.cfg.evaluation.validation_samples
        result = ValidationResult()
        self._measured = 0
        solo_count = 2 * len(self.specialist_personas) if specialists else 0
        per_agent = len(tasks) * (k + solo_count)
        self.events.emit("measure_start", total=per_agent * len(self.agents))
        details = {}
        names = list(self.agents)
        wholes = await asyncio.gather(
            *(
                calllog.within(
                    {"mode": "agent", "system": f"agent {n}"},
                    self._accuracy(self._team(self.agents[n]), tasks, k),
                )
                for n in names
            )
        )
        pairs = (
            [(n, s) for n in names for s in self.agents[n].specialists]
            if specialists
            else []
        )
        solos = await asyncio.gather(
            *(
                calllog.within(
                    {
                        "mode": "specialist" if persona else "specialist_neutral",
                        "system": f"{n}/{s}",
                    },
                    self._accuracy(_solo(self.agents[n], s, persona), tasks, 1),
                )
                for persona in (True, False)
                for n, s in pairs
            )
        )
        with_persona, without = solos[: len(pairs)], solos[len(pairs) :]
        for name, (whole, detail) in zip(names, wholes):
            result.whole[name], details[name] = whole, detail
        for (n, s), (acc, _), (plain, _) in zip(pairs, with_persona, without):
            result.specialists.setdefault(n, {})[s] = acc
            result.specialists_no_persona.setdefault(n, {})[s] = plain
        self.events.emit(
            "validation_done",
            whole=result.whole,
            specialists=result.specialists,
            specialists_no_persona=result.specialists_no_persona,
        )
        names = list(self.agents)
        if len(names) > 1:
            pairs = list(itertools.combinations(names, 2))
            keys = [(t.task_id, 0) for t in tasks]
            total = len(pairs) * len(keys)
            result.divergence = (
                sum(
                    _normalise(details[a][key][0]) != _normalise(details[b][key][0])
                    for a, b in pairs
                    for key in keys
                )
                / total
            )
            result.disagreement = (
                sum(
                    details[a][key][1] != details[b][key][1]
                    for a, b in pairs
                    for key in keys
                )
                / total
            )
        return result

    async def _benchmarks(self, stage: str, agents: list[str]) -> list[BenchmarkScores]:
        ev = self.cfg.evaluation
        out_dir = self.run_dir / "benchmarks" / stage
        results: list[BenchmarkScores] = []

        async def run(
            system: str, mode: str, answerer: scoring.Answerer, samples: int
        ) -> None:
            entry = BenchmarkScores(system=system, mode=mode)
            for bench in ev.benchmarks:
                tasks = BENCHMARKS[bench]()
                if bench not in EVALPLUS_BENCHMARKS:
                    tasks = benchmark_subset(tasks, ev.max_items, ev.subset_seed)
                self.console.print(
                    f"  {stage} · {mode} · {system} · {bench} ({len(tasks)})"
                )
                before = dict(self.client.usage)
                with calllog.tagged(
                    stage=stage, mode=mode, system=system, benchmark=bench
                ):
                    entry.scores[bench] = await scoring.score(
                        bench,
                        tasks,
                        answerer,
                        samples,
                        out_dir / mode / system.replace("/", "_") / f"{bench}.jsonl",
                        self.grader,
                    )
                used = {k: self.client.usage[k] - before[k] for k in before}
                entry.scores[bench]["tokens_per_problem"] = (
                    used["prompt_tokens"] + used["completion_tokens"]
                ) / (len(tasks) * samples)
                entry.scores[bench]["calls_per_problem"] = used["calls"] / (
                    len(tasks) * samples
                )
                self.events.emit(
                    "benchmark",
                    stage=stage,
                    mode=mode,
                    system=system,
                    bench=bench,
                    scores=entry.scores[bench],
                )
            results.append(entry)

        for name in agents:
            agent = self.agents[name]
            await run(f"agent {name}", "agent", self._team(agent), ev.samples)
            if ev.specialists:
                for s in agent.specialists:
                    await run(f"{name}/{s}", "specialist", _solo(agent, s), 1)
                    await run(
                        f"{name}/{s}",
                        "specialist_neutral",
                        _solo(agent, s, persona=False),
                        1,
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
                            self.events,
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
                self.events.emit(
                    "progress", cycle=cycle, done=len(rounds), total=len(tasks)
                )
                self.console.print(f"    {len(rounds)}/{len(tasks)} problems")
        return rounds

    # ── sleep ────────────────────────────────────────────────────────────────

    async def _reflect(
        self, cycle: int, rounds: list[RoundResult], metrics: CycleMetrics
    ) -> None:
        """Every problem the agent got wrong and later learned the answer to is
        reflected on, in chunks: each case gets a note on what went wrong,
        added to its episode."""
        for name, agent in self.agents.items():
            if agent.memory is None:
                continue
            failures = [
                r
                for r in rounds
                if name in r.known
                and any(not d.result.passed for d in r.results[name].decisions)
            ]
            chunks = [
                failures[i : i + MAX_CASES_PER_REFLECTION]
                for i in range(0, len(failures), MAX_CASES_PER_REFLECTION)
            ]
            for chunk_number, chunk in enumerate(chunks, start=1):
                cases = [
                    self._case_text(i, r, name) for i, r in enumerate(chunk, start=1)
                ]
                notes, call = await agent.reflect(cases, cycle * 100 + chunk_number)
                metrics.reflections[name] = call
                annotated = []
                for number, note in notes:
                    if 1 <= number <= len(chunk):
                        task_id = chunk[number - 1].task.task_id
                        entry = agent.memory.annotate(task_id, cycle, note)
                        if entry is not None:
                            annotated.append(
                                {"task_id": task_id, "note": note, "entry": entry.id}
                            )
                self.events.emit(
                    "reflection",
                    agent=name,
                    cycle=cycle,
                    chunk=chunk_number,
                    chunks=len(chunks),
                    cases=cases,
                    case_tasks=[r.task.task_id for r in chunk],
                    response=call.response,
                    annotated=annotated,
                )
            agent.memory.prune()

    @staticmethod
    def _case_text(number: int, r: RoundResult, agent: str) -> str:
        known = r.known[agent]
        wrong = [d for d in r.results[agent].decisions if not d.result.passed]
        source = "your team" if known.found_by == agent else f"agent {known.found_by}"
        return (
            f"### Case {number} ({r.task.domain})\nProblem:\n{r.task.prompt}\n"
            f"Your team's wrong answer:\n{wrong[-1].submitted[:1500]}\n"
            f"Correct answer (found by {source}):\n{known.response[:1500]}"
        )

    async def _train(
        self, cycle: int, rounds: list[RoundResult], metrics: CycleMetrics
    ) -> list[tuple[str, str]]:
        jobs = {}
        by_task = {r.task.task_id: r for r in rounds}
        for name, agent in self.agents.items():
            credit_log: list[dict] = []
            pairs = build_pairs(
                rounds,
                name,
                self.records[name],
                winner_take_all=self.cfg.training.winner_take_all,
                blame_adopted=self.cfg.training.blame_adopted,
                surprise_threshold=self.cfg.training.surprise_threshold,
                anchor_ratio=self.cfg.training.anchor_ratio,
                owner_learns=self.cfg.training.owner_learns,
                log=credit_log,
            )
            self.events.emit(
                "credit_rules",
                agent=name,
                cycle=cycle,
                decisions=credit_log,
                records=self.records[name].snapshot(),
                win_rate=dict(self.records[name].win_rate),
            )
            if self.cfg.training.rationalize:
                await self._rationalize(name, pairs, by_task, cycle)
            self.events.emit(
                "pairs",
                agent=name,
                cycle=cycle,
                pairs={
                    role: [pair.model_dump(exclude={"prompt"}) for pair in p]
                    for role, p in pairs.items()
                },
            )
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
                folder = self.run_dir / "adapters" / name / role
                write_dataset(role_pairs, folder / f"cycle_{cycle}_new.jsonl")
                if len(role_pairs) < self.cfg.training.min_pairs:
                    continue
                out = folder / f"cycle_{cycle}"
                dataset = folder / f"cycle_{cycle}_pairs.jsonl"
                replayed = replay_rows(
                    folder,
                    cycle,
                    math.ceil(self.cfg.training.replay_ratio * len(role_pairs)),
                    derive_seed(agent.seed, role, "replay", cycle),
                )
                write_dataset(role_pairs, dataset, replayed)
                metrics.replayed.setdefault(name, {})[role] = len(replayed)
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
        self.events.emit("training", cycle=cycle, jobs=list(jobs))

        def progress(key: str, record: dict) -> None:
            self.events.emit("train_progress", cycle=cycle, job=key, **record)

        await run_jobs(
            jobs, self.cfg.training.slots_dir, self.cfg.training.global_slots, progress
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
            self.events.emit(
                "adapter", cycle=cycle, agent=name, role=role, model=served
            )
            updated.append((name, role))
            history = self.registry.history(name, role)
            if len(history) >= 3:
                await self.client.unload_adapter(history[-3].served_name)
        return updated

    async def _rationalize(
        self,
        name: str,
        pairs: dict[str, list[PreferencePair]],
        by_task: dict[str, RoundResult],
        cycle: int,
    ) -> None:
        """Replace another's reply as the chosen answer with the specialist's own
        reasoning to the same answer, when that reasoning reaches it."""
        agent = self.agents[name]
        todo = {
            id(pair): pair
            for role, role_pairs in pairs.items()
            if role != COORDINATOR
            for pair in role_pairs
            if pair.chosen_from == "known"
        }

        async def one(pair: PreferencePair) -> None:
            r = by_task[pair.task_id]
            known = r.known[name]
            call = await agent.rationalize(pair.role, r.task, known.answer, cycle)
            answer = (
                await self.grader.code(r.task, call.response)
                if r.task.kind == "code"
                else extract_answer(r.task, call.response)
            )
            reached = await self._reaches(r, name, known.answer, call.response, answer)
            if reached:
                pair.chosen = call.response
                pair.chosen_from = "rationalized"
            self.events.emit(
                "rationalize",
                agent=name,
                cycle=cycle,
                role=pair.role,
                task_id=pair.task_id,
                known_answer=known.answer,
                answer=answer,
                reached=reached,
                response=call.response,
            )

        await asyncio.gather(*(one(pair) for pair in todo.values()))

    async def _reaches(
        self, r: RoundResult, name: str, known: str, reply: str, answer: str
    ) -> bool:
        """Does the rationalisation reach the accepted answer? Maths and choice
        are graded against the answer the world accepted (the task's reference,
        which the accepted reply was judged to match); code must pass the shown
        examples and behave like the accepted code on the team's test inputs."""
        if r.task.kind != "code":
            return await self.grader.correct(r.task, reply, r.task.answer)
        if r.task.examples:
            ran = await self.sandbox.run_tests(
                answer, r.task.model_copy(update={"tests": r.task.examples})
            )
            if not ran.passed:
                return False
        probes = r.results[name].decisions[0].probes
        if not probes:
            return " ".join(answer.split()) == " ".join(known.split()) or bool(
                r.task.examples
            )
        reference = await self.sandbox.run_probes(known, probes, r.task.setup_code)
        output = await self.sandbox.run_probes(answer, probes, r.task.setup_code)
        return behaves_alike(reference, output) is True

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
        await self._servers_ready()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._build_agents(with_memory=False)
        first = self.cfg.agents.names[0]
        report = ExperimentReport(config=self.cfg.model_dump(mode="json"))
        self._phase("baseline", 0)
        report.baseline_benchmarks = await self._benchmarks("baseline", [first])
        path = self.run_dir / "report.json"
        path.write_text(report.model_dump_json(indent=2))
        calllog.close()
        await self.client.aclose()
        return path

    async def _servers_ready(self) -> None:
        await self.client.wait_ready()
        if self.grader.verifier is not None:
            await self.grader.verifier.wait_ready()

    async def run(self) -> ExperimentReport:
        await self._servers_ready()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        report = ExperimentReport(config=self.cfg.model_dump(mode="json"))
        curriculum = self._curriculum()
        self._build_agents()
        self.console.print(
            f"[bold]{self.cfg.experiment.name}[/bold]: agents {list(self.agents)} · "
            f"domains {self.cfg.data.domains} · {len(curriculum.validation)} validation tasks"
        )
        self.events.emit(
            "run_start",
            name=self.cfg.experiment.name,
            config=self.cfg.model_dump(mode="json"),
            agents=list(self.agents),
            personas={
                n: r.persona.model_dump()
                for n, r in next(iter(self.agents.values())).roles.items()
            },
            cycles=self.cfg.experiment.cycles,
            tasks_per_cycle=self.cfg.experiment.rounds_per_cycle,
            validation_tasks=len(curriculum.validation),
        )

        self.console.print("[bold yellow]Baseline validation[/bold yellow]")
        self._phase("validate", 0)
        start = await self._validate(curriculum.validation)
        report.baseline_validation = start.whole
        report.baseline_validation_specialists = start.specialists
        report.baseline_validation_specialists_no_persona = start.specialists_no_persona
        best_mean = {n: sum(v.values()) / len(v) for n, v in start.whole.items()}
        self._save(report)

        for cycle in range(1, self.cfg.experiment.cycles + 1):
            self.console.print(f"[bold cyan]Cycle {cycle}[/bold cyan]")
            metrics = CycleMetrics(cycle=cycle)
            self._phase("wake", cycle)
            rounds = await self._compete(cycle, curriculum.cycle(cycle))
            self._summarise(rounds, metrics)

            self.console.print("  Sleep")
            self._phase("reflect", cycle)
            await self._reflect(cycle, rounds, metrics)
            self._phase("train", cycle)
            updated = await self._train(cycle, rounds, metrics)
            self._snapshot_memory(cycle)

            self._phase("validate", cycle)
            validation = await self._validate(
                curriculum.validation,
                specialists=self.cfg.evaluation.specialists_every_cycle
                or cycle == self.cfg.experiment.cycles,
            )
            metrics.validation = validation.whole
            metrics.validation_specialists = validation.specialists
            metrics.validation_specialists_no_persona = (
                validation.specialists_no_persona
            )
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
                    self.events.emit("rollback", cycle=cycle, agent=name, mean=mean)
                else:
                    best_mean[name] = max(best_mean[name], mean)
            report.cycles.append(metrics)
            self._save(report)
            self.console.print(f"  Validation: {validation.whole}")

        self.console.print("[bold green]Final benchmarks[/bold green]")
        self._phase("benchmarks", self.cfg.experiment.cycles)
        report.final_benchmarks = await self._benchmarks("final", list(self.agents))
        self._save(report)
        self._phase("done", self.cfg.experiment.cycles)
        self.events.close()
        calllog.close()
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

    def _snapshot_memory(self, cycle: int) -> None:
        """Each agent's whole memory as it stood after this cycle's sleep."""
        source = self.run_dir / "memory"
        if source.exists():
            shutil.copytree(
                source,
                self.run_dir / "memory_snapshots" / f"cycle_{cycle}",
                dirs_exist_ok=True,
            )

    def _save(self, report: ExperimentReport) -> None:
        (self.run_dir / "report.json").write_text(report.model_dump_json(indent=2))


__all__ = ["Experiment", "COORDINATOR"]
