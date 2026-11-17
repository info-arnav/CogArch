"""An agent: four specialists, a coordinator, and a memory.

Each attempt follows the brain's flow from proposal to selected action:

  1. round 1    every specialist answers on its own
  2. evidence   what can be checked without a model: answers grouped by
                agreement; for code, the problem's shown examples run and
                behaviour compared on the team's own test inputs (CodeT)
  3. round 2    only if configured: each specialist sees the others' answers
                and the evidence, and revises
  4. choice     the coordinator chooses one position, every time, seeing the
                answers and the evidence (as the basal ganglia gate every
                action). Which specialist to trust is learned in its weights
                from its rewarded choices, never shown to it. It never writes
                an answer of its own, so whose answer was submitted is always
                known.

Only the submitted reply is judged by the environment, which says only
whether it was wrong.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src import calllog
from src.environment import Environment, answers_match, extract_answer
from src.events import NULL_LOG, EventLog
from src.inference.client import ChatModel
from src.inference.prompts import (
    REJECT_ALL,
    Persona,
    choice_options,
    coordinator_messages,
    discussion_messages,
    parse_choice,
    parse_probes,
    parse_reflection,
    probe_messages,
    rationalize_messages,
    reflection_messages,
    specialist_labels,
    specialist_messages,
)
from src.memory.controller import AgentMemory
from src.models.core import (
    AgentResult,
    Decision,
    Evidence,
    LLMCall,
    Message,
    Proposal,
    Task,
    Verdict,
)

COORDINATOR = "coordinator"


def _group(
    proposals: list[Proposal], same: Callable[[Proposal, Proposal], bool]
) -> list[list[str]]:
    groups: list[list[Proposal]] = []
    for p in proposals:
        for g in groups:
            if same(g[0], p):
                g.append(p)
                break
        else:
            groups.append([p])
    return [[p.specialist for p in g] for g in groups]


def _relabel(evidence: Evidence, labels: dict[str, str]) -> Evidence:
    """The evidence with every specialist named by its letter."""
    return Evidence(
        answers={labels[n]: a for n, a in evidence.answers.items()},
        examples={labels[n]: e for n, e in evidence.examples.items()},
        groups=[[labels[n] for n in g] for g in evidence.groups],
        differences=[
            {
                "call": d["call"],
                "outputs": [[[labels[n] for n in g], o] for g, o in d["outputs"]],
            }
            for d in evidence.differences
        ],
    )


def _fallback_choice(shown: list[Proposal], evidence: Evidence) -> str:
    """When the coordinator names no valid position: the largest group, then
    the most shown examples passed, then the first position shown."""
    size = {n: len(g) for g in evidence.groups for n in g}
    return max(
        shown,
        key=lambda p: (
            size.get(p.specialist, 0),
            evidence.examples.get(p.specialist, [0, 0])[0],
            -shown.index(p),
        ),
    ).specialist


SHOWN_DIFFERENCES = 3
SHOWN_OUTPUT_CHARS = 80


def _differences(
    probes: list[str],
    groups: list[list[str]],
    behaviour: dict[str, list[str] | None],
) -> list[dict]:
    """Test inputs on which the groups' code gives different outputs."""
    found = []
    for i, call in enumerate(probes):
        outputs = []
        for g in groups:
            run = behaviour[g[0]]
            out = "does not run" if run is None else run[i]
            outputs.append([g, out[:SHOWN_OUTPUT_CHARS]])
        if len({o for _, o in outputs}) > 1:
            found.append({"call": call, "outputs": outputs})
        if len(found) == SHOWN_DIFFERENCES:
            break
    return found


async def _none() -> None:
    return None


async def _empty() -> str:
    return ""


def derive_seed(*parts: object) -> int:
    """Stable 31-bit seed (Python's hash() is salted per process)."""
    digest = hashlib.sha256("/".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


@dataclass
class Role:
    """A persona and the model currently serving it (base model or a LoRA adapter)."""

    persona: Persona
    model: str

    @property
    def name(self) -> str:
        return self.persona.name


@dataclass
class Choice:
    chosen: str
    chosen_by: str
    order: list[str]
    call: LLMCall | None
    prompt: list[Message]


class Agent:
    def __init__(
        self,
        name: str,
        seed: int,
        specialists: dict[str, Role],
        coordinator: Role,
        client: ChatModel,
        environment: Environment,
        memory: AgentMemory | None,
        max_attempts: int,
        max_tokens: int,
        probes_per_problem: int,
        discussion: str = "disagreement",
        discussion_style: str = "critique",
        selector: str = "coordinator",
        allow_retry: bool = True,
        memory_at_eval: bool = False,
        events: EventLog = NULL_LOG,
    ) -> None:
        self.name = name
        self.seed = seed
        self.specialists = specialists
        self.coordinator = coordinator
        self.helper = Role(
            coordinator.persona.model_copy(update={"name": "helper"}),
            coordinator.model,
        )
        self.client = client
        self.environment = environment
        self.memory = memory
        self.max_attempts = max_attempts
        self.max_tokens = max_tokens
        self.probes_per_problem = probes_per_problem
        self.discussion = discussion
        self.discussion_style = discussion_style
        self.selector = selector
        self.allow_retry = allow_retry
        self.memory_at_eval = memory_at_eval
        self.events = events

    @property
    def labels(self) -> dict[str, str]:
        return specialist_labels(list(self.specialists))

    @property
    def roles(self) -> dict[str, Role]:
        return {**self.specialists, COORDINATOR: self.coordinator}

    async def _call(
        self,
        role: Role,
        stage: str,
        messages: list[Message],
        *seed_parts: object,
        trace: dict[str, Any] | None = None,
        choices: list[str] | None = None,
    ) -> LLMCall:
        seed = derive_seed(self.seed, role.name, stage, *seed_parts)
        call_id = None
        on_delta = None
        if trace is not None:
            call_id = self.events.new_id()
            self.events.emit(
                "call_start",
                call=call_id,
                agent=self.name,
                role=role.name,
                stage=stage,
                model=role.model,
                temperature=role.persona.temperature,
                messages=[m.model_dump() for m in messages],
                **trace,
            )
            if self.events.stream_tokens:

                def on_delta(text: str) -> None:
                    self.events.emit("delta", call=call_id, text=text)

        start = time.perf_counter()
        with calllog.tagged(agent=self.name, role=role.name, step=stage):
            response = await self.client.chat(
                messages,
                model=role.model,
                temperature=role.persona.temperature,
                max_tokens=self.max_tokens,
                seed=seed,
                on_delta=on_delta,
                choices=choices,
            )
        if call_id is not None:
            self.events.emit(
                "call_end",
                call=call_id,
                response=response,
                seconds=round(time.perf_counter() - start, 2),
            )
        return LLMCall(
            role=role.name,
            stage=stage,
            model=role.model,
            messages=messages,
            response=response,
            temperature=role.persona.temperature,
            seed=seed,
        )

    async def _proposals(
        self, task: Task, calls: list[LLMCall], round_: int
    ) -> list[Proposal]:
        grader = self.environment.grader

        async def one(call: LLMCall) -> Proposal:
            answer = (
                await grader.code(task, call.response)
                if task.kind == "code"
                else extract_answer(task, call.response)
            )
            return Proposal(
                specialist=call.role,
                round=round_,
                response=call.response,
                answer=answer,
            )

        return list(await asyncio.gather(*(one(c) for c in calls)))

    async def _evidence(
        self, task: Task, proposals: list[Proposal], probes: list[str]
    ) -> Evidence:
        """Answers grouped by agreement; for code, shown examples run and
        behaviour compared on the team's own test inputs."""
        answers = {p.specialist: p.answer for p in proposals}
        if task.kind != "code":
            return Evidence(
                answers=answers,
                groups=_group(
                    proposals, lambda a, b: answers_match(task, a.answer, b.answer)
                ),
            )
        sandbox = self.environment.sandbox
        examples: dict[str, list[int]] = {}
        if task.examples:
            shown = task.model_copy(update={"tests": task.examples})
            verdicts = await asyncio.gather(
                *(sandbox.run_tests(p.answer, shown) for p in proposals)
            )
            examples = {
                p.specialist: [v.tests_passed, len(task.examples)]
                for p, v in zip(proposals, verdicts)
            }
        if probes:
            outputs = await asyncio.gather(
                *(
                    (
                        sandbox.run_probes(p.answer, probes, task.setup_code)
                        if p.answer
                        else _none()
                    )
                    for p in proposals
                )
            )
            behaviour = {p.specialist: o for p, o in zip(proposals, outputs)}
            groups = _group(
                proposals,
                lambda a, b: behaviour[a.specialist] is not None
                and behaviour[a.specialist] == behaviour[b.specialist],
            )
            differences = _differences(probes, groups, behaviour)
        else:
            groups = _group(
                proposals,
                lambda a, b: bool(a.answer)
                and " ".join(a.answer.split()) == " ".join(b.answer.split()),
            )
            differences = []
        return Evidence(
            answers=answers, examples=examples, groups=groups, differences=differences
        )

    async def _deliberate(
        self,
        task: Task,
        attempt: int,
        memory_text: str,
        previous: Decision | None,
        purpose: str,
        probes: list[str],
        trace: dict[str, Any] | None = None,
    ) -> Decision:
        """Answer alone; discuss only if the team disagrees; the coordinator
        chooses one position unless everyone agrees. Returns an unjudged decision."""
        roles = list(self.specialists.values())
        key = (purpose, task.task_id, attempt)
        step = None if trace is None else {**trace, "attempt": attempt}

        first = await asyncio.gather(
            *(
                self._call(
                    r,
                    "round1",
                    specialist_messages(r.persona, task, memory_text, previous),
                    *key,
                    trace=step,
                )
                for r in roles
            )
        )
        round1 = await self._proposals(task, list(first), 1)
        evidence = await self._evidence(task, round1, probes)
        self._emit_round(step, 1, round1, evidence)

        discuss = self.discussion == "always" or (
            self.discussion == "disagreement" and not evidence.unanimous
        )
        second: list[LLMCall] = []
        if discuss:
            second = list(
                await asyncio.gather(
                    *(
                        self._call(
                            r,
                            "round2",
                            discussion_messages(
                                r.persona,
                                task,
                                own,
                                [p for p in round1 if p.specialist != r.name],
                                evidence,
                                memory_text,
                                previous,
                                self.discussion_style,
                            ),
                            *key,
                            trace=step,
                        )
                        for r, own in zip(roles, round1)
                    )
                )
            )
            round2 = await self._proposals(task, second, 2)
            evidence = await self._evidence(task, round2, probes)
            self._emit_round(step, 2, round2, evidence)
        else:
            round2 = [p.model_copy(update={"round": 2}) for p in round1]

        may_reject = self.allow_retry and self.selector == "coordinator"
        choice = await self._choose(
            task, round2, evidence, previous, may_reject, key, step
        )
        first_choice: Choice | None = None
        retry: list[LLMCall] = []
        if choice.chosen == REJECT_ALL:
            first_choice = choice
            retry = list(
                await asyncio.gather(
                    *(
                        self._call(
                            r,
                            "round2",
                            specialist_messages(
                                r.persona, task, memory_text, previous, retry=True
                            ),
                            *key,
                            "retry",
                            trace=step,
                        )
                        for r in roles
                    )
                )
            )
            round2 = await self._proposals(task, retry, 2)
            evidence = await self._evidence(task, round2, probes)
            self._emit_round(step, 2, round2, evidence)
            choice = await self._choose(
                task,
                round2,
                evidence,
                previous,
                False,
                (*key, "again"),
                step,
            )
        position = next(p for p in round2 if p.specialist == choice.chosen)
        if step is not None:
            self.events.emit(
                "decision",
                agent=self.name,
                answer=position.answer,
                chosen=choice.chosen,
                chosen_by=choice.chosen_by,
                rejected_all=first_choice is not None,
                discussed=discuss,
                order=choice.order,
                evidence=evidence.model_dump(),
                **step,
            )
        decide_calls = [c.call for c in (first_choice, choice) if c and c.call]
        return Decision(
            attempt_number=attempt,
            labels=self.labels,
            calls=[*first, *second, *retry, *decide_calls],
            round1=round1,
            round2=round2,
            specialist_prompts={
                r.name: specialist_messages(r.persona, task, "", previous)
                for r in roles
            },
            coordinator_prompt=choice.prompt,
            coordinator_output=choice.call.response if choice.call else "",
            rejected_all=first_choice is not None,
            could_reject=may_reject and first_choice is None,
            first_choice_prompt=first_choice.prompt if first_choice else [],
            first_choice_output=(
                first_choice.call.response if first_choice and first_choice.call else ""
            ),
            discussed=discuss,
            probes=probes,
            evidence=evidence,
            chosen=choice.chosen,
            chosen_by=choice.chosen_by,
            submitted=position.response,
            answer=position.answer,
            result=Verdict(passed=False),
        )

    async def _choose(
        self,
        task: Task,
        positions: list[Proposal],
        evidence: Evidence,
        previous: Decision | None,
        may_reject: bool,
        key: tuple,
        step: dict[str, Any] | None,
    ) -> Choice:
        """One decision over the positions: by the coordinator (which may send
        the team back when `may_reject`), or by agreement and majority when the
        selector is "majority"."""
        shown = list(positions)
        random.Random(derive_seed(self.seed, "order", *key)).shuffle(shown)
        names = [p.specialist for p in shown]
        labels = self.labels
        by_label = {v: k for k, v in labels.items()}
        lettered = [
            p.model_copy(update={"specialist": labels[p.specialist]}) for p in shown
        ]
        lettered_evidence = _relabel(evidence, labels)
        options = choice_options(sorted(p.specialist for p in lettered), may_reject)
        if self.selector == "majority":
            largest = max(len(g) for g in evidence.groups)
            leaders = [g for g in evidence.groups if len(g) == largest]
            if evidence.unanimous:
                return Choice(positions[0].specialist, "agreement", names, None, [])
            if len(leaders) == 1 and all(evidence.answers[n] for n in leaders[0]):
                return Choice(leaders[0][0], "majority", names, None, [])
        prompt = coordinator_messages(
            self.coordinator.persona,
            task,
            lettered,
            lettered_evidence,
            previous,
            may_reject,
        )
        call = await self._call(
            self.coordinator,
            "decide",
            prompt,
            *key,
            trace=step,
            choices=options,
        )
        letter = parse_choice(call.response, options)
        if letter is None:
            return Choice(
                _fallback_choice(shown, evidence),
                "fallback",
                names,
                call,
                prompt,
            )
        return Choice(by_label.get(letter, letter), "coordinator", names, call, prompt)

    def _emit_round(
        self,
        step: dict[str, Any] | None,
        number: int,
        proposals: list[Proposal],
        evidence: Evidence,
    ) -> None:
        if step is not None:
            self.events.emit(
                "round",
                agent=self.name,
                round=number,
                answers={p.specialist: p.answer for p in proposals},
                evidence=evidence.model_dump(),
                **step,
            )

    async def _recall(
        self, task: Task, touch: bool, trace: dict[str, Any] | None
    ) -> str:
        if self.memory is None:
            return ""
        context = await self.memory.recall(task.prompt, task.domain, touch=touch)
        text = context.to_prompt_text()
        calllog.record(
            "recall",
            agent=self.name,
            task_id=task.task_id,
            min_similarity=context.min_similarity,
            candidates=[c.model_dump() for c in context.candidates],
            text=text,
        )
        if trace is not None:
            self.events.emit(
                "recall",
                agent=self.name,
                query=task.prompt,
                domain=task.domain,
                min_similarity=context.min_similarity,
                candidates=[c.model_dump() for c in context.candidates],
                text=text,
                **trace,
            )
        return text

    async def _probes(self, task: Task) -> list[str]:
        if task.kind != "code":
            return []
        probes, _ = await self.write_probes(task)
        return probes

    async def solve(self, task: Task, cycle: int = 0) -> AgentResult:
        """Work on a task until the world accepts an answer or attempts run out."""
        trace = {"cycle": cycle, "task_id": task.task_id}
        memory_text, probes = await asyncio.gather(
            self._recall(task, touch=True, trace=trace), self._probes(task)
        )
        attempts = task.max_attempts or self.max_attempts

        decisions: list[Decision] = []
        previous: Decision | None = None
        for attempt in range(1, attempts + 1):
            decision = await self._deliberate(
                task, attempt, memory_text, previous, "solve", probes, trace
            )
            decision.result = await self.environment.judge(task, decision.submitted)
            self.events.emit(
                "verdict",
                agent=self.name,
                attempt=attempt,
                answer=decision.answer,
                passed=decision.result.passed,
                feedback=decision.result.feedback,
                tests_passed=decision.result.tests_passed,
                tests_total=decision.result.tests_total,
                **trace,
            )
            decisions.append(decision)
            if decision.result.passed:
                break
            previous = decision
        return AgentResult(agent=self.name, task_id=task.task_id, decisions=decisions)

    async def answer(self, task: Task, sample: int = 0) -> str:
        """The whole brain's submitted reply for one Pass@1 sample."""
        return (await self.decide(task, sample)).submitted

    async def decide(self, task: Task, sample: int = 0) -> Decision:
        """The whole brain, one attempt, no feedback: the full decision. Memory
        is read, never written, when `memory_at_eval` is on."""
        memory_text, probes = await asyncio.gather(
            (
                self._recall(task, touch=False, trace=None)
                if self.memory_at_eval
                else _empty()
            ),
            self._probes(task),
        )
        return await self._deliberate(
            task, 1, memory_text, None, f"eval{sample}", probes
        )

    async def solo_answer(
        self, specialist: str, task: Task, sample: int = 0, persona: bool = True
    ) -> str:
        """One specialist alone (no discussion, no coordinator): its reply. With
        `persona` off, the same weights and temperature with no system prompt,
        which measures what the persona costs or adds."""
        role = self.specialists[specialist]
        messages = specialist_messages(role.persona, task)
        if not persona:
            messages = [m for m in messages if m.role != "system"]
        call = await self._call(
            role,
            "eval" if persona else "eval_no_persona",
            messages,
            "solo",
            task.task_id,
            sample,
        )
        return call.response

    async def write_probes(self, task: Task) -> tuple[list[str], LLMCall]:
        """Example calls the agent invents to compare code solutions' behaviour."""
        call = await self._call(
            self.helper,
            "probes",
            probe_messages(task, self.probes_per_problem),
            task.task_id,
        )
        return (
            parse_probes(call.response, task.entry_point, self.probes_per_problem),
            call,
        )

    async def rationalize(
        self, specialist: str, task: Task, known_answer: str, cycle: int
    ) -> LLMCall:
        """Sleep: the specialist reasons its own way to an answer the world accepted."""
        role = self.specialists[specialist]
        return await self._call(
            role,
            "rationalize",
            rationalize_messages(role.persona, task, known_answer),
            task.task_id,
            cycle,
        )

    async def reflect(
        self, cases: list[str], cycle: int
    ) -> tuple[list[tuple[int, str]], LLMCall]:
        """Sleep: compare wrong answers with correct ones and say, for each case,
        what went wrong."""
        call = await self._call(
            self.helper, "reflect", reflection_messages(cases), cycle
        )
        return parse_reflection(call.response), call
