"""An agent: four specialists, a coordinator, and a memory.

Each attempt has three stages, like specialised brain regions exchanging
information through a shared workspace before the executive decides:

  1. round 1   every specialist answers on its own
  2. round 2   every specialist sees the others' answers, discusses, and revises
  3. decide    the coordinator weighs the final positions and writes the answer

Only the coordinator's answer is judged by the environment, which says only
whether it was wrong.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
from dataclasses import dataclass

from src.environment import Environment, extract_answer
from src.inference.client import ChatModel
from src.inference.prompts import (
    Persona,
    coordinator_messages,
    discussion_messages,
    parse_probes,
    parse_reflection,
    probe_messages,
    reflection_messages,
    specialist_messages,
)
from src.memory.controller import AgentMemory
from src.models.core import (
    AgentResult,
    Decision,
    LLMCall,
    Message,
    Proposal,
    Task,
    Verdict,
)


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
        discussion: bool = True,
    ) -> None:
        self.name = name
        self.seed = seed
        self.specialists = specialists
        self.coordinator = coordinator
        self.client = client
        self.environment = environment
        self.memory = memory
        self.max_attempts = max_attempts
        self.max_tokens = max_tokens
        self.probes_per_problem = probes_per_problem
        self.discussion = discussion

    @property
    def roles(self) -> dict[str, Role]:
        return {**self.specialists, "coordinator": self.coordinator}

    async def _call(
        self, role: Role, stage: str, messages: list[Message], *seed_parts: object
    ) -> LLMCall:
        seed = derive_seed(self.seed, role.name, stage, *seed_parts)
        response = await self.client.chat(
            messages,
            model=role.model,
            temperature=role.persona.temperature,
            max_tokens=self.max_tokens,
            seed=seed,
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

    async def _deliberate(
        self,
        task: Task,
        attempt: int,
        memory_text: str,
        previous: Decision | None,
        purpose: str,
    ) -> Decision:
        """Round 1, discussion, decision. Returns an unjudged decision."""
        roles = list(self.specialists.values())
        key = (purpose, task.task_id, attempt)

        first = await asyncio.gather(
            *(
                self._call(
                    r,
                    "round1",
                    specialist_messages(r.persona, task, memory_text, previous),
                    *key,
                )
                for r in roles
            )
        )
        round1 = [
            Proposal(
                specialist=c.role,
                round=1,
                response=c.response,
                answer=extract_answer(task, c.response),
            )
            for c in first
        ]

        second: list[LLMCall] = []
        if self.discussion:
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
                                memory_text,
                                previous,
                            ),
                            *key,
                        )
                        for r, own in zip(roles, round1)
                    )
                )
            )
            round2 = [
                Proposal(
                    specialist=c.role,
                    round=2,
                    response=c.response,
                    answer=extract_answer(task, c.response),
                )
                for c in second
            ]
        else:
            round2 = [p.model_copy(update={"round": 2}) for p in round1]

        shown = list(round2)
        random.Random(derive_seed(self.seed, "order", *key)).shuffle(shown)
        decide = await self._call(
            self.coordinator,
            "decide",
            coordinator_messages(
                self.coordinator.persona, task, shown, memory_text, previous
            ),
            *key,
        )
        return Decision(
            attempt_number=attempt,
            calls=[*first, *second, decide],
            round1=round1,
            round2=round2,
            specialist_prompts={
                r.name: specialist_messages(r.persona, task, "", previous)
                for r in roles
            },
            coordinator_prompt=coordinator_messages(
                self.coordinator.persona, task, shown, "", previous
            ),
            coordinator_output=decide.response,
            answer=extract_answer(task, decide.response),
            result=Verdict(passed=False),
        )

    async def solve(self, task: Task) -> AgentResult:
        """Work on a task until the world accepts an answer or attempts run out."""
        memory_text = ""
        if self.memory is not None:
            memory_text = (await self.memory.recall(task.prompt)).to_prompt_text()
        attempts = task.max_attempts or self.max_attempts

        decisions: list[Decision] = []
        previous: Decision | None = None
        for attempt in range(1, attempts + 1):
            decision = await self._deliberate(
                task, attempt, memory_text, previous, "solve"
            )
            decision.result = await self.environment.judge(task, decision.answer)
            decisions.append(decision)
            if decision.result.passed:
                break
            previous = decision
        return AgentResult(agent=self.name, task_id=task.task_id, decisions=decisions)

    async def answer(self, task: Task, sample: int = 0) -> str:
        """The whole brain, one attempt, no memory, no feedback (a Pass@1 sample)."""
        decision = await self._deliberate(task, 1, "", None, f"eval{sample}")
        return decision.answer

    async def solo_answer(self, specialist: str, task: Task, sample: int = 0) -> str:
        """One specialist alone with its own persona (no discussion, no coordinator)."""
        role = self.specialists[specialist]
        call = await self._call(
            role,
            "eval",
            specialist_messages(role.persona, task),
            "solo",
            task.task_id,
            sample,
        )
        return extract_answer(task, call.response)

    async def write_probes(self, task: Task) -> tuple[list[str], LLMCall]:
        """Example calls the agent invents to compare code solutions' behaviour."""
        call = await self._call(
            self.coordinator,
            "probes",
            probe_messages(task, self.probes_per_problem),
            task.task_id,
        )
        return (
            parse_probes(call.response, task.entry_point, self.probes_per_problem),
            call,
        )

    async def reflect(
        self, lessons: list[str], cases: list[str], max_new: int, cycle: int
    ) -> tuple[list[tuple[str, int | None, str]], LLMCall]:
        """Sleep: compare wrong answers with correct ones; edit the lessons."""
        call = await self._call(
            self.coordinator,
            "reflect",
            reflection_messages(lessons, cases, max_new),
            cycle,
        )
        return parse_reflection(call.response, len(lessons)), call
