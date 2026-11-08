"""An agent: four specialists, a coordinator, and a memory.

Per attempt, every specialist proposes a solution, the coordinator reads the
proposals and decides the team's answer, and only that answer is judged by
the environment. The verdict ("wrong" or accepted) is what the whole agent
learns from.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
from dataclasses import dataclass

from src.execution.sandbox import Sandbox
from src.inference.client import ChatModel
from src.inference.prompts import (
    Persona,
    coordinator_messages,
    extract_code,
    parse_coordinator,
    parse_probes,
    parse_reflection,
    probe_messages,
    reflection_messages,
    specialist_messages,
)
from src.memory.controller import AgentMemory
from src.models.code import (
    AgentResult,
    CodeProblem,
    Decision,
    ExecutionResult,
    LLMCall,
    Message,
    Proposal,
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
        sandbox: Sandbox,
        memory: AgentMemory | None,
        max_attempts: int,
        max_tokens: int,
        probes_per_problem: int,
    ) -> None:
        self.name = name
        self.seed = seed
        self.specialists = specialists
        self.coordinator = coordinator
        self.client = client
        self.sandbox = sandbox
        self.memory = memory
        self.max_attempts = max_attempts
        self.max_tokens = max_tokens
        self.probes_per_problem = probes_per_problem

    @property
    def roles(self) -> dict[str, Role]:
        return {**self.specialists, "coordinator": self.coordinator}

    async def _call(
        self, role: Role, messages: list[Message], *seed_parts: object
    ) -> LLMCall:
        seed = derive_seed(self.seed, role.name, *seed_parts)
        response = await self.client.chat(
            messages,
            model=role.model,
            temperature=role.persona.temperature,
            max_tokens=self.max_tokens,
            seed=seed,
        )
        return LLMCall(
            role=role.name,
            model=role.model,
            messages=messages,
            response=response,
            temperature=role.persona.temperature,
            seed=seed,
        )

    async def _attempt(
        self,
        problem: CodeProblem,
        attempt_number: int,
        memory_text: str,
        previous: Decision | None,
        purpose: str,
    ) -> Decision:
        """Specialists propose, the coordinator decides. Not yet judged."""
        roles = list(self.specialists.values())
        specialist_calls = await asyncio.gather(
            *(
                self._call(
                    role,
                    specialist_messages(role.persona, problem, memory_text, previous),
                    purpose,
                    problem.task_id,
                    attempt_number,
                )
                for role in roles
            )
        )
        proposals = [
            Proposal(specialist=call.role, code=extract_code(call.response))
            for call in specialist_calls
        ]
        shown = list(proposals)
        random.Random(
            derive_seed(self.seed, "order", purpose, problem.task_id, attempt_number)
        ).shuffle(shown)
        coordinator_call = await self._call(
            self.coordinator,
            coordinator_messages(
                self.coordinator.persona, problem, shown, memory_text, previous
            ),
            purpose,
            problem.task_id,
            attempt_number,
        )
        code, choice = parse_coordinator(coordinator_call.response, proposals)
        return Decision(
            attempt_number=attempt_number,
            calls=[*specialist_calls, coordinator_call],
            proposals=proposals,
            coordinator_prompt=coordinator_messages(
                self.coordinator.persona, problem, shown, "", previous
            ),
            coordinator_output=coordinator_call.response,
            code=code,
            choice=choice,
            result=ExecutionResult(passed=False, tests_passed=0, tests_total=0),
        )

    async def solve(self, problem: CodeProblem) -> AgentResult:
        """Work on a problem until the world accepts an answer or attempts run out."""
        memory_text = ""
        if self.memory is not None:
            memory_text = (await self.memory.recall(problem.prompt)).to_prompt_text()

        decisions: list[Decision] = []
        previous: Decision | None = None
        for attempt_number in range(1, self.max_attempts + 1):
            decision = await self._attempt(
                problem, attempt_number, memory_text, previous, "solve"
            )
            decision.result = await self.sandbox.run_tests(decision.code, problem)
            decisions.append(decision)
            if decision.result.passed:
                break
            previous = decision

        return AgentResult(
            agent=self.name, problem_id=problem.task_id, decisions=decisions
        )

    async def answer(self, problem: CodeProblem, sample: int = 0) -> str:
        """One attempt with no memory and no feedback: a Pass@1 sample."""
        decision = await self._attempt(problem, 1, "", None, f"eval{sample}")
        return decision.code

    async def write_probes(self, problem: CodeProblem) -> tuple[list[str], LLMCall]:
        """Example calls the agent invents to compare solutions' behaviour."""
        call = await self._call(
            self.coordinator,
            probe_messages(problem, self.probes_per_problem),
            "probes",
            problem.task_id,
        )
        return (
            parse_probes(call.response, problem.entry_point, self.probes_per_problem),
            call,
        )

    async def reflect(
        self, lessons: list[str], cases: list[str], max_new: int, cycle: int
    ) -> tuple[list[tuple[str, int | None, str]], LLMCall]:
        """Sleep: compare wrong answers with accepted solutions; edit the lessons."""
        call = await self._call(
            self.coordinator,
            reflection_messages(lessons, cases, max_new),
            "reflect",
            cycle,
        )
        return parse_reflection(call.response, len(lessons)), call
