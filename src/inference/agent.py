"""An agent: four specialists, a coordinator, and a memory.

Per attempt, every specialist proposes a solution, the coordinator reads the
proposals and decides the team's answer, and only that answer is judged by
the environment. The verdict ("wrong" or accepted) is what the whole agent
learns from.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass

from src.execution.sandbox import Sandbox
from src.inference.client import ChatModel
from src.inference.prompts import (
    Persona,
    coordinator_messages,
    extract_code,
    parse_coordinator,
    parse_lessons,
    parse_probes,
    probe_messages,
    reflection_messages,
    specialist_messages,
)
from src.memory.controller import AgentMemory
from src.models.code import AgentResult, CodeProblem, Decision, Message, Proposal


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

    async def _ask(
        self, role: Role, messages: list[Message], *seed_parts: object
    ) -> str:
        return await self.client.chat(
            messages,
            model=role.model,
            temperature=role.persona.temperature,
            max_tokens=self.max_tokens,
            seed=derive_seed(self.seed, role.name, *seed_parts),
        )

    async def _attempt(
        self,
        problem: CodeProblem,
        attempt_number: int,
        memory_text: str,
        previous: Decision | None,
        purpose: str,
    ) -> tuple[list[Proposal], list[Message], str, str, str | None]:
        roles = list(self.specialists.values())
        responses = await asyncio.gather(
            *(
                self._ask(
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
            Proposal(specialist=role.name, code=extract_code(text))
            for role, text in zip(roles, responses)
        ]
        output = await self._ask(
            self.coordinator,
            coordinator_messages(
                self.coordinator.persona, problem, proposals, memory_text, previous
            ),
            purpose,
            problem.task_id,
            attempt_number,
        )
        code, choice = parse_coordinator(output, proposals)
        training_prompt = coordinator_messages(
            self.coordinator.persona, problem, proposals, "", previous
        )
        return proposals, training_prompt, output, code, choice

    async def solve(self, problem: CodeProblem) -> AgentResult:
        """Work on a problem until the world accepts an answer or attempts run out."""
        memory_text = ""
        if self.memory is not None:
            memory_text = (await self.memory.recall(problem.prompt)).to_prompt_text()

        decisions: list[Decision] = []
        previous: Decision | None = None
        for attempt_number in range(1, self.max_attempts + 1):
            proposals, prompt, output, code, choice = await self._attempt(
                problem, attempt_number, memory_text, previous, "solve"
            )
            decision = Decision(
                attempt_number=attempt_number,
                proposals=proposals,
                coordinator_prompt=prompt,
                coordinator_output=output,
                code=code,
                choice=choice,
                result=await self.sandbox.run_tests(code, problem),
            )
            decisions.append(decision)
            if decision.result.passed:
                break
            previous = decision

        return AgentResult(
            agent=self.name, problem_id=problem.task_id, decisions=decisions
        )

    async def answer(self, problem: CodeProblem) -> str:
        """One attempt with no memory and no feedback: the agent's Pass@1 answer."""
        *_, code, _ = await self._attempt(problem, 1, "", None, "eval")
        return code

    async def write_probes(self, problem: CodeProblem) -> list[str]:
        """Example calls the agent invents to compare solutions' behaviour."""
        text = await self._ask(
            self.coordinator,
            probe_messages(problem, self.probes_per_problem),
            "probes",
            problem.task_id,
        )
        return parse_probes(text, problem.entry_point, self.probes_per_problem)

    async def reflect(self, experiences: str, count: int, cycle: int) -> list[str]:
        """Sleep: distil recent experiences into general lessons."""
        text = await self._ask(
            self.coordinator, reflection_messages(experiences, count), "reflect", cycle
        )
        return parse_lessons(text, count)
