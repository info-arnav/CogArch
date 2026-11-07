"""Shared test fixtures: a scripted chat model and agent factory."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from src.config import FeedbackLevel
from src.execution.sandbox import Sandbox
from src.inference.agent import Agent, Role
from src.inference.prompts import load_coordinator, load_specialist
from src.models.code import CodeProblem, Message

SPECIALISTS = ["logical", "creative", "skeptical", "empathetic"]


def code_reply(code: str) -> str:
    return f"```python\n{code}\n```"


class ScriptedChat:
    """Returns replies from a function of (role system prompt, user prompt, model)."""

    def __init__(self, reply: Callable[[str, str, str], str]) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    async def chat(
        self,
        messages: list[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> str:
        self.calls.append({"messages": messages, "model": model, "seed": seed})
        return self.reply(messages[0].content, messages[-1].content, model)


def make_agent(
    chat: ScriptedChat,
    name: str = "a",
    max_attempts: int = 3,
    memory=None,
) -> Agent:
    return Agent(
        name=name,
        seed=1,
        specialists={n: Role(load_specialist(n), "base") for n in SPECIALISTS},
        coordinator=Role(load_coordinator(), "base"),
        client=chat,
        sandbox=Sandbox(FeedbackLevel.BINARY, timeout_seconds=5),
        memory=memory,
        max_attempts=max_attempts,
        max_tokens=256,
        probes_per_problem=4,
    )


def is_coordinator(system: str) -> bool:
    return system.startswith("You lead a team")


@pytest.fixture
def add_problem() -> CodeProblem:
    return CodeProblem(
        task_id="mbpp/1",
        prompt="Write a function that adds two numbers.",
        entry_point="add",
        tests=["assert add(1, 2) == 3", "assert add(-1, 1) == 0"],
    )
