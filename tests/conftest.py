"""Shared test fixtures: a scripted chat model, tasks, and an agent factory."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from src.config import FeedbackLevel
from src.environment import Environment
from src.execution.sandbox import Sandbox
from src.inference.agent import Agent, Role
from src.inference.prompts import load_coordinator, load_specialist
from src.models.core import Message, Task

SPECIALISTS = ["logical", "creative", "skeptical", "empathetic"]

# reply(role, stage, user_prompt) -> text
Reply = Callable[[str, str, str], str]


def role_of(system: str) -> str:
    if system.startswith("You lead a team"):
        return "coordinator"
    if system.startswith("You write example calls"):
        return "probes"
    if system.startswith("You maintain"):
        return "reflect"
    for name in SPECIALISTS:
        if name.capitalize() in system:
            return name
    return "unknown"


def stage_of(role: str, user: str) -> str:
    if role in ("coordinator", "probes", "reflect"):
        return role
    return "round2" if "Your teammates' answers" in user else "round1"


class ScriptedChat:
    def __init__(self, reply: Reply) -> None:
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
        role = role_of(messages[0].content)
        stage = stage_of(role, messages[-1].content)
        self.calls.append(
            {"messages": messages, "role": role, "stage": stage, "seed": seed}
        )
        return self.reply(role, stage, messages[-1].content)


def make_agent(
    chat: ScriptedChat,
    name: str = "a",
    max_attempts: int = 3,
    memory=None,
    discussion: bool = True,
) -> Agent:
    sandbox = Sandbox(FeedbackLevel.BINARY, timeout_seconds=5)
    return Agent(
        name=name,
        seed=1,
        specialists={n: Role(load_specialist(n), "base") for n in SPECIALISTS},
        coordinator=Role(load_coordinator(), "base"),
        client=chat,
        environment=Environment(sandbox, FeedbackLevel.BINARY),
        memory=memory,
        max_attempts=max_attempts,
        max_tokens=256,
        probes_per_problem=4,
        discussion=discussion,
    )


def code_reply(code: str) -> str:
    return f"```python\n{code}\n```"


@pytest.fixture
def add_task() -> Task:
    return Task(
        task_id="mbpp/1",
        domain="code",
        kind="code",
        prompt="Write a function that adds two numbers.",
        entry_point="add",
        tests=["assert add(1, 2) == 3", "assert add(-1, 1) == 0"],
    )


@pytest.fixture
def math_task() -> Task:
    return Task(
        task_id="gsm8k/test/0",
        domain="math",
        kind="math",
        prompt="What is 6 times 7?",
        answer="42",
    )


@pytest.fixture
def choice_task() -> Task:
    return Task(
        task_id="siqa/validation/0",
        domain="social",
        kind="choice",
        prompt="Jo lost a friend. How does Jo feel?",
        choices=["happy", "sad", "bored"],
        answer="B",
        max_attempts=1,
    )
