"""Personas and every prompt an agent's parts receive, plus output parsing."""

from __future__ import annotations

import re
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel

from src.models.code import CodeProblem, Decision, Message, Proposal

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


class Persona(BaseModel):
    name: str
    description: str
    temperature: float
    system_prompt: str


def load_specialist(name: str, prompts_dir: Path = PROMPTS_DIR) -> Persona:
    with open(prompts_dir / "specialists" / f"{name}.yaml") as f:
        return Persona.model_validate(yaml.safe_load(f))


def load_coordinator(prompts_dir: Path = PROMPTS_DIR) -> Persona:
    with open(prompts_dir / "coordinator.yaml") as f:
        return Persona.model_validate(yaml.safe_load(f))


# ── shared pieces ────────────────────────────────────────────────────────────


def task_text(problem: CodeProblem) -> str:
    return f"{problem.prompt}\n\nImplement the function `{problem.entry_point}`."


def _code_block(code: str) -> str:
    return f"```python\n{code}\n```"


def _with_memory(text: str, memory_text: str) -> str:
    return f"{memory_text}\n\n---\n\n{text}" if memory_text else text


def _previous_attempt(previous: Decision) -> str:
    return (
        f"Your team's previous answer:\n{_code_block(previous.code)}\n\n"
        f"{previous.result.feedback}"
    )


def _messages(system: str, user: str) -> list[Message]:
    return [Message(role="system", content=system), Message(role="user", content=user)]


# ── specialists ──────────────────────────────────────────────────────────────


def specialist_messages(
    persona: Persona,
    problem: CodeProblem,
    memory_text: str = "",
    previous: Decision | None = None,
) -> list[Message]:
    user = task_text(problem)
    if previous is not None:
        user += f"\n\n{_previous_attempt(previous)}\n\nPropose a corrected solution."
    return _messages(persona.system_prompt, _with_memory(user, memory_text))


# ── coordinator ──────────────────────────────────────────────────────────────


def coordinator_messages(
    persona: Persona,
    problem: CodeProblem,
    proposals: list[Proposal],
    memory_text: str = "",
    previous: Decision | None = None,
) -> list[Message]:
    sections = [task_text(problem)]
    if previous is not None:
        sections.append(_previous_attempt(previous))
    sections.append(
        "Proposals from your specialists:\n\n"
        + "\n\n".join(f"### {p.specialist}\n{_code_block(p.code)}" for p in proposals)
    )
    sections.append("Decide the team's final answer.")
    return _messages(
        persona.system_prompt, _with_memory("\n\n".join(sections), memory_text)
    )


def coordinator_selection(proposal: Proposal) -> str:
    """A well-formed coordinator reply that adopts one proposal unchanged."""
    return f"BASED ON: {proposal.specialist}\n{_code_block(proposal.code)}"


_BASED_ON = re.compile(r"^\s*BASED ON:\s*(.+)$", re.MULTILINE | re.IGNORECASE)


def parse_coordinator(output: str, specialists: list[str]) -> tuple[str, list[str]]:
    """Return (final code, specialists named in the BASED ON line)."""
    match = _BASED_ON.search(output)
    named: list[str] = []
    if match:
        mentioned = {part.strip().lower() for part in match.group(1).split(",")}
        named = [s for s in specialists if s in mentioned]
    return extract_code(output), named


# ── self-made probes (the agent's own example calls) ────────────────────────


def probe_messages(problem: CodeProblem, count: int) -> list[Message]:
    return _messages(
        "You write example calls to test Python functions.",
        f"{task_text(problem)}\n\n"
        f"Write {count} different example calls of `{problem.entry_point}`, "
        "covering typical inputs and edge cases. Output only the calls, one per line, "
        f"each a single Python expression starting with `{problem.entry_point}(`. "
        "Do not write expected results.",
    )


def parse_probes(text: str, entry_point: str, limit: int) -> list[str]:
    probes = []
    for line in extract_code(text).splitlines():
        line = line.strip().rstrip(",")
        if line.startswith(f"{entry_point}(") and line not in probes:
            probes.append(line)
    return probes[:limit]


# ── sleep: reflection ────────────────────────────────────────────────────────


def reflection_messages(experiences: str, count: int) -> list[Message]:
    return _messages(
        "You review a coding team's recent experiences and distil lasting lessons.",
        f"Recent experiences:\n\n{experiences}\n\n"
        f"Write up to {count} short, general lessons that would help the team avoid "
        "being wrong on future problems. One lesson per line, each starting with '- '. "
        "Lessons must be general advice, not answers to specific problems.",
    )


def parse_lessons(text: str, limit: int) -> list[str]:
    lessons = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(("- ", "* ")):
            lesson = line[2:].strip()
            if lesson and lesson not in lessons:
                lessons.append(lesson)
    return lessons[:limit]


# ── parsing ──────────────────────────────────────────────────────────────────

_FENCED = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


def extract_code(response: str) -> str:
    """The first fenced code block, or the whole response if there is none."""
    match = _FENCED.search(response)
    return (match.group(1) if match else response).strip()
