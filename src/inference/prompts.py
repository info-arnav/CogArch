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


# EvalPlus's instruction for chat models, so scores are comparable to published ones.
INSTRUCTION = (
    "Please provide a self-contained Python script that solves the following problem "
    "in a markdown code block:"
)


def _code_block(code: str) -> str:
    return f"```python\n{code}\n```"


def task_text(problem: CodeProblem) -> str:
    body = (
        _code_block(problem.prompt.strip())
        if problem.prompt_is_code
        else problem.prompt.strip()
    )
    return (
        f"{INSTRUCTION}\n{body}\n\n"
        f"The solution must define the function `{problem.entry_point}`."
    )


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
    """Proposals are listed in the order given; callers shuffle them per decision
    so that position bias cannot masquerade as trust in a specialist."""
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
    """A well-formed coordinator reply that submits one proposal unchanged."""
    return f"CHOOSE: {proposal.specialist}"


COMBINED = "combined"
_CHOOSE = re.compile(r"^\W*CHOOSE\W*:?\W*(\w+)", re.MULTILINE | re.IGNORECASE)
_COMBINE = re.compile(r"^\W*COMBINE\b", re.MULTILINE | re.IGNORECASE)


def parse_coordinator(output: str, proposals: list[Proposal]) -> tuple[str, str | None]:
    """Return (submitted code, choice).

    choice is a specialist's name (its proposal is submitted unchanged),
    "combined" (the reply's own code block is submitted), or None when the
    reply follows neither form (its code block, if any, is submitted).
    """
    by_name = {p.specialist: p for p in proposals}
    chosen = _CHOOSE.search(output)
    if chosen and chosen.group(1).lower() in by_name:
        return by_name[chosen.group(1).lower()].code, chosen.group(1).lower()
    if _COMBINE.search(output):
        return extract_code(output), COMBINED
    return extract_code(output), None


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
    """Example calls from anywhere in the reply (models often restate the
    function in one code block and put the calls in another)."""
    probes: list[str] = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if line.startswith(f"{entry_point}(") and line not in probes:
            probes.append(line)
    return probes[:limit]


# ── sleep: reflection (ExpeL-style rule list editing) ───────────────────────


def reflection_messages(
    lessons: list[str], cases: list[str], max_new: int
) -> list[Message]:
    current = (
        "\n".join(f"{i}. {lesson}" for i, lesson in enumerate(lessons, start=1))
        if lessons
        else "(none yet)"
    )
    return _messages(
        "You maintain a coding team's list of lessons learned from its mistakes.",
        f"Current lessons:\n{current}\n\n"
        "Cases from the last cycle where the team was told its answer was wrong and "
        "a solution was later accepted:\n\n" + "\n\n".join(cases) + "\n\n"
        "Compare each wrong answer with the accepted solution and update the lessons. "
        "Reply with operations, one per line:\n"
        "ADD: <new lesson>\n"
        "AGREE <number>   (an existing lesson is confirmed by these cases)\n"
        "DISAGREE <number>   (an existing lesson is contradicted or useless)\n"
        "EDIT <number>: <improved lesson>\n\n"
        f"Use at most {max_new} ADD or EDIT operations. A lesson must be a general rule "
        "about a kind of mistake that would help on other problems. Never mention "
        "function names, specific problems or their solutions.",
    )


def parse_reflection(text: str, existing: int) -> list[tuple[str, int | None, str]]:
    """Operations as (kind, lesson number or None, text)."""
    ops: list[tuple[str, int | None, str]] = []
    for line in text.splitlines():
        line = line.strip().lstrip("-* ").strip()
        if m := re.match(r"(?i)ADD\s*:\s*(.+)", line):
            ops.append(("add", None, m.group(1).strip()))
        elif m := re.match(r"(?i)EDIT\s+(\d+)\s*:\s*(.+)", line):
            if 1 <= int(m.group(1)) <= existing:
                ops.append(("edit", int(m.group(1)), m.group(2).strip()))
        elif m := re.match(r"(?i)(AGREE|DISAGREE)\s+(\d+)", line):
            if 1 <= int(m.group(2)) <= existing:
                ops.append((m.group(1).lower(), int(m.group(2)), ""))
    return ops


# ── parsing ──────────────────────────────────────────────────────────────────

_FENCED = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


def extract_code(response: str) -> str:
    """The first fenced code block, or the whole response if there is none."""
    match = _FENCED.search(response)
    return (match.group(1) if match else response).strip()
