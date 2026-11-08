"""Personas and every prompt an agent's parts receive, plus output parsing."""

from __future__ import annotations

import re
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel

from src.models.core import Decision, Message, Proposal, Task

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
MAX_SHOWN_CHARS = 1800  # per teammate reply in discussion and decision prompts


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


# ── the task, as every part sees it ─────────────────────────────────────────

# EvalPlus's instruction for chat models, so code scores are comparable.
CODE_INSTRUCTION = (
    "Please provide a self-contained Python script that solves the following problem "
    "in a markdown code block:"
)
MATH_INSTRUCTION = (
    "Solve the following problem step by step. "
    "Put your final answer within \\boxed{}."
)
CHOICE_INSTRUCTION = (
    "Answer the following multiple-choice question. Think it through briefly, then end "
    "with a line of the form 'FINAL ANSWER: <letter>'."
)


def _code_block(code: str) -> str:
    return f"```python\n{code}\n```"


def task_text(task: Task) -> str:
    if task.kind == "code":
        body = (
            _code_block(task.prompt.strip())
            if task.prompt_is_code
            else task.prompt.strip()
        )
        return (
            f"{CODE_INSTRUCTION}\n{body}\n\n"
            f"The solution must define the function `{task.entry_point}`."
        )
    if task.kind == "math":
        return f"{MATH_INSTRUCTION}\n\n{task.prompt.strip()}"
    options = "\n".join(f"{'ABCDEFGHIJ'[i]}. {c}" for i, c in enumerate(task.choices))
    return f"{CHOICE_INSTRUCTION}\n\n{task.prompt.strip()}\n\n{options}"


def _shown(text: str) -> str:
    text = text.strip()
    return text if len(text) <= MAX_SHOWN_CHARS else text[:MAX_SHOWN_CHARS] + " …"


def _with_memory(text: str, memory_text: str) -> str:
    return f"{memory_text}\n\n---\n\n{text}" if memory_text else text


def _previous(task: Task, previous: Decision) -> str:
    shown = _code_block(previous.answer) if task.kind == "code" else previous.answer
    return f"Your team's previous answer:\n{shown}\n\n{previous.result.feedback}"


def _messages(system: str, user: str) -> list[Message]:
    return [Message(role="system", content=system), Message(role="user", content=user)]


# ── round 1: each specialist on its own ─────────────────────────────────────


def specialist_messages(
    persona: Persona,
    task: Task,
    memory_text: str = "",
    previous: Decision | None = None,
) -> list[Message]:
    user = task_text(task)
    if previous is not None:
        user += f"\n\n{_previous(task, previous)}\n\nGive a corrected answer."
    return _messages(persona.system_prompt, _with_memory(user, memory_text))


# ── round 2: discussion ─────────────────────────────────────────────────────


def discussion_messages(
    persona: Persona,
    task: Task,
    own: Proposal,
    teammates: list[Proposal],
    memory_text: str = "",
    previous: Decision | None = None,
) -> list[Message]:
    others = "\n\n".join(f"### {p.specialist}\n{_shown(p.response)}" for p in teammates)
    sections = [task_text(task)]
    if previous is not None:
        sections.append(_previous(task, previous))
    sections += [
        f"Your answer:\n{_shown(own.response)}",
        f"Your teammates' answers:\n\n{others}",
        "Discuss: where do you agree or disagree with your teammates, and why? Point out "
        "any mistakes, including your own. Then give your final answer in the required "
        "format.",
    ]
    return _messages(
        persona.system_prompt, _with_memory("\n\n".join(sections), memory_text)
    )


# ── the decision ────────────────────────────────────────────────────────────


def coordinator_messages(
    persona: Persona,
    task: Task,
    positions: list[Proposal],
    memory_text: str = "",
    previous: Decision | None = None,
) -> list[Message]:
    """Positions are listed in the order given; callers shuffle them per decision
    so position bias cannot masquerade as trust in a specialist."""
    sections = [task_text(task)]
    if previous is not None:
        sections.append(_previous(task, previous))
    sections.append(
        "Your specialists' final positions after discussion:\n\n"
        + "\n\n".join(f"### {p.specialist}\n{_shown(p.response)}" for p in positions)
    )
    sections.append("Decide the team's final answer.")
    return _messages(
        persona.system_prompt, _with_memory("\n\n".join(sections), memory_text)
    )


def coordinator_endorsement(proposal: Proposal) -> str:
    """A well-formed decision that adopts one specialist's reasoning and answer."""
    return f"{proposal.specialist}'s reasoning holds up.\n\n{proposal.response.strip()}"


# ── credit assignment for code: the agent's own example calls ───────────────


def probe_messages(task: Task, count: int) -> list[Message]:
    return _messages(
        "You write example calls to test Python functions.",
        f"{task_text(task)}\n\n"
        f"Write {count} different example calls of `{task.entry_point}`, covering typical "
        "inputs and edge cases. Output only the calls, one per line, each a single Python "
        f"expression starting with `{task.entry_point}(`. Do not write expected results.",
    )


def parse_probes(text: str, entry_point: str, limit: int) -> list[str]:
    """Example calls from anywhere in the reply."""
    probes: list[str] = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if line.startswith(f"{entry_point}(") and line not in probes:
            probes.append(line)
    return probes[:limit]


# ── sleep: reflection (ExpeL-style lesson editing) ──────────────────────────


def reflection_messages(
    lessons: list[str], cases: list[str], max_new: int
) -> list[Message]:
    current = (
        "\n".join(f"{i}. {lesson}" for i, lesson in enumerate(lessons, start=1))
        if lessons
        else "(none yet)"
    )
    return _messages(
        "You maintain a team's list of lessons learned from its mistakes.",
        f"Current lessons:\n{current}\n\n"
        "Cases from the last cycle where the team was told its answer was wrong and a "
        "correct answer was later found:\n\n" + "\n\n".join(cases) + "\n\n"
        "Compare each wrong answer with the correct one and update the lessons. Reply "
        "with operations, one per line:\n"
        "ADD: <new lesson>\n"
        "AGREE <number>   (an existing lesson is confirmed by these cases)\n"
        "DISAGREE <number>   (an existing lesson is contradicted or useless)\n"
        "EDIT <number>: <improved lesson>\n\n"
        f"Use at most {max_new} ADD or EDIT operations. A lesson must be a general rule "
        "about a kind of mistake that would help on other problems. Never mention "
        "specific problems, names, function names or answers.",
    )


def parse_reflection(text: str, existing: int) -> list[tuple[str, int | None, str]]:
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
