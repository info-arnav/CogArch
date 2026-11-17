"""Personas and every prompt an agent's parts receive, plus output parsing."""

from __future__ import annotations

import re
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel

from src.models.core import Decision, Evidence, Message, Proposal, Task

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
REASONING_SHOWN_CHARS = 900


class Persona(BaseModel):
    name: str
    description: str
    temperature: float
    system_prompt: str


def load_specialist(name: str, prompts_dir: Path = PROMPTS_DIR) -> Persona:
    with open(prompts_dir / "specialists" / f"{name}.yaml") as f:
        return Persona.model_validate(yaml.safe_load(f))


NEUTRAL_PROMPT = (
    "You are one of four members of a team. Show your reasoning, then give your "
    "answer in exactly the format the problem asks for."
)


def neutral_persona(persona: Persona) -> Persona:
    """The same specialist (name, temperature) with the neutral team prompt."""
    return persona.model_copy(update={"system_prompt": NEUTRAL_PROMPT})


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
    "Answer the following multiple-choice question. Think it through, then end "
    "with a line of the form 'FINAL ANSWER: <letter>'."
)
MEMORY_HEADER = (
    "Notes from your memory. They come from earlier problems and may not apply; "
    "use them only where they help."
)


def answer_format(task: Task) -> str:
    if task.kind == "code":
        return (
            "your complete solution in one ```python code block, with no tests, "
            "prints or example usage inside it"
        )
    if task.kind == "math":
        return "your final answer within \\boxed{}"
    return "a line of the form 'FINAL ANSWER: <letter>'"


def _code_block(code: str) -> str:
    return f"```python\n{code}\n```"


def task_text(task: Task) -> str:
    if task.kind == "code":
        body = (
            _code_block(task.prompt.strip())
            if task.prompt_is_code
            else task.prompt.strip()
        )
        examples = (
            "\n\nYour code should pass this example:\n"
            + _code_block("\n".join(task.examples))
            if task.examples and not task.prompt_is_code
            else ""
        )
        return (
            f"{CODE_INSTRUCTION}\n{body}{examples}\n\n"
            f"The solution must define the function `{task.entry_point}`."
        )
    if task.kind == "math":
        return f"{MATH_INSTRUCTION}\n\n{task.prompt.strip()}"
    options = "\n".join(f"{'ABCDEFGHIJ'[i]}. {c}" for i, c in enumerate(task.choices))
    return f"{CHOICE_INSTRUCTION}\n\n{task.prompt.strip()}\n\n{options}"


def _with_memory(text: str, memory_text: str) -> str:
    return f"{text}\n\n---\n{MEMORY_HEADER}\n\n{memory_text}" if memory_text else text


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
    retry: bool = False,
) -> list[Message]:
    user = task_text(task)
    if previous is not None:
        user += f"\n\n{_previous(task, previous)}\n\nGive a corrected answer."
    if retry:
        user += f"\n\n{retry_note()}"
    user += f"\n\nEnd with {answer_format(task)}."
    return _messages(persona.system_prompt, _with_memory(user, memory_text))


# ── evidence: what can be checked without a model ───────────────────────────


def _position(task: Task, p: Proposal) -> str:
    if task.kind == "code":
        return _code_block(p.answer) if p.answer else "(no code found in the reply)"
    reasoning = p.response.strip()
    if len(reasoning) > REASONING_SHOWN_CHARS:
        reasoning = "… " + reasoning[-REASONING_SHOWN_CHARS:]
    return f"{reasoning}\nAnswer: {p.answer or '(no answer found)'}"


def evidence_text(task: Task, evidence: Evidence, names: list[str]) -> str:
    lines = []
    for name in names:
        if task.kind == "code":
            passed, total = evidence.examples.get(name, [0, 0])
            lines.append(
                f"- {name}: passes {passed} of {total} shown examples"
                if total
                else f"- {name}: no shown examples to run"
            )
        else:
            lines.append(
                f"- {name}: {evidence.answers.get(name) or '(no answer found)'}"
            )
    if task.kind == "code" and evidence.groups:
        alike = [" = ".join(g) for g in evidence.groups]
        lines.append(
            "Groups whose code gives identical outputs on the team's own test inputs: "
            + "; ".join(alike)
            + ". Programs written independently that agree are more often right."
        )
        for d in evidence.differences:
            outs = "; ".join(f"{' = '.join(g)} → {o}" for g, o in d["outputs"])
            lines.append(f"On {d['call']}: {outs}")
    return "\n".join(lines)


# ── discussion: only when the specialists disagree ──────────────────────────


def discussion_messages(
    persona: Persona,
    task: Task,
    own: Proposal,
    teammates: list[Proposal],
    evidence: Evidence,
    memory_text: str = "",
    previous: Decision | None = None,
    style: str = "critique",
) -> list[Message]:
    others = "\n\n".join(f"### {p.specialist}\n{_position(task, p)}" for p in teammates)
    names = [own.specialist, *(p.specialist for p in teammates)]
    sections = [task_text(task)]
    if previous is not None:
        sections.append(_previous(task, previous))
    sections += [
        f"Your answer:\n{_position(task, own)}",
        f"Your teammates' answers:\n\n{others}",
        f"Checked results:\n{evidence_text(task, evidence, names)}",
    ]
    rules = DISCUSSION_RULES[style].format(answer_format=answer_format(task))
    system = f"{persona.system_prompt.strip()}\n\n{rules}"
    return _messages(system, _with_memory("\n\n".join(sections), memory_text))


DISCUSSION_RULES = {
    "critique": (
        "Your team has answered a problem and disagrees. Reply in this order:\n"
        "1. One line per teammate: 'AGREE <name>: <reason>' or "
        "'DISAGREE <name>: <reason>'. Base each reason on the checked results and "
        "on a specific step of their reasoning.\n"
        "2. Check your own answer again.\n"
        "3. End with {answer_format}."
    ),
    "hold": (
        "Your team has answered a problem and disagrees. Reply in this order:\n"
        "1. One line per teammate: 'AGREE <name>: <reason>' or "
        "'DISAGREE <name>: <reason>'. Base each reason on the checked results and "
        "on a specific step of their reasoning.\n"
        "2. Decide whether to change your answer. Change it only if a teammate's "
        "reasoning or the checked results show a specific mistake in yours; name the "
        "mistake. Being outnumbered is not a reason to change. Otherwise keep your "
        "answer.\n"
        "3. End with {answer_format}."
    ),
    "du": (
        "Using the reasoning from your teammates as additional advice, give an "
        "updated answer. Examine your solution and theirs step by step. End with "
        "{answer_format}."
    ),
}


# ── the choice ──────────────────────────────────────────────────────────────


def coordinator_messages(
    persona: Persona,
    task: Task,
    positions: list[Proposal],
    evidence: Evidence,
    memory_text: str = "",
    previous: Decision | None = None,
    may_reject: bool = False,
) -> list[Message]:
    """Positions are listed in the order given; callers shuffle them per decision
    so position bias cannot masquerade as trust in a specialist. No subject or
    track record is shown: which specialist to trust on which kind of problem is
    learned in the coordinator's weights from its rewarded choices, as the
    striatum learns which cortical channel to gate."""
    names = [p.specialist for p in positions]
    sections = [task_text(task)]
    if previous is not None:
        sections.append(_previous(task, previous))
    sections.append(
        "Your specialists' positions:\n\n"
        + "\n\n".join(f"### {p.specialist}\n{_position(task, p)}" for p in positions)
    )
    sections.append(f"Checked results:\n{evidence_text(task, evidence, names)}")
    options = ", ".join(names)
    if may_reject:
        sections.append(
            "Choose the one specialist whose position is most likely to be correct. "
            "Name the specialist by its letter, never the answer itself. If you are "
            "confident that none of them is correct, you may send the specialists back "
            "to try again instead, once. Explain briefly, then end with a line of the "
            "form 'FINAL CHOICE: <name>', where <name> is the specialist's letter, one "
            f"of: {options}, none."
        )
    else:
        sections.append(
            "Choose the one specialist whose position is most likely to be correct. "
            "Name the specialist by its letter, never the answer itself. Explain "
            "briefly, then end with a line of the form 'FINAL CHOICE: <name>', where "
            f"<name> is the specialist's letter, one of: {options}."
        )
    return _messages(
        persona.system_prompt, _with_memory("\n\n".join(sections), memory_text)
    )


REJECT_ALL = "none"


LABELS = "WXYZ"


def specialist_labels(names: list[str]) -> dict[str, str]:
    """The fixed letter each specialist is known by to the coordinator, so its
    trust in each one is learned from outcomes, not suggested by a name. The
    letters come from the end of the alphabet so they never coincide with a
    multiple-choice option (A to J)."""
    return {name: LABELS[i] for i, name in enumerate(names)}


_LETTER_CHOICE = re.compile(
    r"(?i:final choice|choice|choose|chosen|choosing|select|go with|specialist|position)"
    r"\W{0,4}([W-Z])\b"
)


def parse_choice(text: str, names: list[str], may_reject: bool = False) -> str | None:
    """The named position after 'FINAL CHOICE:' ('none' when rejecting is
    allowed), else the last position named in the reply's final lines."""
    found = re.findall(
        r"FINAL CHOICE\W*[:\-]?\s*\**\s*([A-Za-z]+)", text, re.IGNORECASE
    )
    allowed = {n.lower(): n for n in ([*names, REJECT_ALL] if may_reject else names)}
    for word in reversed(found):
        if word.lower() in allowed:
            return allowed[word.lower()]
    tail = text[-300:]
    if all(len(n) == 1 for n in names):
        letters = [m for m in _LETTER_CHOICE.findall(tail) if m in names]
        return letters[-1] if letters else None
    lowered = tail.lower()
    last = max(((lowered.rfind(n.lower()), n) for n in names), default=(-1, None))
    return last[1] if last[0] >= 0 else None


def coordinator_rejection() -> str:
    """A well-formed decision to send the specialists back to try again."""
    return (
        "None of these positions holds up against the checked results.\n\n"
        f"FINAL CHOICE: {REJECT_ALL}"
    )


def retry_note() -> str:
    return (
        "Your coordinator rejected every answer your team gave to this problem. "
        "Solve it again from the start, checking each step."
    )


def coordinator_endorsement(label: str) -> str:
    """A well-formed choice of one specialist's position, by its letter."""
    return (
        f"{label}'s reasoning holds up and its answer fits the checked "
        f"results.\n\nFINAL CHOICE: {label}"
    )


# ── sleep: explaining a known answer in the specialist's own words ──────────


def rationalize_messages(
    persona: Persona, task: Task, known_answer: str
) -> list[Message]:
    """STaR-style rationalisation: the specialist is shown the accepted answer and
    works its way to it, so training targets keep a full line of reasoning."""
    shown = _code_block(known_answer) if task.kind == "code" else known_answer
    return _messages(
        persona.system_prompt,
        f"{task_text(task)}\n\n---\nA correct answer to this problem is:\n{shown}\n\n"
        "Solve the problem yourself, in your own way, showing the reasoning that leads to "
        "this answer. Do not mention that you were given the answer. End in exactly the "
        "format the problem asks for.",
    )


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


# ── sleep: reflection on what went wrong ────────────────────────────────────


def reflection_messages(cases: list[str]) -> list[Message]:
    return _messages(
        "You review a team's mistakes.",
        "Cases from the last cycle where the team was told its answer was wrong and a "
        "correct answer was later found:\n\n" + "\n\n".join(cases) + "\n\n"
        "Compare each wrong answer with the correct one. For every case, say in one "
        "sentence what went wrong, one line per case:\n"
        "CASE <number>: <what went wrong>",
    )


def parse_reflection(text: str) -> list[tuple[int, str]]:
    """(case number, what went wrong) for each case explained."""
    notes: list[tuple[int, str]] = []
    for line in text.splitlines():
        line = line.strip().lstrip("-* ").strip().replace("**", "")
        if m := re.match(r"(?i)CASE\s+(\d+)\s*:\s*(.+)", line):
            notes.append((int(m.group(1)), m.group(2).strip()))
    return notes
