"""Data models: tasks, the agent's deliberation, rounds, training data, reports."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


Kind = Literal["code", "math", "choice"]


class Task(BaseModel):
    """A problem from any domain. Gold answers and tests stay with the environment."""

    task_id: str
    domain: str = Field(description="e.g. code, math, reasoning, social, emotion")
    kind: Kind = Field(description="How answers are extracted and judged")
    prompt: str = Field(description="Problem text shown to agents")
    prompt_is_code: bool = Field(
        default=False,
        description="The prompt is a function stub (HumanEval/MBPP+ style)",
    )
    choices: list[str] = Field(
        default_factory=list, description="Options for kind=choice"
    )
    entry_point: str = Field(default="", description="Function name for kind=code")
    tests: list[str] = Field(
        default_factory=list, description="Hidden asserts, kind=code"
    )
    examples: list[str] = Field(
        default_factory=list,
        description="Asserts the problem statement shows (kind=code); used to try code",
    )
    setup_code: str = ""
    answer: str = Field(default="", description="Hidden gold answer for math/choice")
    max_attempts: int | None = Field(
        default=None,
        description="Override attempts (multiple choice allows 1, so 'wrong' cannot "
        "be used to eliminate options)",
    )


class Verdict(BaseModel):
    """The environment's judgement of one submission.

    `tests_passed` / `tests_total` and `error_types` are environment-side
    bookkeeping for analysis; the agent is only ever shown `feedback`.
    """

    passed: bool
    tests_passed: int = 0
    tests_total: int = 0
    error_types: list[str] = Field(default_factory=list)
    timed_out: bool = False
    feedback: str = ""


class Message(BaseModel):
    role: str
    content: str


class LLMCall(BaseModel):
    """One request to a part of an agent's brain, exactly as sent and answered."""

    role: str = Field(description="Specialist name or 'coordinator'")
    stage: str = Field(description="round1, round2, decide, probes, reflect, eval")
    model: str
    messages: list[Message]
    response: str
    temperature: float
    seed: int


class Proposal(BaseModel):
    """A specialist's answer in one round: its full reply and the extracted answer."""

    specialist: str
    round: int
    response: str
    answer: str


class Evidence(BaseModel):
    """What can be checked about the positions without a model.

    Every position's extracted answer; for code, how many of the problem's
    shown examples it passes and which positions behave identically on the
    team's own test inputs (CodeT-style agreement)."""

    answers: dict[str, str] = Field(default_factory=dict)
    examples: dict[str, list[int]] = Field(
        default_factory=dict, description="position -> [examples passed, examples]"
    )
    groups: list[list[str]] = Field(
        default_factory=list, description="positions that agree with each other"
    )
    differences: list[dict] = Field(
        default_factory=list,
        description="code: test inputs on which the groups' outputs differ, "
        "{call, outputs: [[group, output], ...]}",
    )

    @property
    def unanimous(self) -> bool:
        return len(self.groups) == 1 and all(self.answers.values())


class Decision(BaseModel):
    """One attempt: propose → (discuss if they disagree) → choose → judged."""

    attempt_number: int
    labels: dict[str, str] = Field(
        default_factory=dict, description="specialist -> letter the coordinator sees"
    )
    calls: list[LLMCall] = Field(default_factory=list)
    round1: list[Proposal]
    round2: list[Proposal]
    specialist_prompts: dict[str, list[Message]] = Field(
        default_factory=dict,
        description="Each specialist's round-1 prompt without memory (training prompt)",
    )
    coordinator_prompt: list[Message] = Field(
        default_factory=list,
        description="The coordinator's prompt without memory (training prompt); "
        "empty when the specialists agreed and no choice was needed",
    )
    coordinator_output: str = ""
    rejected_all: bool = Field(
        default=False,
        description="The coordinator first sent the team back to try again; round 2 "
        "holds the second answers",
    )
    could_reject: bool = Field(
        default=False,
        description="The final choice was allowed to send the team back (it had not "
        "been sent back already)",
    )
    first_choice_prompt: list[Message] = Field(default_factory=list)
    first_choice_output: str = ""
    discussed: bool = True
    probes: list[str] = Field(default_factory=list)
    evidence: Evidence = Field(default_factory=Evidence)
    chosen: str = Field(description="The specialist whose position was submitted")
    chosen_by: str = Field(
        description="coordinator, majority (the one largest agreeing group), "
        "agreement (all specialists agreed), or fallback (the coordinator named "
        "no valid position)"
    )
    submitted: str = Field(description="The full reply that was submitted")
    answer: str = Field(description="The answer extracted from the submitted reply")
    result: Verdict


class AgentResult(BaseModel):
    agent: str
    task_id: str
    decisions: list[Decision]

    @property
    def solved(self) -> bool:
        return any(d.result.passed for d in self.decisions)

    @property
    def solution(self) -> Decision | None:
        return next((d for d in self.decisions if d.result.passed), None)


class Discovery(BaseModel):
    """An answer the world accepted, with the reply that produced it."""

    answer: str
    response: str
    found_by: str


class RoundResult(BaseModel):
    """All agents on one task, and what each came to know."""

    timestamp: str = Field(default_factory=_now)
    cycle: int
    task: Task
    results: dict[str, AgentResult]
    known: dict[str, Discovery] = Field(
        default_factory=dict, description="agent -> accepted answer it knows"
    )
    correct: dict[str, dict[str, bool]] = Field(
        default_factory=dict,
        description="agent -> 'attempt:round:specialist' -> answer matches the known one",
    )
    probes: dict[str, list[str]] = Field(default_factory=dict)


class PreferencePair(BaseModel):
    task_id: str
    domain: str
    agent: str
    role: str
    prompt: list[Message]
    chosen: str
    rejected: str
    source: str = Field(description="who found the chosen answer: 'self' or a peer")
    chosen_from: str = Field(
        default="own",
        description="own (the specialist's right answer), known (the accepted reply), "
        "rationalized (the specialist's own reasoning to the known answer), anchor",
    )
    responsibility: str = Field(
        default="",
        description="winner (got it right) or owner (best domain record when nobody did)",
    )


class CycleMetrics(BaseModel):
    cycle: int
    rounds: int = 0
    solved_rate: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="agent -> domain -> solved during training"
    )
    learned_from_peer: dict[str, int] = Field(default_factory=dict)
    discussion_changed: dict[str, float] = Field(
        default_factory=dict,
        description="agent -> share of specialist answers changed by the discussion",
    )
    discussion_fixed: dict[str, float] = Field(
        default_factory=dict,
        description="agent -> of changed answers with a known truth, share wrong→right",
    )
    specialist_accuracy: dict[str, dict[str, dict[str, float]]] = Field(
        default_factory=dict,
        description="agent -> specialist -> domain -> round-1 accuracy (known truth only)",
    )
    domain_records: dict[str, dict[str, dict[str, float]]] = Field(
        default_factory=dict,
        description="agent -> specialist -> domain -> running record after this cycle",
    )
    preference_pairs: dict[str, dict[str, int]] = Field(default_factory=dict)
    pairs_by_domain: dict[str, dict[str, dict[str, int]]] = Field(
        default_factory=dict, description="agent -> role -> domain -> pairs"
    )
    pair_sources: dict[str, dict[str, int]] = Field(default_factory=dict)
    training: dict[str, dict[str, dict]] = Field(default_factory=dict)
    validation: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="agent -> domain -> whole-agent Pass@1"
    )
    validation_specialists: dict[str, dict[str, dict[str, float]]] = Field(
        default_factory=dict, description="agent -> specialist -> domain -> solo Pass@1"
    )
    validation_specialists_no_persona: dict[str, dict[str, dict[str, float]]] = Field(
        default_factory=dict,
        description="agent -> specialist -> domain -> solo Pass@1, same weights, no persona",
    )
    rolled_back: list[str] = Field(default_factory=list)
    replayed: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="agent -> role -> pairs replayed from earlier cycles",
    )
    reflections: dict[str, LLMCall] = Field(default_factory=dict)
    answer_divergence: float | None = None
    outcome_disagreement: float | None = None


class BenchmarkScores(BaseModel):
    """benchmark -> metric -> score, for one evaluated system."""

    system: str = Field(description="e.g. 'agent a', 'a/logical', 'base'")
    mode: str = Field(
        description="agent (whole), specialist (solo, own persona), plain"
    )
    scores: dict[str, dict[str, float]] = Field(default_factory=dict)


class ExperimentReport(BaseModel):
    started_at: str = Field(default_factory=_now)
    config: dict
    baseline_validation: dict[str, dict[str, float]] = Field(default_factory=dict)
    baseline_validation_specialists: dict[str, dict[str, dict[str, float]]] = Field(
        default_factory=dict
    )
    baseline_validation_specialists_no_persona: dict[
        str, dict[str, dict[str, float]]
    ] = Field(default_factory=dict)
    baseline_benchmarks: list[BenchmarkScores] = Field(default_factory=list)
    cycles: list[CycleMetrics] = Field(default_factory=list)
    final_benchmarks: list[BenchmarkScores] = Field(default_factory=list)
