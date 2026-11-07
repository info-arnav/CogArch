"""Data models for problems, agent decisions, rounds and training data."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CodeProblem(BaseModel):
    """A coding task. Tests are held by the environment and never shown to agents."""

    task_id: str
    prompt: str = Field(description="Task text shown to agents")
    entry_point: str
    tests: list[str] = Field(description="Assert statements run by the environment")
    setup_code: str = ""


class ExecutionResult(BaseModel):
    """The environment's verdict on one submission.

    `tests_passed` / `tests_total` are kept for analysis only; the agent is
    only ever shown `feedback`.
    """

    passed: bool
    tests_passed: int
    tests_total: int
    error_types: list[str] = Field(default_factory=list)
    timed_out: bool = False
    feedback: str = ""


class Proposal(BaseModel):
    """A specialist's candidate solution. Never judged by the environment directly."""

    specialist: str
    code: str


class Message(BaseModel):
    role: str
    content: str


class Decision(BaseModel):
    """One attempt: the specialists propose, the coordinator decides, the world judges."""

    attempt_number: int
    proposals: list[Proposal]
    coordinator_prompt: list[Message] = Field(
        description="The coordinator's prompt without memory (the DPO training prompt)"
    )
    coordinator_output: str
    code: str
    choice: str | None = Field(
        default=None,
        description="Specialist whose proposal was submitted, 'combined', or None",
    )
    result: ExecutionResult


class AgentResult(BaseModel):
    """One agent's attempts on one problem."""

    agent: str
    problem_id: str
    decisions: list[Decision]

    @property
    def solved(self) -> bool:
        return any(d.result.passed for d in self.decisions)

    @property
    def solution(self) -> str | None:
        """The code the world accepted, if any."""
        return next((d.code for d in self.decisions if d.result.passed), None)


class Discovery(BaseModel):
    """A solution the world accepted, and who found it."""

    code: str
    found_by: str


class RoundResult(BaseModel):
    """All agents' results on one problem, plus what each agent came to know."""

    timestamp: str = Field(default_factory=_now)
    cycle: int
    problem: CodeProblem
    results: dict[str, AgentResult]
    known_solution: dict[str, Discovery] = Field(
        default_factory=dict,
        description="agent -> accepted solution it knows (own or shared by a peer)",
    )
    probes: dict[str, list[str]] = Field(
        default_factory=dict, description="agent -> example calls it wrote itself"
    )
    agreement: dict[str, dict[str, bool]] = Field(
        default_factory=dict,
        description="agent -> proposal key -> behaves like the known solution",
    )


class PreferencePair(BaseModel):
    """A DPO pair for one role (a specialist or the coordinator) of one agent."""

    problem_id: str
    agent: str
    role: str
    prompt: list[Message]
    chosen: str
    rejected: str
    source: str = Field(description="who found the chosen solution: 'self' or a peer")


class EvalScore(BaseModel):
    """EvalPlus Pass@1 for one agent: benchmark -> {'base': x, 'plus': y}."""

    agent: str
    scores: dict[str, dict[str, float]] = Field(default_factory=dict)


class CycleMetrics(BaseModel):
    cycle: int
    rounds: int = 0
    solved_rate: dict[str, float] = Field(default_factory=dict)
    learned_from_peer: dict[str, int] = Field(
        default_factory=dict,
        description="agent -> problems it failed but received a solution for from a peer",
    )
    preference_pairs: dict[str, dict[str, int]] = Field(default_factory=dict)
    pair_sources: dict[str, dict[str, int]] = Field(default_factory=dict)
    coordinator_choices: dict[str, dict[str, int]] = Field(default_factory=dict)
    validation_pass_at_1: dict[str, float] = Field(default_factory=dict)
    rolled_back: list[str] = Field(default_factory=list)
    lessons: dict[str, list[str]] = Field(default_factory=dict)
    code_divergence: float | None = None
    outcome_disagreement: float | None = None


class ExperimentReport(BaseModel):
    started_at: str = Field(default_factory=_now)
    config: dict
    baseline_validation: dict[str, float] = Field(default_factory=dict)
    baseline_eval: list[EvalScore] = Field(default_factory=list)
    cycles: list[CycleMetrics] = Field(default_factory=list)
    final_eval: list[EvalScore] = Field(default_factory=list)
