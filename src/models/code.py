"""Data models for the code competition pipeline."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class CodeProblem(BaseModel):
    """A single coding problem with hidden test cases."""

    task_id: str
    prompt: str = Field(..., description="Problem description shown to agents")
    entry_point: str = Field(..., description="Function name the agent must implement")
    test_assertions: list[str] = Field(
        ..., description="Individual assert statements — hidden from agents"
    )
    source: str = Field(default="mbpp", description="'mbpp' or 'humaneval'")
    difficulty: str = Field(default="medium", description="easy / medium / hard")


class AttemptResult(BaseModel):
    """Result of executing a single code attempt against test cases."""

    attempt_number: int
    code: str
    pass_count: int = 0
    total_tests: int = 0
    failed_assertions: list[str] = Field(default_factory=list)
    error: str = ""
    timed_out: bool = False

    @property
    def pass_rate(self) -> float:
        return self.pass_count / self.total_tests if self.total_tests > 0 else 0.0

    @property
    def passed_all(self) -> bool:
        return self.pass_count == self.total_tests and self.total_tests > 0


class SpecialistSolution(BaseModel):
    """All attempts by one specialist on one problem."""

    specialist_name: str
    attempts: list[AttemptResult] = Field(default_factory=list)

    @property
    def best_attempt(self) -> AttemptResult | None:
        if not self.attempts:
            return None
        return max(self.attempts, key=lambda a: (a.pass_rate, -a.attempt_number))

    @property
    def best_pass_rate(self) -> float:
        best = self.best_attempt
        return best.pass_rate if best else 0.0


class AgentResult(BaseModel):
    """All specialist solutions for one agent on one problem."""

    problem_id: str
    solutions: dict[str, SpecialistSolution] = Field(default_factory=dict)
    chosen_specialist: str = ""

    @property
    def best_solution(self) -> SpecialistSolution | None:
        if not self.solutions:
            return None
        return max(self.solutions.values(), key=lambda s: s.best_pass_rate)

    @property
    def pass_rate(self) -> float:
        best = self.best_solution
        return best.best_pass_rate if best else 0.0


class CodeCompetitiveResult(BaseModel):
    """Outcome of one competitive round between two agents."""

    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    problem: CodeProblem
    agent_a: AgentResult
    agent_b: AgentResult
    winner: str = Field(..., description="'a', 'b', or 'tie'")

    @property
    def winning_agent(self) -> AgentResult:
        return self.agent_a if self.winner == "a" else self.agent_b

    @property
    def losing_agent(self) -> AgentResult:
        return self.agent_b if self.winner == "a" else self.agent_a


class CodeTrainingExample(BaseModel):
    """A DPO preference pair derived from a competitive round."""

    problem_id: str
    specialist_name: str
    prompt: str = Field(..., description="Problem description (plain text)")
    chosen_code: str = Field(..., description="Higher pass-rate code")
    rejected_code: str = Field(..., description="Lower pass-rate code")
    chosen_pass_rate: float = 0.0
    rejected_pass_rate: float = 0.0
    training_signal: str = Field(
        default="dpo", description="'win', 'partial_win', 'both_fail_ranked'"
    )
    source_round_id: str = ""


class CycleMetrics(BaseModel):
    """Per-cycle tracking for the code experiment."""

    cycle: int
    humaneval_pass_at_1: float = 0.0
    mbpp_train_avg_pass_rate: float = 0.0
    specialist_win_rates: dict[str, float] = Field(default_factory=dict)
    dpo_pairs_generated: int = 0
    rounds_competed: int = 0
    both_failed_rounds: int = 0


class CodeExperimentReport(BaseModel):
    """Full report from a code self-improvement experiment."""

    model: str
    train_benchmark: str
    eval_benchmark: str
    baseline_pass_at_1: float = 0.0
    final_pass_at_1: float = 0.0
    improvement: float = 0.0
    cycles: list[CycleMetrics] = Field(default_factory=list)
    total_dpo_pairs: int = 0
    total_rounds: int = 0
