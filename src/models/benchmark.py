"""Data models for benchmarks, competitive training, and sleep cycle."""

from pydantic import BaseModel, Field


class BenchmarkItem(BaseModel):
    """A single benchmark test item with ground truth."""

    question: str
    expected_answer: str
    category: str = ""
    difficulty: str = "medium"


class CompetitiveResult(BaseModel):
    """Result of a single competitive round between two agents."""

    test_item: BenchmarkItem
    agent_a_answer: str
    agent_b_answer: str
    agent_a_score: float = Field(..., ge=0.0, le=1.0)
    agent_b_score: float = Field(..., ge=0.0, le=1.0)
    winner: str = Field(..., description="'a', 'b', or 'tie'")
    agent_a_primary_specialist: str = ""
    agent_b_primary_specialist: str = ""
    agent_a_attribution: dict[str, float] = Field(default_factory=dict)
    agent_b_attribution: dict[str, float] = Field(default_factory=dict)


class TrainingExample(BaseModel):
    """A single training example assembled for specialist fine-tuning."""

    input: str
    specialist_name: str
    target_output: str
    training_signal: str = Field(
        ..., description="'win', 'learn_from_winner', or 'vindicated'"
    )
    weight: float = Field(default=1.0, description="Loss multiplier")
    source_interaction_id: str = ""
    reasoning_trace: str = Field(
        default="",
        description="Actual reasoning trace from the specialist (or winner for learn_from_winner)",
    )
    coordinator_confidence: float = Field(
        default=0.8,
        description="Real confidence score from the coordinator for this interaction",
    )
    attribution_weight: float = Field(
        default=1.0,
        description="How much the coordinator weighted this specialist",
    )


class DPOExample(BaseModel):
    """A preference pair for DPO fine-tuning."""

    specialist_name: str
    prompt: str = Field(..., description="The user question (plain text)")
    chosen: str = Field(
        ..., description="Preferred assistant response (winner's trace)"
    )
    rejected: str = Field(
        ..., description="Dispreferred assistant response (loser's trace)"
    )
    training_signal: str = Field(default="dpo", description="'dpo_win' or 'dpo_learn'")
    source_interaction_id: str = ""


class SleepReport(BaseModel):
    """Report from a complete sleep cycle."""

    sleep_cycle_num: int
    items_curated: int
    training_examples_generated: int
    specialist_improvements: dict[str, float] = Field(
        default_factory=dict, description="Accuracy delta per specialist"
    )
    routing_accuracy_before: float = 0.0
    routing_accuracy_after: float = 0.0
    checkpoints_saved: list[str] = Field(default_factory=list)
    vindication_cases_found: int = 0
    status: str = Field(
        default="success",
        description="success | no_data | failed",
    )
