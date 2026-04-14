"""Data models for interaction logging."""

from datetime import datetime

from pydantic import BaseModel, Field

from src.models.coordinator import CoordinatorOutput
from src.models.specialist import SpecialistOutput


class InteractionRecord(BaseModel):
    """A single logged interaction through the full pipeline."""

    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO timestamp",
    )
    input: str = Field(..., description="User input query")
    round1_outputs: dict[str, SpecialistOutput] = Field(
        ..., description="Round 1 specialist outputs"
    )
    round2_outputs: dict[str, SpecialistOutput] = Field(
        ..., description="Round 2 specialist outputs"
    )
    coordinator_output: CoordinatorOutput
    attribution: dict[str, float] = Field(..., description="Final attribution weights")
    primary_specialist: str
    outcome_score: float | None = Field(
        default=None, description="Filled in later by evaluation"
    )
    vindication: dict[str, bool] | None = Field(
        default=None,
        description="Which deprioritized specialists were actually right",
    )
    log_priority: str = Field(default="medium")
