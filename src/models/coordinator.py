"""Data models for coordinator output and self-state."""

from pydantic import BaseModel, Field


class SelfState(BaseModel):
    """The coordinator's persistent self-awareness state."""

    uncertainty: float = Field(
        default=0.5, ge=0.0, le=1.0, description="System uncertainty level"
    )
    recent_routing_history: list[dict] = Field(
        default_factory=list, description="Last N routing decisions"
    )
    turn_count: int = Field(default=0, description="Turns in current session")
    dominant_specialist_streak: dict[str, int] = Field(
        default_factory=dict,
        description="Consecutive times each specialist was primary",
    )


class CoordinatorOutput(BaseModel):
    """Output from the coordinator's synthesis pass."""

    final_answer: str = Field(..., description="Synthesized response")
    attribution: dict[str, float] = Field(
        ..., description="Specialist contribution weights"
    )
    primary_specialist: str = Field(..., description="Highest-weighted specialist")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Overall system confidence"
    )
    specialist_agreement: float = Field(
        ..., ge=0.0, le=1.0, description="How much specialists agreed"
    )
    reasoning: str = Field(
        ..., description="Why the coordinator weighted specialists this way"
    )
    updated_self_state: SelfState = Field(
        ..., description="Self-state after this interaction"
    )
    should_log: bool = Field(default=True, description="Flag for experience log")
    log_priority: str = Field(default="medium", description="high | medium | low")
