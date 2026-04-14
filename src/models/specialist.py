"""Data models for specialist configuration and output."""

from pydantic import BaseModel, Field


class SpecialistConfig(BaseModel):
    """Configuration loaded from a specialist YAML file."""

    name: str
    description: str = ""
    model_id: str | None = None
    temperature: float = 0.7
    max_tokens: int = 2048
    system_prompt: str = ""


class SpecialistOutput(BaseModel):
    """Output from a single specialist's reasoning process."""

    specialist_name: str
    answer: str = Field(..., description="The specialist's answer")
    reasoning_trace: str = Field(..., description="Full chain-of-thought reasoning")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    revision_notes: str | None = Field(
        default=None, description="What changed in Round 2"
    )
    endorsed: list[str] = Field(
        default_factory=list, description="Specialists this one agrees with"
    )
    challenged: list[str] = Field(
        default_factory=list, description="Specialists this one disagrees with"
    )
