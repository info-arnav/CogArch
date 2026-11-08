"""Typed experiment configuration loaded from YAML."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, model_validator


class FeedbackLevel(str, Enum):
    """What the environment tells an agent after a wrong submission.

    Agents are never shown tests, expected outputs or the correct answer.
    """

    BINARY = "binary"  # "your solution is wrong"
    ERROR_TYPE = "error_type"  # + exception class names


class ModelConfig(BaseModel):
    base: str = Field(description="Hugging Face id of the base model")
    served_name: str = Field(description="Model name the vLLM server exposes")
    max_tokens: int = 768


class InferenceConfig(BaseModel):
    base_url: str = "http://localhost:8000/v1"
    request_timeout_seconds: float = 300.0
    max_concurrent_requests: int = 64


class AgentsConfig(BaseModel):
    names: list[str] = Field(min_length=1)
    seeds: dict[str, int]
    specialists: list[str] = Field(min_length=1)
    share_discoveries: bool = True

    @model_validator(mode="after")
    def _seed_per_agent(self) -> AgentsConfig:
        missing = set(self.names) - set(self.seeds)
        if missing:
            raise ValueError(f"agents.seeds is missing entries for {sorted(missing)}")
        return self


class EnvironmentConfig(BaseModel):
    max_attempts: int = Field(default=3, ge=1)
    feedback: FeedbackLevel = FeedbackLevel.BINARY
    execution_timeout_seconds: float = 10.0
    memory_limit_mb: int = 1024
    probes_per_problem: int = 6


class DataConfig(BaseModel):
    validation_size: int = Field(default=50, ge=1)
    split_seed: int = 0


class TrainingConfig(BaseModel):
    enabled: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-5
    beta: float = 0.1
    nll_weight: float = Field(
        default=1.0,
        description="Weight of the NLL term on chosen responses (RPO; 0 = plain DPO)",
    )
    epochs: int = 3
    batch_size: int = 2
    grad_accumulation: int = 1
    max_length: int = 3072
    min_pairs: int = 4
    global_slots: int = Field(
        default=4,
        description="DPO jobs allowed at once on this machine, across all runs",
    )
    slots_dir: Path = Path("data/runs/.training-slots")
    rollback_threshold: float = 0.1


class MemoryConfig(BaseModel):
    enabled: bool = True
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    working_capacity: int = 10
    max_episodic: int = 2000
    retrieve_k: int = 3
    min_similarity: float = 0.5
    lessons_per_sleep: int = 5
    lessons_in_prompt: int = 5


class EvaluationConfig(BaseModel):
    samples: int = Field(
        default=5, ge=1, description="Pass@1 samples per EvalPlus task"
    )
    validation_samples: int = Field(default=3, ge=1)
    benchmarks: list[str] = Field(
        default_factory=lambda: ["humaneval", "mbpp"],
        description="EvalPlus datasets, scored at the start and end of a run ([] to skip)",
    )

    @model_validator(mode="after")
    def _known_benchmarks(self) -> EvaluationConfig:
        unknown = set(self.benchmarks) - {"humaneval", "mbpp"}
        if unknown:
            raise ValueError(f"Unknown EvalPlus benchmarks: {sorted(unknown)}")
        return self


class ExperimentConfig(BaseModel):
    name: str
    cycles: int = Field(default=3, ge=1)
    rounds_per_cycle: int = Field(default=60, ge=1)
    round_batch_size: int = Field(
        default=4,
        ge=1,
        description="Problems solved concurrently; memory and sharing update between batches",
    )
    output_dir: Path = Path("data/runs")


class Config(BaseModel):
    model: ModelConfig
    inference: InferenceConfig = InferenceConfig()
    agents: AgentsConfig
    environment: EnvironmentConfig = EnvironmentConfig()
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    memory: MemoryConfig = MemoryConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    experiment: ExperimentConfig

    @property
    def run_dir(self) -> Path:
        return self.experiment.output_dir / self.experiment.name


def load_config(path: Path | str) -> Config:
    """Load and validate a config file.

    The INFERENCE_BASE_URL environment variable overrides inference.base_url.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = Config.model_validate(raw)
    base_url = os.environ.get("INFERENCE_BASE_URL")
    if base_url:
        cfg.inference.base_url = base_url
    return cfg
