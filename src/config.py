"""Typed experiment configuration loaded from YAML."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Literal

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
    max_tokens: int = 2048


class InferenceConfig(BaseModel):
    base_url: str = "http://localhost:8000/v1"
    request_timeout_seconds: float = 300.0
    max_concurrent_requests: int = 64


class GradingConfig(BaseModel):
    """xVerify judges maths and choice replies (MASLab's default grader); the
    base model extracts code when EvalPlus's sanitiser finds none. Without a
    verifier URL, rule-based extraction grades instead (tests only)."""

    verifier_url: str | None = None
    verifier_model: str = "IAAR-Shanghai/xVerify-3B-Ia"
    extract_code_with_model: bool = True


class AgentsConfig(BaseModel):
    names: list[str] = Field(min_length=1)
    seeds: dict[str, int]
    specialists: list[str] = Field(min_length=1)
    share_discoveries: bool = True
    discussion: Literal["disagreement", "always", "never"] = Field(
        default="disagreement",
        description="When specialists discuss: only when their answers disagree "
        "(default), on every problem, or never",
    )
    personas: Literal["full", "neutral"] = Field(
        default="full",
        description="Specialists' system prompts: their personas, or one neutral "
        "team prompt for all (the no-persona control)",
    )
    selector: Literal["coordinator", "majority"] = Field(
        default="coordinator",
        description="Who picks the submitted position when specialists disagree: "
        "the coordinator, or the largest agreeing group with the coordinator "
        "breaking ties",
    )
    allow_retry: bool = Field(
        default=True,
        description="The coordinator may reject every answer once per attempt and "
        "send the specialists back to answer again",
    )
    discussion_style: Literal["critique", "hold", "du"] = Field(
        default="critique",
        description="Discussion instructions (src/inference/prompts.DISCUSSION_RULES): "
        "critique each teammate; critique but keep one's answer unless shown a "
        "mistake; or Du et al.'s debate prompt",
    )

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
    domains: list[str] = Field(
        default_factory=lambda: ["code", "math", "reasoning", "social"],
        description="Training domains (see src/tasks)",
    )
    validation_per_domain: int = Field(default=25, ge=1)
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
    gradient_checkpointing: bool = Field(
        default=False,
        description="Recompute activations to save GPU memory, at about a third "
        "more training time; needed only where memory is tight",
    )
    min_pairs: int = 4
    winner_take_all: bool = Field(
        default=True, description="Only the best solo winner learns each task"
    )
    conscience: float = Field(
        default=0.5, description="Homeostasis: handicap for specialists that win often"
    )
    blame_adopted: bool = Field(
        default=True,
        description="A specialist whose adopted answer was wrong learns the correct one",
    )
    surprise_threshold: float | None = Field(
        default=0.5, description="Surprising pairs count twice (None disables)"
    )
    owner_learns: bool = Field(
        default=True,
        description="When no specialist was right alone, the one with the best "
        "domain record learns the found answer",
    )
    rationalize: bool = Field(
        default=True,
        description="When a specialist learns an answer someone else found, train on "
        "its own reasoning to that answer (checked to reach it) instead of the "
        "other's reply",
    )
    replay_ratio: float = Field(
        default=0.5,
        description="Sleep replay: pairs from earlier cycles mixed into each "
        "training set, as a share of the new pairs (0 disables)",
    )
    anchor_ratio: float = Field(
        default=0.5,
        description="Extra pairs from the specialist's own right answers in every "
        "domain, as a share of its main pairs (0 disables)",
    )
    global_slots: int = Field(
        default=4,
        description="DPO jobs allowed at once on this machine, across all runs",
    )
    slots_dir: Path = Path("data/runs/.training-slots")
    rollback_threshold: float = 0.1


class MemoryConfig(BaseModel):
    enabled: bool = True
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    max_episodic: int = 2000
    retrieve_k: int = 3
    min_similarity: dict[str, float] = Field(
        default_factory=lambda: {
            "code": 0.815,
            "math": 0.746,
            "reasoning": 0.689,
            "social": 0.824,
        },
        description="Per domain: the 99.5th percentile of similarity between random "
        "training problems (bge-small-en-v1.5), so only unusually close problems "
        "are recalled",
    )


class EvaluationConfig(BaseModel):
    benchmarks: list[str] = Field(
        default_factory=lambda: [
            "humaneval",
            "mbpp",
            "gsm8k",
            "math500",
            "arc_challenge",
            "siqa",
            "emobench_eu",
            "emobench_ea",
        ],
        description="Held-out benchmarks (see src/tasks.BENCHMARKS); [] to skip",
    )
    max_items: int | None = Field(
        default=300,
        description="Fixed subset per benchmark for agent and specialist modes "
        "(EvalPlus code benchmarks always run in full; plain mode always runs in full)",
    )
    subset_seed: int = 0
    samples: int = Field(default=1, ge=1, description="Answers per task, agent mode")
    validation_samples: int = Field(default=2, ge=1)
    specialists_every_cycle: bool = Field(
        default=False,
        description="Validate each specialist alone (with and without persona) "
        "after every sleep; otherwise only before training and after the last cycle",
    )
    specialists: bool = Field(
        default=True, description="Score each specialist alone with its persona"
    )
    memory: bool = Field(
        default=True,
        description="The whole agent reads (never writes) its memory on benchmarks",
    )
    plain: bool = Field(
        default=True,
        description="Score single models with no persona (base model at the start; "
        "every fine-tuned specialist at the end), comparable to published scores",
    )

    @model_validator(mode="after")
    def _known_benchmarks(self) -> EvaluationConfig:
        from src.tasks import BENCHMARKS

        unknown = set(self.benchmarks) - set(BENCHMARKS)
        if unknown:
            raise ValueError(f"Unknown benchmarks: {sorted(unknown)}")
        return self


class EventsConfig(BaseModel):
    stream_tokens: bool = Field(
        default=False,
        description="Log every reply as it is generated (for watching live)",
    )


class ExperimentConfig(BaseModel):
    name: str
    cycles: int = Field(default=5, ge=1)
    rounds_per_cycle: int = Field(default=100, ge=1)
    round_batch_size: int = Field(
        default=4,
        ge=1,
        description="Problems solved concurrently; memory and sharing update between batches",
    )
    output_dir: Path = Path("data/runs")


class Config(BaseModel):
    model: ModelConfig
    inference: InferenceConfig = InferenceConfig()
    grading: GradingConfig = GradingConfig()
    agents: AgentsConfig
    environment: EnvironmentConfig = EnvironmentConfig()
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    memory: MemoryConfig = MemoryConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    events: EventsConfig = EventsConfig()
    experiment: ExperimentConfig

    @property
    def run_dir(self) -> Path:
        return self.experiment.output_dir / self.experiment.name


def load_config(path: Path | str) -> Config:
    """Load and validate a config file.

    For machines with several GPUs, environment variables place a run:
    INFERENCE_BASE_URL (its model server), VERIFIER_URL (the grader) and
    TRAINING_SLOTS_DIR (training slots shared by the runs on its GPU).
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = Config.model_validate(raw)
    if base_url := os.environ.get("INFERENCE_BASE_URL"):
        cfg.inference.base_url = base_url
    if verifier := os.environ.get("VERIFIER_URL"):
        cfg.grading.verifier_url = verifier
    if slots := os.environ.get("TRAINING_SLOTS_DIR"):
        cfg.training.slots_dir = Path(slots)
    return cfg
