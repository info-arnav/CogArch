"""Shared test fixtures — mock LLM backend and helpers."""

import json

import pytest

from src.inference.backends.base import LLMBackend
from src.models.coordinator import CoordinatorOutput, SelfState
from src.models.specialist import SpecialistConfig, SpecialistOutput


class MockBackend(LLMBackend):
    """Deterministic mock backend for testing without API calls."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self._responses = responses or {}
        self._call_log: list[dict] = []
        self._default_specialist_response = (
            "REASONING: This is test reasoning.\n"
            "ANSWER: test answer\n"
            "CONFIDENCE: 0.85"
        )
        self._default_coordinator_response = json.dumps(
            {
                "final_answer": "test synthesis",
                "attribution": {"logical": 0.6, "creative": 0.4},
                "primary_specialist": "logical",
                "confidence": 0.8,
                "specialist_agreement": 0.7,
                "reasoning": "Mock coordinator reasoning",
                "should_log": True,
                "log_priority": "medium",
            }
        )

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        self._call_log.append(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        # Check if we have a specific response for this model
        if model in self._responses:
            return self._responses[model]

        # Check user message content for matching
        user_msg = messages[-1]["content"] if messages else ""
        for key, response in self._responses.items():
            if key in user_msg:
                return response

        # Default: coordinator or specialist based on prompt content
        sys_msg = (
            messages[0]["content"]
            if messages and messages[0]["role"] == "system"
            else ""
        )
        if (
            "synthesize" in sys_msg.lower()
            or "coordinator" in sys_msg.lower()
            or "final_answer" in user_msg.lower()
        ):
            return self._default_coordinator_response
        return self._default_specialist_response

    @property
    def call_count(self) -> int:
        return len(self._call_log)


class CorrectAnswerBackend(MockBackend):
    """Mock backend that tries to return the expected answer for benchmark items."""

    def __init__(self, answer_map: dict[str, str] | None = None) -> None:
        super().__init__()
        self._answer_map = answer_map or {}

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        self._call_log.append({"messages": messages, "model": model})
        user_msg = messages[-1]["content"] if messages else ""

        # Check for known question -> answer mapping
        for question, answer in self._answer_map.items():
            if question in user_msg:
                # Detect coordinator by system prompt content
                sys_msg = (
                    messages[0]["content"]
                    if messages and messages[0]["role"] == "system"
                    else ""
                )
                is_coordinator = (
                    "synthesize" in sys_msg.lower()
                    or "coordinator" in sys_msg.lower()
                    or "final_answer" in user_msg.lower()
                )
                if is_coordinator:
                    return json.dumps(
                        {
                            "final_answer": answer,
                            "attribution": {"logical": 0.8, "creative": 0.2},
                            "primary_specialist": "logical",
                            "confidence": 0.9,
                            "specialist_agreement": 0.9,
                            "reasoning": "Answer is clear",
                            "should_log": True,
                            "log_priority": "low",
                        }
                    )
                return (
                    f"REASONING: The answer is {answer}.\n"
                    f"ANSWER: {answer}\n"
                    f"CONFIDENCE: 0.95"
                )

        # Fallback
        return await super().generate(messages, model, temperature, max_tokens)


@pytest.fixture
def mock_backend() -> MockBackend:
    return MockBackend()


@pytest.fixture
def specialist_config() -> SpecialistConfig:
    return SpecialistConfig(
        name="logical",
        system_prompt="You are a logical specialist.",
        temperature=0.4,
    )


@pytest.fixture
def sample_specialist_output() -> SpecialistOutput:
    return SpecialistOutput(
        specialist_name="logical",
        answer="Paris",
        reasoning_trace="France's capital is Paris.",
        confidence=0.9,
    )


@pytest.fixture
def sample_coordinator_output() -> CoordinatorOutput:
    return CoordinatorOutput(
        final_answer="Paris",
        attribution={"logical": 0.7, "creative": 0.3},
        primary_specialist="logical",
        confidence=0.9,
        specialist_agreement=0.8,
        reasoning="Both specialists agree",
        should_log=True,
        log_priority="medium",
        updated_self_state=SelfState(),
    )
