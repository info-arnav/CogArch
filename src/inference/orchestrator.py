"""Orchestrator — main inference loop tying specialists and coordinator together."""

import asyncio

from src.inference.coordinator import Coordinator
from src.inference.specialist import Specialist
from src.models.coordinator import CoordinatorOutput
from src.models.specialist import SpecialistOutput


class OrchestratorResult:
    """Full result of a single inference run."""

    def __init__(
        self,
        input_text: str,
        round1_outputs: dict[str, SpecialistOutput],
        round2_outputs: dict[str, SpecialistOutput],
        coordinator_output: CoordinatorOutput,
    ):
        self.input_text = input_text
        self.round1_outputs = round1_outputs
        self.round2_outputs = round2_outputs
        self.coordinator_output = coordinator_output

    @property
    def answer(self) -> str:
        return self.coordinator_output.final_answer

    @property
    def attribution(self) -> dict[str, float]:
        return self.coordinator_output.attribution

    @property
    def confidence(self) -> float:
        return self.coordinator_output.confidence


class Orchestrator:
    """Main inference loop: broadcast → Round 1 → Round 2 → Coordinator."""

    def __init__(
        self,
        specialists: dict[str, Specialist],
        coordinator: Coordinator,
        enable_revision: bool = True,
    ):
        self.specialists = specialists
        self.coordinator = coordinator
        self.enable_revision = enable_revision

    async def run(self, input_text: str) -> OrchestratorResult:
        """Full waking inference cycle."""

        # Round 1: All specialists reason independently in parallel
        round1_outputs = await self._run_round1(input_text)

        # Round 2: Each specialist sees others' outputs, optionally revises
        if self.enable_revision:
            round2_outputs = await self._run_round2(input_text, round1_outputs)
        else:
            round2_outputs = round1_outputs

        # Coordinator synthesizes all outputs
        coordinator_output = await self.coordinator.synthesize(
            input_text, round1_outputs, round2_outputs
        )

        return OrchestratorResult(
            input_text=input_text,
            round1_outputs=round1_outputs,
            round2_outputs=round2_outputs,
            coordinator_output=coordinator_output,
        )

    async def _run_round1(self, input_text: str) -> dict[str, SpecialistOutput]:
        """Broadcast input to all specialists in parallel."""
        tasks = {
            name: specialist.generate(input_text)
            for name, specialist in self.specialists.items()
        }
        results = await asyncio.gather(*tasks.values())
        return dict(zip(tasks.keys(), results))

    async def _run_round2(
        self, input_text: str, round1_outputs: dict[str, SpecialistOutput]
    ) -> dict[str, SpecialistOutput]:
        """Each specialist sees peers' Round 1 outputs and may revise."""
        tasks = {}
        for name, specialist in self.specialists.items():
            peer_outputs = {k: v for k, v in round1_outputs.items() if k != name}
            tasks[name] = specialist.revise(
                input_text, round1_outputs[name], peer_outputs
            )
        results = await asyncio.gather(*tasks.values())
        return dict(zip(tasks.keys(), results))
