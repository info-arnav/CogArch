"""Coordinator — synthesizes specialist outputs into a final answer."""

import json
import re

from src.config import load_coordinator_prompt
from src.inference.backends.base import LLMBackend
from src.models.coordinator import CoordinatorOutput, SelfState
from src.models.specialist import SpecialistOutput


class Coordinator:
    """Lightweight coordinator that synthesizes specialist outputs."""

    def __init__(
        self,
        backend: LLMBackend,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ):
        self.model = model
        self.backend = backend
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.self_state = SelfState()
        prompt_cfg = load_coordinator_prompt("synthesis")
        self._synthesis_prompt = prompt_cfg["prompt"]
        self._system_message = prompt_cfg.get(
            "system_message",
            "You are a coordination model. Respond only with valid JSON.",
        )

    async def synthesize(
        self,
        input_text: str,
        round1_outputs: dict[str, SpecialistOutput],
        round2_outputs: dict[str, SpecialistOutput],
    ) -> CoordinatorOutput:
        """Read all specialist outputs and produce a synthesis."""
        all_outputs = self._format_outputs(round1_outputs, round2_outputs)
        agreements, disagreements = self._extract_consensus(round2_outputs)

        prompt = self._synthesis_prompt.format(
            input=input_text,
            all_outputs=all_outputs,
            agreements=agreements or "none noted",
            disagreements=disagreements or "none noted",
            uncertainty=self.self_state.uncertainty,
            turn_count=self.self_state.turn_count,
        )

        messages = [
            {"role": "system", "content": self._system_message},
            {"role": "user", "content": prompt},
        ]
        raw = await self.backend.generate(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        result = self._parse_response(raw, round2_outputs)
        self._update_self_state(result)
        return result

    def _format_outputs(
        self,
        round1: dict[str, SpecialistOutput],
        round2: dict[str, SpecialistOutput],
    ) -> str:
        """Format both rounds of specialist outputs for the coordinator prompt."""
        lines = []
        for name in round1:
            r1 = round1[name]
            r2 = round2.get(name)
            lines.append(f"[{name}]")
            lines.append(f"  Round 1 — Answer: {r1.answer}")
            lines.append(f"  Round 1 — Confidence: {r1.confidence}")
            lines.append(f"  Round 1 — Reasoning: {r1.reasoning_trace}")
            if r2:
                lines.append(f"  Round 2 — Answer: {r2.answer}")
                lines.append(f"  Round 2 — Confidence: {r2.confidence}")
                if r2.revision_notes:
                    lines.append(f"  Round 2 — Changes: {r2.revision_notes}")
                if r2.endorsed:
                    lines.append(f"  Endorsed: {', '.join(r2.endorsed)}")
                if r2.challenged:
                    lines.append(f"  Challenged: {', '.join(r2.challenged)}")
            lines.append("")
        return "\n".join(lines)

    def _extract_consensus(
        self, round2: dict[str, SpecialistOutput]
    ) -> tuple[str, str]:
        """Extract agreements and disagreements from Round 2."""
        agreements = []
        disagreements = []
        for name, output in round2.items():
            for e in output.endorsed:
                agreements.append(f"{name} endorsed {e}")
            for c in output.challenged:
                disagreements.append(f"{name} challenged {c}")
        return "; ".join(agreements), "; ".join(disagreements)

    def _parse_response(
        self, raw: str, round2_outputs: dict[str, SpecialistOutput]
    ) -> CoordinatorOutput:
        """Parse JSON response from the coordinator LLM."""
        # Strip markdown code fences if present
        cleaned = re.sub(r"```json\s*", "", raw)
        cleaned = re.sub(r"```\s*$", "", cleaned).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback: construct a best-effort response
            specialist_names = list(round2_outputs.keys())
            even_weight = 1.0 / len(specialist_names) if specialist_names else 0
            data = {
                "final_answer": raw.strip(),
                "attribution": {n: even_weight for n in specialist_names},
                "primary_specialist": (
                    specialist_names[0] if specialist_names else "unknown"
                ),
                "confidence": 0.5,
                "specialist_agreement": 0.5,
                "reasoning": "Failed to parse structured output, using raw response.",
                "should_log": True,
                "log_priority": "high",
            }

        return CoordinatorOutput(
            final_answer=data.get("final_answer", ""),
            attribution=data.get("attribution", {}),
            primary_specialist=data.get("primary_specialist", "unknown"),
            confidence=float(data.get("confidence", 0.5)),
            specialist_agreement=float(data.get("specialist_agreement", 0.5)),
            reasoning=data.get("reasoning", ""),
            should_log=data.get("should_log", True),
            log_priority=data.get("log_priority", "medium"),
            updated_self_state=self.self_state.model_copy(),
        )

    def _update_self_state(self, result: CoordinatorOutput) -> None:
        """Update self-state after an interaction."""
        self.self_state.turn_count += 1
        self.self_state.uncertainty = 1.0 - result.confidence

        # Track routing history
        self.self_state.recent_routing_history.append(
            {
                "primary": result.primary_specialist,
                "confidence": result.confidence,
                "agreement": result.specialist_agreement,
            }
        )
        # Keep last 20
        if len(self.self_state.recent_routing_history) > 20:
            self.self_state.recent_routing_history = (
                self.self_state.recent_routing_history[-20:]
            )

        # Track dominant specialist streaks
        primary = result.primary_specialist
        for name in self.self_state.dominant_specialist_streak:
            if name != primary:
                self.self_state.dominant_specialist_streak[name] = 0
        self.self_state.dominant_specialist_streak[primary] = (
            self.self_state.dominant_specialist_streak.get(primary, 0) + 1
        )
