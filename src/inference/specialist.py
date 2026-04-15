"""Specialist — one class, driven by YAML config. Handles Round 1 and Round 2."""

import json
import re
from pathlib import Path

from src.inference.backends.base import LLMBackend
from src.models.specialist import SpecialistConfig, SpecialistOutput


class Specialist:
    """A single specialist driven by its YAML personality config."""

    def __init__(
        self,
        config: SpecialistConfig,
        backend: LLMBackend,
        default_model: str = "llama3:8b",
        registry: object | None = None,
    ):
        self.config = config
        self.backend = backend
        self._base_system_prompt = config.system_prompt
        if registry is not None:
            from src.training.model_registry import ModelRegistry

            assert isinstance(registry, ModelRegistry)
            self.model = registry.latest(config.name)
        else:
            self.model = config.model_id or default_model

    @property
    def name(self) -> str:
        return self.config.name

    async def generate(self, input_text: str) -> SpecialistOutput:
        """Round 1: Independent reasoning on the input."""
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": input_text},
        ]
        raw = await self.backend.generate(
            messages=messages,
            model=self.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return self._parse_output(raw)

    async def revise(
        self,
        input_text: str,
        own_output: SpecialistOutput,
        peer_outputs: dict[str, SpecialistOutput],
    ) -> SpecialistOutput:
        """Round 2: See other specialists' outputs, optionally revise."""
        peers_text = "\n\n".join(
            f"[{name}] (confidence: {out.confidence})\n"
            f"Reasoning: {out.reasoning_trace}\n"
            f"Answer: {out.answer}"
            for name, out in peer_outputs.items()
        )

        revision_prompt = (
            f"You are the {self.config.name} specialist. You already produced your answer.\n"
            f"Now review the other specialists' answers and reasoning.\n\n"
            f"YOUR ROUND 1 OUTPUT:\n"
            f"Reasoning: {own_output.reasoning_trace}\n"
            f"Answer: {own_output.answer}\n"
            f"Confidence: {own_output.confidence}\n\n"
            f"OTHER SPECIALISTS' OUTPUTS:\n{peers_text}\n\n"
            f"You may:\n"
            f"- ENDORSE another specialist's reasoning (say whose and why)\n"
            f"- CHALLENGE another specialist's reasoning (say whose and why)\n"
            f"- REVISE your own answer based on what you've seen\n"
            f"- MAINTAIN your original answer if you still believe it's correct\n\n"
            f"Output your final answer using the same format:\n"
            f"REASONING: <your reasoning>\n"
            f"ANSWER: <your answer>\n"
            f"CONFIDENCE: <0.0 to 1.0>\n"
            f"ENDORSED: <comma-separated specialist names, or none>\n"
            f"CHALLENGED: <comma-separated specialist names, or none>\n"
            f"REVISION_NOTES: <what changed and why, or 'no changes'>"
        )

        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": revision_prompt},
        ]
        raw = await self.backend.generate(
            messages=messages,
            model=self.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return self._parse_output(raw, is_revision=True)

    def _parse_output(self, raw: str, is_revision: bool = False) -> SpecialistOutput:
        """Parse structured output from the LLM response."""
        reasoning = self._extract_field(raw, "REASONING", raw) or raw
        answer = self._extract_field(raw, "ANSWER", raw.strip()) or raw.strip()
        confidence_str = self._extract_field(raw, "CONFIDENCE", "0.5") or "0.5"

        try:
            confidence = float(confidence_str.strip())
            confidence = max(0.0, min(1.0, confidence))
        except ValueError:
            confidence = 0.5

        endorsed = []
        challenged = []
        revision_notes = None

        if is_revision:
            endorsed_str = self._extract_field(raw, "ENDORSED", "")
            challenged_str = self._extract_field(raw, "CHALLENGED", "")
            revision_notes = self._extract_field(raw, "REVISION_NOTES", None)

            if endorsed_str and endorsed_str.lower() != "none":
                endorsed = [s.strip() for s in endorsed_str.split(",") if s.strip()]
            if challenged_str and challenged_str.lower() != "none":
                challenged = [s.strip() for s in challenged_str.split(",") if s.strip()]

        return SpecialistOutput(
            specialist_name=self.config.name,
            answer=answer,
            reasoning_trace=reasoning,
            confidence=confidence,
            revision_notes=revision_notes,
            endorsed=endorsed,
            challenged=challenged,
        )

    def inject_few_shot_examples(self, path: str | Path, max_k: int = 5) -> int:
        """Load top-K curated examples and prepend to system prompt as few-shot demos.

        Prioritizes vindicated > win > learn_from_winner, then by weight descending.
        Returns the number of examples actually injected.
        """
        _path = Path(path)
        if not _path.exists():
            return 0

        examples: list[dict] = []
        with open(_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        examples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not examples:
            return 0

        _priority = {"vindicated": 0, "win": 1, "learn_from_winner": 2}
        examples.sort(
            key=lambda e: (
                _priority.get(e.get("training_signal", ""), 3),
                -e.get("weight", 1.0),
            )
        )
        top_k = examples[:max_k]

        few_shot_block = "\n\n--- EXPERIENCE: EXAMPLES FROM PAST INTERACTIONS ---\n"
        for ex in top_k:
            few_shot_block += f"\nQ: {ex['input']}\n" f"A: {ex['target_output']}\n"
        few_shot_block += "--- END EXAMPLES ---\n"

        self.config = self.config.model_copy(
            update={"system_prompt": self._base_system_prompt + few_shot_block}
        )
        return len(top_k)

    def clear_few_shot_examples(self) -> None:
        """Reset system prompt to original, removing all injected examples."""
        self.config = self.config.model_copy(
            update={"system_prompt": self._base_system_prompt}
        )

    @staticmethod
    def _extract_field(text: str, field: str, default: str | None) -> str | None:
        """Extract a labeled field from structured LLM output."""
        pattern = rf"{field}\s*:\s*(.+?)(?=\n[A-Z_]+\s*:|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return default
