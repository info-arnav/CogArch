"""Dataset builder — assembles per-specialist training sets from curated interactions."""

from pathlib import Path

from src.models.benchmark import DPOExample, TrainingExample
from src.models.interaction import InteractionRecord


class DatasetBuilder:
    """Builds per-specialist training datasets from curated interaction records."""

    def __init__(
        self,
        output_dir: str | Path = "data/training",
        winner_weight: float = 1.0,
        loser_weight: float = 0.5,
        vindication_weight: float = 2.0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.winner_weight = winner_weight
        self.loser_weight = loser_weight
        self.vindication_weight = vindication_weight

    def build(
        self, curated: list[InteractionRecord]
    ) -> dict[str, list[TrainingExample]]:
        """Build per-specialist SFT training examples from curated interactions."""
        datasets: dict[str, list[TrainingExample]] = {}

        for record in curated:
            primary = record.primary_specialist
            is_correct = (
                record.outcome_score is not None and record.outcome_score >= 0.5
            )
            coord_conf = record.coordinator_output.confidence

            for name, r2_out in record.round2_outputs.items():
                if name not in datasets:
                    datasets[name] = []

                attribution_w = record.attribution.get(name, 1.0)

                # Vindicated: deprioritized but actually right
                if record.vindication and record.vindication.get(name):
                    datasets[name].append(
                        TrainingExample(
                            input=record.input,
                            specialist_name=name,
                            target_output=r2_out.answer,
                            training_signal="vindicated",
                            weight=self.vindication_weight,
                            source_interaction_id=record.timestamp,
                            reasoning_trace=r2_out.reasoning_trace,
                            coordinator_confidence=coord_conf,
                            attribution_weight=attribution_w,
                        )
                    )
                # Winner: primary specialist and outcome was correct
                elif name == primary and is_correct:
                    datasets[name].append(
                        TrainingExample(
                            input=record.input,
                            specialist_name=name,
                            target_output=r2_out.answer,
                            training_signal="win",
                            weight=self.winner_weight,
                            source_interaction_id=record.timestamp,
                            reasoning_trace=r2_out.reasoning_trace,
                            coordinator_confidence=coord_conf,
                            attribution_weight=attribution_w,
                        )
                    )
                # Loser: cross-specialist learning from the winner's full trace
                elif name != primary and is_correct:
                    winner_out = record.round2_outputs.get(primary)
                    if winner_out:
                        datasets[name].append(
                            TrainingExample(
                                input=record.input,
                                specialist_name=name,
                                target_output=winner_out.answer,
                                training_signal="learn_from_winner",
                                weight=self.loser_weight,
                                source_interaction_id=record.timestamp,
                                reasoning_trace=winner_out.reasoning_trace,
                                coordinator_confidence=coord_conf,
                                attribution_weight=record.attribution.get(primary, 1.0),
                            )
                        )

        return datasets

    def build_dpo(
        self, curated: list[InteractionRecord]
    ) -> dict[str, list[DPOExample]]:
        """Build per-specialist DPO preference pairs from competitive outcomes.

        For each correct interaction:
          - Winner gets: chosen = their trace, rejected = lowest-attribution peer's trace
          - Each loser gets: chosen = winner's trace, rejected = their own trace
        """
        dpo: dict[str, list[DPOExample]] = {}

        for record in curated:
            primary = record.primary_specialist
            is_correct = (
                record.outcome_score is not None and record.outcome_score >= 0.5
            )
            if not is_correct:
                continue

            winner_out = record.round2_outputs.get(primary)
            if not winner_out:
                continue

            coord_conf = round(record.coordinator_output.confidence, 2)
            chosen_text = (
                f"REASONING: {winner_out.reasoning_trace}\n"
                f"ANSWER: {winner_out.answer}\n"
                f"CONFIDENCE: {coord_conf}"
            )

            for name, r2_out in record.round2_outputs.items():
                if name not in dpo:
                    dpo[name] = []

                loser_text = (
                    f"REASONING: {r2_out.reasoning_trace}\n"
                    f"ANSWER: {r2_out.answer}\n"
                    f"CONFIDENCE: {round(r2_out.confidence, 2)}"
                )

                if name == primary:
                    # Winner: reinforce their output vs the weakest peer
                    others = [
                        (n, o) for n, o in record.round2_outputs.items() if n != name
                    ]
                    if not others:
                        continue
                    worst_name, worst_out = min(
                        others,
                        key=lambda x: record.attribution.get(x[0], 0.0),
                    )
                    rejected_text = (
                        f"REASONING: {worst_out.reasoning_trace}\n"
                        f"ANSWER: {worst_out.answer}\n"
                        f"CONFIDENCE: {round(worst_out.confidence, 2)}"
                    )
                    dpo[name].append(
                        DPOExample(
                            specialist_name=name,
                            prompt=record.input,
                            chosen=chosen_text,
                            rejected=rejected_text,
                            training_signal="dpo_win",
                            source_interaction_id=record.timestamp,
                        )
                    )
                else:
                    # Loser: learn winner's trace, contrast with own trace
                    dpo[name].append(
                        DPOExample(
                            specialist_name=name,
                            prompt=record.input,
                            chosen=chosen_text,
                            rejected=loser_text,
                            training_signal="dpo_learn",
                            source_interaction_id=record.timestamp,
                        )
                    )

        return dpo

    def save(self, datasets: dict[str, list[TrainingExample]]) -> dict[str, Path]:
        """Save per-specialist SFT datasets as JSONL files."""
        paths: dict[str, Path] = {}
        for name, examples in datasets.items():
            path = self.output_dir / f"{name}.jsonl"
            with open(path, "w") as f:
                for ex in examples:
                    f.write(ex.model_dump_json() + "\n")
            paths[name] = path
        return paths

    def save_dpo(self, dpo: dict[str, list[DPOExample]]) -> dict[str, Path]:
        """Save per-specialist DPO datasets as JSONL files."""
        paths: dict[str, Path] = {}
        for name, examples in dpo.items():
            if not examples:
                continue
            path = self.output_dir / f"{name}_dpo.jsonl"
            with open(path, "w") as f:
                for ex in examples:
                    f.write(ex.model_dump_json() + "\n")
            paths[name] = path
        return paths

    def summary(self, datasets: dict[str, list[TrainingExample]]) -> dict:
        """Summarize assembled datasets."""
        result: dict[str, dict] = {}
        for name, examples in datasets.items():
            signals: dict[str, int] = {}
            for ex in examples:
                signals[ex.training_signal] = signals.get(ex.training_signal, 0) + 1
            result[name] = {"total": len(examples), "signals": signals}
        return result
