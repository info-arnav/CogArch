"""Dataset builder — assembles per-specialist training sets from curated interactions."""

from pathlib import Path

from src.models.benchmark import TrainingExample
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
        """Build per-specialist training examples from curated interactions."""
        datasets: dict[str, list[TrainingExample]] = {}

        for record in curated:
            primary = record.primary_specialist
            is_correct = (
                record.outcome_score is not None and record.outcome_score >= 0.5
            )

            for name, r2_out in record.round2_outputs.items():
                if name not in datasets:
                    datasets[name] = []

                # Vindicated specialist: was deprioritized but actually right
                if record.vindication and record.vindication.get(name):
                    datasets[name].append(
                        TrainingExample(
                            input=record.input,
                            specialist_name=name,
                            target_output=r2_out.answer,
                            training_signal="vindicated",
                            weight=self.vindication_weight,
                            source_interaction_id=record.timestamp,
                        )
                    )
                # Winning specialist: was primary and correct
                elif name == primary and is_correct:
                    datasets[name].append(
                        TrainingExample(
                            input=record.input,
                            specialist_name=name,
                            target_output=r2_out.answer,
                            training_signal="win",
                            weight=self.winner_weight,
                            source_interaction_id=record.timestamp,
                        )
                    )
                # Losing specialist: learn from the winner's reasoning
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
                            )
                        )

        return datasets

    def save(self, datasets: dict[str, list[TrainingExample]]) -> dict[str, Path]:
        """Save per-specialist datasets as JSONL files."""
        paths: dict[str, Path] = {}
        for name, examples in datasets.items():
            path = self.output_dir / f"{name}.jsonl"
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
