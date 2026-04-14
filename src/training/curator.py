"""Curator — selects high-signal interactions from experience log for training."""

from src.models.interaction import InteractionRecord


class Curator:
    """Scores and selects the most valuable interactions for fine-tuning."""

    def __init__(
        self,
        max_items: int = 200,
        vindication_weight: int = 3,
        disagreement_weight: int = 2,
        loss_weight: int = 3,
    ) -> None:
        self.max_items = max_items
        self.vindication_weight = vindication_weight
        self.disagreement_weight = disagreement_weight
        self.loss_weight = loss_weight

    def priority_score(self, record: InteractionRecord) -> int:
        """Compute a priority score for a single interaction record.

        Higher score = more valuable for training.
        """
        score = 0

        # Competitive loss: agent lost this round (outcome low)
        if record.outcome_score is not None and record.outcome_score < 0.5:
            score += self.loss_weight

        # Vindication: a deprioritized specialist was actually right
        if record.vindication and any(record.vindication.values()):
            score += self.vindication_weight

        # High specialist disagreement (low agreement → confusion)
        if record.coordinator_output.specialist_agreement < 0.3:
            score += self.disagreement_weight

        # Coordinator flagged as high priority
        if record.log_priority == "high":
            score += 2

        # Clear signal: very high or very low outcome
        if record.outcome_score is not None:
            if record.outcome_score > 0.9 or record.outcome_score < 0.1:
                score += 1

        return score

    def select(self, records: list[InteractionRecord]) -> list[InteractionRecord]:
        """Select top interactions by priority score, up to max_items."""
        scored = [(self.priority_score(r), r) for r in records]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[: self.max_items]]
