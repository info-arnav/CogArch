"""Metrics tracker — routing accuracy, vindication rate, calibration, consensus quality."""

import json
from pathlib import Path

from src.models.interaction import InteractionRecord


class MetricsTracker:
    """Computes and stores evaluation metrics from logged interactions."""

    def __init__(self, metrics_dir: str | Path = "data/metrics") -> None:
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def routing_accuracy(self, records: list[InteractionRecord]) -> float:
        """Fraction of interactions where primary specialist had the best answer.

        Only considers records that have an outcome_score and vindication data.
        """
        scored = [r for r in records if r.outcome_score is not None and r.vindication]
        if not scored:
            return 0.0
        correct = 0
        for r in scored:
            vind = r.vindication
            assert vind is not None
            if not any(vind.values()):
                correct += 1
        return correct / len(scored)

    def vindication_rate(self, records: list[InteractionRecord]) -> float:
        """Fraction of interactions where a deprioritized specialist was right."""
        scored = [r for r in records if r.vindication]
        if not scored:
            return 0.0
        vindicated = 0
        for r in scored:
            vind = r.vindication
            assert vind is not None
            if any(vind.values()):
                vindicated += 1
        return vindicated / len(scored)

    def coordinator_calibration(self, records: list[InteractionRecord]) -> float:
        """Expected Calibration Error (ECE) — lower is better.

        Bins predictions by confidence and compares to actual accuracy.
        """
        scored = [r for r in records if r.outcome_score is not None]
        if not scored:
            return 1.0

        bins: dict[int, list[tuple[float, float]]] = {i: [] for i in range(10)}
        for r in scored:
            conf = r.coordinator_output.confidence
            bin_idx = min(int(conf * 10), 9)
            assert r.outcome_score is not None
            bins[bin_idx].append((conf, r.outcome_score))

        total = len(scored)
        ece = 0.0
        for entries in bins.values():
            if not entries:
                continue
            avg_conf = sum(c for c, _ in entries) / len(entries)
            avg_acc = sum(s for _, s in entries) / len(entries)
            ece += abs(avg_acc - avg_conf) * (len(entries) / total)
        return ece

    def consensus_quality(self, records: list[InteractionRecord]) -> dict[str, float]:
        """Measure how much Round 2 revision improves answers.

        Returns dict with agreement_rate, revision_rate, improvement_rate.
        """
        if not records:
            return {
                "agreement_rate": 0.0,
                "revision_rate": 0.0,
                "improvement_rate": 0.0,
            }

        total_specialists = 0
        maintained = 0
        revised = 0
        improved = 0

        for r in records:
            for name, r2_out in r.round2_outputs.items():
                r1_out = r.round1_outputs.get(name)
                if not r1_out:
                    continue
                total_specialists += 1
                if r1_out.answer.strip() == r2_out.answer.strip():
                    maintained += 1
                else:
                    revised += 1
                    if r2_out.confidence > r1_out.confidence:
                        improved += 1

        if total_specialists == 0:
            return {
                "agreement_rate": 0.0,
                "revision_rate": 0.0,
                "improvement_rate": 0.0,
            }

        return {
            "agreement_rate": maintained / total_specialists,
            "revision_rate": revised / total_specialists,
            "improvement_rate": improved / revised if revised else 0.0,
        }

    def compute_all(self, records: list[InteractionRecord]) -> dict:
        """Compute all metrics and return as dict."""
        return {
            "routing_accuracy": self.routing_accuracy(records),
            "vindication_rate": self.vindication_rate(records),
            "coordinator_calibration_ece": self.coordinator_calibration(records),
            "consensus_quality": self.consensus_quality(records),
            "total_interactions": len(records),
        }

    def save(self, metrics: dict, label: str = "latest") -> Path:
        """Save metrics snapshot to JSON."""
        path = self.metrics_dir / f"{label}.json"
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        return path
