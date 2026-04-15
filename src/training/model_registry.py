"""Model registry — versioned specialist model tracking with rollback.

Stored as JSON at models/registry.json.
One entry per specialist, each with an ordered version history.

Usage:
    registry = ModelRegistry()
    registry.set_baseline_score("logical", 0.42)  # call once before first fine-tune
    registry.register("logical", "logical-2026-04-15", score=0.65)
    model_name = registry.latest("logical")        # "logical-2026-04-15"
    model_name = registry.rollback("logical")      # back to "llama3:8b" if only 1 version
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

FALLBACK_MODEL = "llama3:8b"


class ModelRegistry:
    """Tracks fine-tuned model versions per specialist.

    Provides:
    - register(): record a newly fine-tuned model
    - latest(): get the current best model for a specialist
    - promote_or_rollback(): compare new score vs baseline, rollback on regression
    - rollback(): manually revert to previous version
    - set_baseline_score() / get_baseline_score(): seed the initial comparison point
    """

    def __init__(self, registry_path: str | Path = "models/registry.json") -> None:
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict] = self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> dict[str, dict]:
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                data: dict[str, dict] = json.load(f)
                return data
        return {}

    def _save(self) -> None:
        with open(self.registry_path, "w") as f:
            json.dump(self._data, f, indent=2)

    def _ensure(self, specialist_name: str) -> None:
        if specialist_name not in self._data:
            self._data[specialist_name] = {
                "versions": [],
                "latest": FALLBACK_MODEL,
                "baseline_score": None,
            }

    # ── public API ───────────────────────────────────────────────────────────

    def register(
        self,
        specialist_name: str,
        model_name: str,
        score: float | None = None,
    ) -> None:
        """Record a newly fine-tuned model as the latest for this specialist."""
        self._ensure(specialist_name)
        self._data[specialist_name]["versions"].append(
            {
                "model_name": model_name,
                "date": date.today().isoformat(),
                "score": score,
            }
        )
        self._data[specialist_name]["latest"] = model_name
        self._save()

    def latest(self, specialist_name: str) -> str:
        """Return the current best model name, falling back to llama3:8b."""
        return self._data.get(specialist_name, {}).get("latest") or FALLBACK_MODEL

    def promote_or_rollback(
        self,
        specialist_name: str,
        new_score: float,
        regression_threshold: float = 0.05,
    ) -> tuple[str, bool]:
        """Promote the latest model if new_score is not a regression, else rollback.

        Returns (model_name_to_use, was_promoted).
        Regression is defined as new_score < baseline - threshold.
        """
        self._ensure(specialist_name)
        entry = self._data[specialist_name]
        baseline = entry.get("baseline_score")

        if baseline is None or new_score >= baseline - regression_threshold:
            # Promote — update baseline to the better of the two scores
            entry["baseline_score"] = max(new_score, baseline or 0.0)
            self._save()
            return entry["latest"], True

        # Regression detected — roll back to previous version
        model_to_use = self._do_rollback(specialist_name)
        return model_to_use, False

    def rollback(self, specialist_name: str) -> str:
        """Manually revert to the previous model version."""
        return self._do_rollback(specialist_name)

    def set_baseline_score(self, specialist_name: str, score: float) -> None:
        """Set the initial baseline score before any fine-tuning begins."""
        self._ensure(specialist_name)
        self._data[specialist_name]["baseline_score"] = score
        self._save()

    def get_baseline_score(self, specialist_name: str) -> float | None:
        return self._data.get(specialist_name, {}).get("baseline_score")

    def summary(self) -> dict[str, dict]:
        """Snapshot of all specialists: latest model, version count, baseline."""
        return {
            name: {
                "latest": entry.get("latest", FALLBACK_MODEL),
                "versions": len(entry.get("versions", [])),
                "baseline_score": entry.get("baseline_score"),
            }
            for name, entry in self._data.items()
        }

    # ── internals ────────────────────────────────────────────────────────────

    def _do_rollback(self, specialist_name: str) -> str:
        self._ensure(specialist_name)
        entry = self._data[specialist_name]
        versions = entry.get("versions", [])

        if len(versions) >= 2:
            versions.pop()  # remove latest
            prev: str = versions[-1]["model_name"]
            entry["latest"] = prev
            self._save()
            return prev

        # Only one version (or none) — fall back to base model
        entry["latest"] = FALLBACK_MODEL
        self._save()
        return FALLBACK_MODEL
