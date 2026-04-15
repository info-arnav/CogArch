"""Fine-tuner — OpenAI fine-tuning API integration for specialist improvement."""

import json
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.models.benchmark import TrainingExample


class FineTuner:
    """Manages OpenAI fine-tuning jobs for specialist models.

    Converts training examples to OpenAI chat format, uploads files,
    creates fine-tuning jobs, and monitors their completion.
    """

    def __init__(
        self,
        base_model: str = "gpt-4o-mini-2024-07-18",
        output_dir: str | Path = "data/checkpoints",
        n_epochs: int = 3,
        batch_size: str = "auto",
        learning_rate_multiplier: str = "auto",
    ) -> None:
        self.client = OpenAI()
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate_multiplier = learning_rate_multiplier

    def _to_chat_format(
        self,
        example: TrainingExample,
        specialist_system_prompt: str = "",
    ) -> dict[str, Any]:
        """Convert a TrainingExample to OpenAI chat fine-tuning format."""
        system_content = specialist_system_prompt or (
            f"You are the {example.specialist_name} specialist. "
            "Provide clear, well-reasoned answers."
        )
        return {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": example.input},
                {"role": "assistant", "content": example.target_output},
            ],
            "weight": example.weight,
        }

    def prepare_training_file(
        self,
        examples: list[TrainingExample],
        specialist_name: str,
        system_prompt: str = "",
    ) -> Path:
        """Write training examples as JSONL in OpenAI chat format."""
        path = self.output_dir / f"{specialist_name}_finetune.jsonl"
        with open(path, "w") as f:
            for ex in examples:
                record = self._to_chat_format(ex, system_prompt)
                f.write(json.dumps(record) + "\n")
        return path

    def upload_file(self, path: Path) -> str:
        """Upload a training file to OpenAI. Returns the file ID."""
        with open(path, "rb") as f:
            response = self.client.files.create(file=f, purpose="fine-tune")
        return str(response.id)

    def create_job(
        self,
        training_file_id: str,
        suffix: str = "",
        validation_file_id: str | None = None,
    ) -> str:
        """Create an OpenAI fine-tuning job. Returns the job ID."""
        params: dict[str, Any] = {
            "training_file": training_file_id,
            "model": self.base_model,
            "hyperparameters": {
                "n_epochs": self.n_epochs,
                "batch_size": self.batch_size,
                "learning_rate_multiplier": self.learning_rate_multiplier,
            },
        }
        if suffix:
            params["suffix"] = suffix
        if validation_file_id:
            params["validation_file"] = validation_file_id

        job = self.client.fine_tuning.jobs.create(**params)
        return str(job.id)

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Check the status of a fine-tuning job."""
        job = self.client.fine_tuning.jobs.retrieve(job_id)
        return {
            "id": job.id,
            "status": job.status,
            "model": job.model,
            "fine_tuned_model": job.fine_tuned_model,
            "created_at": job.created_at,
            "finished_at": job.finished_at,
            "trained_tokens": job.trained_tokens,
            "error": job.error.message if job.error else None,
        }

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 30,
        timeout: int = 7200,
    ) -> dict[str, Any]:
        """Poll until the fine-tuning job completes or fails."""
        elapsed = 0
        while elapsed < timeout:
            status = self.get_job_status(job_id)
            if status["status"] in ("succeeded", "failed", "cancelled"):
                return status
            time.sleep(poll_interval)
            elapsed += poll_interval
        return self.get_job_status(job_id)

    def fine_tune_specialist(
        self,
        specialist_name: str,
        examples: list[TrainingExample],
        system_prompt: str = "",
        wait: bool = False,
    ) -> dict[str, Any]:
        """End-to-end: prepare file, upload, create job for one specialist.

        Returns dict with file_id, job_id, and optionally the final status.
        """
        if len(examples) < 10:
            return {
                "specialist": specialist_name,
                "status": "skipped",
                "reason": f"Only {len(examples)} examples (minimum 10 required)",
            }

        path = self.prepare_training_file(examples, specialist_name, system_prompt)
        file_id = self.upload_file(path)
        job_id = self.create_job(
            training_file_id=file_id,
            suffix=f"cogarch-{specialist_name}",
        )

        result: dict[str, Any] = {
            "specialist": specialist_name,
            "file_id": file_id,
            "job_id": job_id,
            "training_file": str(path),
            "num_examples": len(examples),
            "status": "created",
        }

        if wait:
            final = self.wait_for_completion(job_id)
            result.update(final)

        return result

    def fine_tune_all(
        self,
        datasets: dict[str, list[TrainingExample]],
        system_prompts: dict[str, str] | None = None,
        wait: bool = False,
    ) -> list[dict[str, Any]]:
        """Fine-tune all specialists with enough training data.

        Args:
            datasets: Per-specialist training examples from DatasetBuilder.
            system_prompts: Optional per-specialist system prompts.
            wait: If True, blocks until all jobs complete.

        Returns:
            List of job result dicts.
        """
        prompts = system_prompts or {}
        results: list[dict[str, Any]] = []
        for name, examples in datasets.items():
            result = self.fine_tune_specialist(
                specialist_name=name,
                examples=examples,
                system_prompt=prompts.get(name, ""),
                wait=wait,
            )
            results.append(result)
        return results

    def save_job_manifest(
        self,
        results: list[dict[str, Any]],
        cycle_num: int = 1,
    ) -> Path:
        """Save fine-tuning job results to a manifest file for tracking."""
        path = self.output_dir / f"finetune_cycle_{cycle_num}.json"
        with open(path, "w") as f:
            json.dump(
                {
                    "cycle": cycle_num,
                    "jobs": results,
                    "base_model": self.base_model,
                },
                f,
                indent=2,
            )
        return path
