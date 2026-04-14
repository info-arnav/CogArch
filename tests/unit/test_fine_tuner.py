"""Unit tests for fine-tuner file preparation and format conversion."""

import json
from pathlib import Path

from src.models.benchmark import TrainingExample
from src.training.fine_tuner import FineTuner


class TestFineTunerFormat:
    """Test training file format conversion (no API calls)."""

    def test_to_chat_format_default_prompt(self) -> None:
        tuner = FineTuner.__new__(FineTuner)
        example = TrainingExample(
            input="What is 2+2?",
            specialist_name="logical",
            target_output="4",
            training_signal="win",
            weight=1.0,
        )
        result = tuner._to_chat_format(example)

        assert "messages" in result
        assert len(result["messages"]) == 3
        assert result["messages"][0]["role"] == "system"
        assert "logical" in result["messages"][0]["content"]
        assert result["messages"][1]["role"] == "user"
        assert result["messages"][1]["content"] == "What is 2+2?"
        assert result["messages"][2]["role"] == "assistant"
        assert result["messages"][2]["content"] == "4"
        assert result["weight"] == 1.0

    def test_to_chat_format_custom_prompt(self) -> None:
        tuner = FineTuner.__new__(FineTuner)
        example = TrainingExample(
            input="test",
            specialist_name="creative",
            target_output="output",
            training_signal="vindicated",
            weight=2.0,
        )
        result = tuner._to_chat_format(example, "Custom system prompt")

        assert result["messages"][0]["content"] == "Custom system prompt"
        assert result["weight"] == 2.0

    def test_prepare_training_file(self, tmp_path: Path) -> None:
        tuner = FineTuner.__new__(FineTuner)
        tuner.output_dir = tmp_path

        examples = [
            TrainingExample(
                input=f"q{i}",
                specialist_name="logical",
                target_output=f"a{i}",
                training_signal="win",
                weight=1.0,
            )
            for i in range(5)
        ]

        path = tuner.prepare_training_file(examples, "logical", "System prompt")

        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 5

        first_record = json.loads(lines[0])
        assert "messages" in first_record
        assert first_record["messages"][0]["content"] == "System prompt"

    def test_prepare_training_file_jsonl_valid(self, tmp_path: Path) -> None:
        tuner = FineTuner.__new__(FineTuner)
        tuner.output_dir = tmp_path

        examples = [
            TrainingExample(
                input="test",
                specialist_name="skeptical",
                target_output="answer",
                training_signal="learn_from_winner",
                weight=0.5,
            )
        ]

        path = tuner.prepare_training_file(examples, "skeptical")

        # Every line should be valid JSON
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                assert "messages" in data
                assert len(data["messages"]) == 3

    def test_skip_if_too_few_examples(self) -> None:
        tuner = FineTuner.__new__(FineTuner)
        tuner.output_dir = Path("/tmp")

        examples = [
            TrainingExample(
                input="q",
                specialist_name="creative",
                target_output="a",
                training_signal="win",
            )
            for _ in range(5)
        ]

        result = tuner.fine_tune_specialist("creative", examples)
        assert result["status"] == "skipped"
        assert "minimum 10" in result["reason"]

    def test_save_job_manifest(self, tmp_path: Path) -> None:
        tuner = FineTuner.__new__(FineTuner)
        tuner.output_dir = tmp_path
        tuner.base_model = "gpt-4o-mini-2024-07-18"

        results = [
            {"specialist": "logical", "job_id": "ftjob-123", "status": "created"},
            {"specialist": "creative", "status": "skipped", "reason": "too few"},
        ]
        path = tuner.save_job_manifest(results, cycle_num=2)

        assert path.exists()
        data = json.loads(path.read_text())
        assert data["cycle"] == 2
        assert len(data["jobs"]) == 2


class TestFineTuneAll:
    """Test fine_tune_all logic without API calls."""

    def test_skips_small_datasets(self) -> None:
        tuner = FineTuner.__new__(FineTuner)
        tuner.output_dir = Path("/tmp")

        datasets = {
            "logical": [
                TrainingExample(
                    input="q",
                    specialist_name="logical",
                    target_output="a",
                    training_signal="win",
                )
            ]
            * 5,
            "creative": [
                TrainingExample(
                    input="q",
                    specialist_name="creative",
                    target_output="a",
                    training_signal="win",
                )
            ]
            * 3,
        }

        results = tuner.fine_tune_all(datasets)
        assert all(r["status"] == "skipped" for r in results)
