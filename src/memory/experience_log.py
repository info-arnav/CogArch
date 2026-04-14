"""Experience log — append-only JSONL record of all interactions."""

import json
from pathlib import Path

from src.models.interaction import InteractionRecord


class ExperienceLog:
    """Append-only JSONL experience log for training data."""

    def __init__(self, log_dir: str | Path = "data/experience_log"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "interactions.jsonl"

    def append(self, record: InteractionRecord) -> None:
        """Append an interaction record to the log."""
        with open(self.log_file, "a") as f:
            f.write(record.model_dump_json() + "\n")

    def read_all(self) -> list[InteractionRecord]:
        """Read all records from the log."""
        if not self.log_file.exists():
            return []
        records = []
        with open(self.log_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(InteractionRecord(**json.loads(line)))
        return records

    def count(self) -> int:
        """Count total logged interactions."""
        if not self.log_file.exists():
            return 0
        with open(self.log_file) as f:
            return sum(1 for line in f if line.strip())
