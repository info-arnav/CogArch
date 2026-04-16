"""HumanEval benchmark loader — EVALUATION ONLY, never used for training.

Downloads openai/openai_humaneval from HuggingFace on first run,
caches to data/benchmarks/humaneval.jsonl.

Pass@1 on this benchmark is the canonical metric for comparing to
published model scores (Claude, GPT-4, Llama, DeepSeek, etc.).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.models.code import CodeProblem

_CACHE = Path("data/benchmarks/humaneval.jsonl")


def _extract_assertions(test_code: str, entry_point: str) -> list[str]:
    """Extract individual assert statements from a HumanEval check() function."""
    assertions = []
    for line in test_code.splitlines():
        stripped = line.strip()
        if stripped.startswith("assert "):
            # Replace `candidate` with the actual entry_point name
            normalized = stripped.replace("candidate", entry_point)
            assertions.append(normalized)
    return assertions


def load(limit: int | None = None) -> list[CodeProblem]:
    """Load HumanEval problems. Downloads from HuggingFace if not cached."""
    if not _CACHE.exists():
        _download()

    problems: list[CodeProblem] = []
    with open(_CACHE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            assertions = _extract_assertions(raw.get("test", ""), raw["entry_point"])
            if not assertions:
                continue
            problems.append(
                CodeProblem(
                    task_id=raw["task_id"],
                    prompt=_clean_prompt(raw["prompt"]),
                    entry_point=raw["entry_point"],
                    test_assertions=assertions,
                    source="humaneval",
                    difficulty="medium",
                )
            )
            if limit and len(problems) >= limit:
                break

    return problems


def _clean_prompt(prompt: str) -> str:
    """Return just the docstring description, stripping the function signature."""
    # Extract content between triple-quotes
    match = re.search(r'"""(.*?)"""', prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    return prompt.strip()


def _download() -> None:
    """Download HumanEval from HuggingFace datasets."""
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as e:
        raise RuntimeError(
            "pip install datasets  — required to download HumanEval"
        ) from e

    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("openai/openai_humaneval", split="test")
    with open(_CACHE, "w") as f:
        for item in ds:
            f.write(json.dumps(dict(item)) + "\n")
