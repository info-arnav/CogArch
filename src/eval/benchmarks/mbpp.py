"""MBPP benchmark loader — TRAINING ONLY, never used for evaluation.

Downloads google-research-datasets/mbpp from HuggingFace on first run,
caches to data/benchmarks/mbpp.jsonl.

Only the sanitized split is used (cleaner problem statements).
Test assertions come from test_list — typically 3 per problem.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.models.code import CodeProblem

_CACHE = Path("data/benchmarks/mbpp.jsonl")


def load(split: str = "train", limit: int | None = None) -> list[CodeProblem]:
    """Load MBPP problems. Downloads from HuggingFace if not cached."""
    if not _CACHE.exists():
        _download()

    problems: list[CodeProblem] = []
    with open(_CACHE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if raw.get("split", "train") != split:
                continue
            assertions = [a.strip() for a in raw.get("test_list", []) if a.strip()]
            if not assertions:
                continue
            prompt_text = (raw.get("text") or raw.get("prompt") or "").strip()
            if not prompt_text:
                continue
            problems.append(
                CodeProblem(
                    task_id=f"mbpp/{raw['task_id']}",
                    prompt=prompt_text,
                    entry_point=_infer_entry_point(raw.get("code", ""), assertions),
                    test_assertions=assertions,
                    source="mbpp",
                    difficulty=_infer_difficulty(assertions),
                )
            )
            if limit and len(problems) >= limit:
                break

    return problems


def _infer_entry_point(code: str, assertions: list[str]) -> str:
    """Extract function name from the canonical solution or first assertion."""
    import re

    # Try to get it from the solution code
    m = re.search(r"^def (\w+)\(", code, re.MULTILINE)
    if m:
        return m.group(1)

    # Fall back: parse from first assertion like `assert func_name(...)`
    if assertions:
        m = re.search(r"assert\s+(\w+)\s*\(", assertions[0])
        if m:
            return m.group(1)

    return "solution"


def _infer_difficulty(assertions: list[str]) -> str:
    """Rough difficulty heuristic based on assertion complexity."""
    total_len = sum(len(a) for a in assertions)
    if total_len < 150:
        return "easy"
    if total_len < 350:
        return "medium"
    return "hard"


def _download() -> None:
    """Download MBPP sanitized split from HuggingFace."""
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as e:
        raise RuntimeError("pip install datasets  — required to download MBPP") from e

    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("google-research-datasets/mbpp", "sanitized")

    with open(_CACHE, "w") as f:
        for split_name in ("train", "validation", "test"):
            if split_name not in ds:
                continue
            for item in ds[split_name]:
                row = dict(item)
                row["split"] = (
                    "train" if split_name in ("train", "validation") else "test"
                )
                f.write(json.dumps(row) + "\n")
