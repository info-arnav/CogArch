"""Non-code tasks with checkable answers: maths, reasoning, social, emotion.

All sources are plain Parquet datasets on the Hugging Face Hub (no dataset
scripts), so they load with current `datasets` releases.

| domain    | training            | evaluation (held out)                 |
|-----------|---------------------|---------------------------------------|
| math      | GSM8K train         | GSM8K test, MATH-500                  |
| reasoning | ARC-Challenge train | ARC-Challenge test                    |
| social    | SocialIQA train     | SocialIQA validation                  |
| emotion   | —                   | EmoBench (understanding, application) |
"""

from __future__ import annotations

import ast
from typing import Any

from src.models.core import Task

LETTERS = "ABCDEFGHIJ"


def _load(repo: str, config: str | None, split: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    return list(load_dataset(repo, config, split=split))


def _as_list(value: Any) -> list[str]:
    """EmoBench stores option lists either as lists or as their string repr."""
    if isinstance(value, str):
        return [str(v) for v in ast.literal_eval(value)]
    return [str(v) for v in value]


def _choice_task(
    task_id: str, domain: str, prompt: str, choices: list[str], correct_index: int
) -> Task:
    return Task(
        task_id=task_id,
        domain=domain,
        kind="choice",
        prompt=prompt,
        choices=choices,
        answer=LETTERS[correct_index],
        max_attempts=1,
    )


# ── maths ───────────────────────────────────────────────────────────────────


def gsm8k(split: str) -> list[Task]:
    return [
        Task(
            task_id=f"gsm8k/{split}/{i}",
            domain="math",
            kind="math",
            prompt=row["question"],
            answer=row["answer"].split("####")[-1].strip().replace(",", ""),
        )
        for i, row in enumerate(_load("openai/gsm8k", "main", split))
    ]


def math500() -> list[Task]:
    return [
        Task(
            task_id=f"math500/{row['unique_id']}",
            domain="math",
            kind="math",
            prompt=row["problem"],
            answer=row["answer"],
        )
        for row in _load("HuggingFaceH4/MATH-500", None, "test")
    ]


# ── reasoning ───────────────────────────────────────────────────────────────


def arc_challenge(split: str) -> list[Task]:
    tasks = []
    for row in _load("allenai/ai2_arc", "ARC-Challenge", split):
        labels = list(row["choices"]["label"])
        if row["answerKey"] not in labels:
            continue
        tasks.append(
            _choice_task(
                f"arc/{row['id']}",
                "reasoning",
                row["question"],
                list(row["choices"]["text"]),
                labels.index(row["answerKey"]),
            )
        )
    return tasks


# ── social ──────────────────────────────────────────────────────────────────


def social_iqa(split: str) -> list[Task]:
    return [
        _choice_task(
            f"siqa/{split}/{i}",
            "social",
            f"{row['context']}\n\nQuestion: {row['question']}",
            [row["answerA"], row["answerB"], row["answerC"]],
            int(row["label"]) - 1,
        )
        for i, row in enumerate(_load("lighteval/siqa", None, split))
    ]


# ── emotion (evaluation only) ───────────────────────────────────────────────


def emobench_understanding() -> list[Task]:
    """Each scenario asks two questions: which emotion, and why (English only)."""
    tasks = []
    for row in _load("SahandSab/EmoBench", "emotional_understanding", "train"):
        if row["language"] != "en":
            continue
        emotions, causes = _as_list(row["emotion_choices"]), _as_list(
            row["cause_choices"]
        )
        subject = row["subject"]
        tasks.append(
            _choice_task(
                f"emobench/eu/{row['qid']}/emotion",
                "emotion",
                f"Scenario: {row['scenario']}\n\n"
                f"Question: What emotion would {subject} ultimately feel in this situation?",
                emotions,
                emotions.index(row["emotion_label"]),
            )
        )
        tasks.append(
            _choice_task(
                f"emobench/eu/{row['qid']}/cause",
                "emotion",
                f"Scenario: {row['scenario']}\n\n"
                f"Question: Why would {subject} feel {row['emotion_label']} in this situation?",
                causes,
                causes.index(row["cause_label"]),
            )
        )
    return tasks


def emobench_application() -> list[Task]:
    tasks = []
    for row in _load("SahandSab/EmoBench", "emotional_application", "train"):
        if row["language"] != "en":
            continue
        choices = _as_list(row["choices"])
        kind = row["question type"].lower()
        tasks.append(
            _choice_task(
                f"emobench/ea/{row['qid']}",
                "emotion",
                f"Scenario: {row['scenario']}\n\n"
                f"Question: In this scenario, what is the most effective {kind} "
                f"for {row['subject']}?",
                choices,
                choices.index(row["label"]),
            )
        )
    return tasks
