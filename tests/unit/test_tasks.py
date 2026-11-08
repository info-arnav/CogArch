"""Task loaders (with the Hub replaced by fixed rows) and the curriculum."""

import pytest

from src import tasks
from src.models.core import Task
from src.tasks import Curriculum, benchmark_subset, general


@pytest.fixture
def rows(monkeypatch):
    data = {
        ("openai/gsm8k", "main", "train"): [
            {"question": "2+2?", "answer": "2+2=4\n#### 4"},
            {"question": "Big?", "answer": "#### 1,000"},
        ],
        ("allenai/ai2_arc", "ARC-Challenge", "test"): [
            {
                "id": "q1",
                "question": "Sky?",
                "answerKey": "B",
                "choices": {"text": ["green", "blue"], "label": ["A", "B"]},
            },
            {
                "id": "q2",
                "question": "Odd?",
                "answerKey": "3",
                "choices": {"text": ["x", "y", "z"], "label": ["1", "2", "3"]},
            },
        ],
        ("lighteval/siqa", None, "validation"): [
            {
                "context": "Jo lost.",
                "question": "Feels?",
                "answerA": "happy",
                "answerB": "sad",
                "answerC": "bored",
                "label": "2",
            },
        ],
        ("SahandSab/EmoBench", "emotional_understanding", "train"): [
            {
                "qid": "1",
                "language": "en",
                "scenario": "S",
                "subject": "Ann",
                "emotion_choices": ["Joy", "Anger"],
                "emotion_label": "Anger",
                "cause_choices": "['x', 'y']",
                "cause_label": "y",
            },
            {
                "qid": "2",
                "language": "zh",
                "scenario": "S",
                "subject": "B",
                "emotion_choices": ["a"],
                "emotion_label": "a",
                "cause_choices": ["b"],
                "cause_label": "b",
            },
        ],
        ("SahandSab/EmoBench", "emotional_application", "train"): [
            {
                "qid": "1",
                "language": "en",
                "scenario": "S",
                "subject": "Sam",
                "question type": "Action",
                "choices": ["p", "q"],
                "label": "q",
            },
        ],
    }
    monkeypatch.setattr(
        general, "_load", lambda repo, config, split: data[(repo, config, split)]
    )


def test_gsm8k_answers_are_plain_numbers(rows):
    out = general.gsm8k("train")
    assert [t.answer for t in out] == ["4", "1000"]
    assert all(
        t.kind == "math" and t.domain == "math" and "####" not in t.prompt for t in out
    )


def test_arc_handles_numeric_labels(rows):
    out = general.arc_challenge("test")
    assert [t.answer for t in out] == ["B", "C"]
    assert out[0].max_attempts == 1 and out[0].domain == "reasoning"


def test_siqa(rows):
    (t,) = general.social_iqa("validation")
    assert (
        t.answer == "B"
        and t.choices == ["happy", "sad", "bored"]
        and t.domain == "social"
    )


def test_emobench_english_only_two_questions_each(rows):
    eu = general.emobench_understanding()
    assert [t.task_id for t in eu] == ["emobench/eu/1/emotion", "emobench/eu/1/cause"]
    assert [t.answer for t in eu] == ["B", "B"]
    (ea,) = general.emobench_application()
    assert ea.answer == "B" and "most effective action" in ea.prompt


def _fake(domain: str, n: int) -> list[Task]:
    return [
        Task(
            task_id=f"{domain}/{i}", domain=domain, kind="math", prompt="p", answer="1"
        )
        for i in range(n)
    ]


def test_curriculum_is_balanced_disjoint_and_deterministic(monkeypatch):
    monkeypatch.setattr(
        tasks,
        "TRAIN_SOURCES",
        {d: (lambda d=d: _fake(d, 50)) for d in ["math", "social"]},
    )
    c = Curriculum(
        ["math", "social"],
        validation_per_domain=5,
        rounds_per_cycle=10,
        cycles=3,
        seed=0,
    )
    assert len(c.validation) == 10
    seen = {t.task_id for t in c.validation}
    for n in (1, 2, 3):
        cycle = c.cycle(n)
        assert sorted(t.domain for t in cycle).count("math") == 5
        ids = {t.task_id for t in cycle}
        assert not ids & seen
        seen |= ids
    again = Curriculum(["math", "social"], 5, 10, 3, seed=0)
    assert [t.task_id for t in again.cycle(2)] == [t.task_id for t in c.cycle(2)]


def test_curriculum_rejects_uneven_split(monkeypatch):
    monkeypatch.setattr(tasks, "TRAIN_SOURCES", {"math": lambda: _fake("math", 50)})
    with pytest.raises(ValueError):
        Curriculum(["math", "social"], 1, 9, 1, 0)


def test_benchmark_subset_is_fixed():
    items = _fake("math", 100)
    a = benchmark_subset(items, 10, seed=3)
    assert len(a) == 10 and a == benchmark_subset(list(reversed(items)), 10, seed=3)
    assert benchmark_subset(items, None, 0) == items
