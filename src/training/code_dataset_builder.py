"""Code DPO dataset builder.

Generates preference pairs from competitive round outcomes.
Key design: every round produces a DPO pair — even when both agents fail —
by ranking solutions by partial pass rate. This means no round is wasted.
"""

from __future__ import annotations

from pathlib import Path

from src.models.code import (
    CodeCompetitiveResult,
    CodeTrainingExample,
)

_MIN_PASS_RATE_DIFF = 0.0  # Any difference qualifies; tie = skipped


def build_dpo_pairs(
    results: list[CodeCompetitiveResult],
) -> dict[str, list[CodeTrainingExample]]:
    """Build per-specialist DPO pairs from competitive round results.

    For each round:
    - If A wins: A's chosen specialist gets (their_code, B's_code) as (chosen, rejected)
                 B's chosen specialist gets (A's_code, their_code) as (chosen, rejected)
    - If tie:    Both still get pairs — the specialist with higher pass rate within
                 the tie is chosen; lowest is rejected.
    - Both fail: Rank by partial pass rate — closer-to-passing is chosen.
                 This ensures failure rounds still generate learning signal.
    """
    datasets: dict[str, list[CodeTrainingExample]] = {}

    for result in results:
        pairs = _extract_pairs(result)
        for pair in pairs:
            datasets.setdefault(pair.specialist_name, []).append(pair)

    return datasets


def save(
    datasets: dict[str, list[CodeTrainingExample]],
    output_dir: str | Path = "data/training",
) -> dict[str, Path]:
    """Save per-specialist DPO datasets as JSONL."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, examples in datasets.items():
        path = out / f"{name}_code_dpo.jsonl"
        with open(path, "w") as f:
            for ex in examples:
                f.write(ex.model_dump_json() + "\n")
        paths[name] = path
    return paths


def _extract_pairs(result: CodeCompetitiveResult) -> list[CodeTrainingExample]:
    pairs = []
    problem = result.problem

    # For each specialist, compare A's solution to B's solution
    all_specialist_names = set(result.agent_a.solutions) | set(result.agent_b.solutions)

    for spec_name in all_specialist_names:
        sol_a = result.agent_a.solutions.get(spec_name)
        sol_b = result.agent_b.solutions.get(spec_name)

        if sol_a is None or sol_b is None:
            continue

        best_a = sol_a.best_attempt
        best_b = sol_b.best_attempt

        if best_a is None or best_b is None:
            continue

        rate_a = sol_a.best_pass_rate
        rate_b = sol_b.best_pass_rate

        # Skip if both have identical pass rates and identical code
        if abs(rate_a - rate_b) < 1e-6 and best_a.code.strip() == best_b.code.strip():
            continue

        # Higher pass rate = chosen; lower = rejected
        if rate_a >= rate_b:
            chosen_code, rejected_code = best_a.code, best_b.code
            chosen_rate, rejected_rate = rate_a, rate_b
        else:
            chosen_code, rejected_code = best_b.code, best_a.code
            chosen_rate, rejected_rate = rate_b, rate_a

        signal = _signal_label(chosen_rate, rejected_rate)

        pairs.append(
            CodeTrainingExample(
                problem_id=problem.task_id,
                specialist_name=spec_name,
                prompt=problem.prompt,
                chosen_code=chosen_code,
                rejected_code=rejected_code,
                chosen_pass_rate=chosen_rate,
                rejected_pass_rate=rejected_rate,
                training_signal=signal,
                source_round_id=result.timestamp,
            )
        )

    return pairs


def _signal_label(chosen_rate: float, rejected_rate: float) -> str:
    if chosen_rate == 1.0:
        return "win"
    if chosen_rate > 0 and rejected_rate == 0:
        return "partial_win"
    if chosen_rate > 0 and rejected_rate > 0:
        return "both_partial"
    return "both_fail_ranked"
