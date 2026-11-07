"""Aggregate finished runs into paper tables and figures.

    python -m cli.main analyze            # writes data/analysis/

Groups runs by condition (pair = two agents, solo = one agent) and reports
means ± standard deviation across agents and replicates.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

COLORS = {"pair": "#2a78d6", "solo": "#eb6834"}
CHOICE_COLORS = {
    "logical": "#2a78d6",
    "creative": "#eb6834",
    "skeptical": "#1baf7a",
    "empathetic": "#eda100",
    "combined": "#6250d6",
}
BENCH_NAMES = {
    ("humaneval", "base"): "HumanEval",
    ("humaneval", "plus"): "HumanEval+",
    ("mbpp", "base"): "MBPP",
    ("mbpp", "plus"): "MBPP+",
}


def _mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return statistics.mean(values), sd


def _fmt(values: list[float], scale: float = 100.0, signed: bool = False) -> str:
    mean, sd = _mean_sd([v * scale for v in values])
    sign = "+" if signed and mean >= 0 else ""
    return f"{sign}{mean:.1f} ± {sd:.1f} (n={len(values)})"


def load_runs(runs_root: Path) -> list[dict[str, Any]]:
    runs = []
    for report in sorted(runs_root.glob("*/report.json")):
        data = json.loads(report.read_text())
        if not data["final_eval"]:
            continue
        agents = data["config"]["agents"]["names"]
        data["name"] = report.parent.name
        data["dir"] = report.parent
        data["condition"] = "solo" if len(agents) == 1 else "pair"
        data["agents"] = agents
        runs.append(data)
    return runs


def by_condition(runs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[run["condition"]].append(run)
    return dict(sorted(groups.items()))


# ── tables ──────────────────────────────────────────────────────────────────


def evalplus_table(groups: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "| Benchmark | Condition | Start | End | Change (pp) |",
        "|---|---|---|---|---|",
    ]
    for (bench, kind), label in BENCH_NAMES.items():
        for condition, runs in groups.items():
            start, end, delta = [], [], []
            for run in runs:
                for b, e in zip(run["baseline_eval"], run["final_eval"]):
                    if bench in b["scores"]:
                        s, f = b["scores"][bench][kind], e["scores"][bench][kind]
                        start.append(s)
                        end.append(f)
                        delta.append(f - s)
            if start:
                lines.append(
                    f"| {label} | {condition} | {_fmt(start)} | {_fmt(end)} | "
                    f"{_fmt(delta, signed=True)} |"
                )
    return "\n".join(lines)


def per_cycle_agent_values(run: dict[str, Any], key: str) -> list[list[float]]:
    """[cycle][agent] values of a per-agent cycle metric."""
    return [[c[key][a] for a in run["agents"] if a in c[key]] for c in run["cycles"]]


def validation_series(runs: list[dict[str, Any]]) -> list[list[float]]:
    """[cycle 0..N][agent values pooled across runs]."""
    cycles = max(len(r["cycles"]) for r in runs)
    series: list[list[float]] = [[] for _ in range(cycles + 1)]
    for run in runs:
        series[0].extend(run["baseline_validation"][a] for a in run["agents"])
        for i, c in enumerate(run["cycles"], start=1):
            series[i].extend(c["validation_pass_at_1"][a] for a in run["agents"])
    return series


def validation_table(groups: dict[str, list[dict[str, Any]]]) -> str:
    cycles = max(len(r["cycles"]) for runs in groups.values() for r in runs)
    header = (
        "| Condition | "
        + " | ".join(["Start"] + [f"Cycle {i}" for i in range(1, cycles + 1)])
        + " |"
    )
    lines = [header, "|---|" + "---|" * (cycles + 1)]
    for condition, runs in groups.items():
        cells = [_fmt(v).split(" (")[0] for v in validation_series(runs)]
        lines.append(f"| {condition} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def training_table(groups: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "| Condition | Cycle | Solved during training | Learned from partner (per agent) | "
        "Training pairs (per agent) | Pairs from partner | Code divergence | "
        "Outcome disagreement |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for condition, runs in groups.items():
        cycles = max(len(r["cycles"]) for r in runs)
        for i in range(cycles):
            solved, peer, pairs, peer_share, div, dis = [], [], [], [], [], []
            for run in runs:
                if i >= len(run["cycles"]):
                    continue
                c = run["cycles"][i]
                for a in run["agents"]:
                    solved.append(c["solved_rate"][a])
                    peer.append(c["learned_from_peer"].get(a, 0))
                    total = sum(c["preference_pairs"].get(a, {}).values())
                    pairs.append(total)
                    from_peer = sum(
                        n
                        for src, n in c["pair_sources"].get(a, {}).items()
                        if src != "self"
                    )
                    if total:
                        peer_share.append(from_peer / total)
                if c["code_divergence"] is not None:
                    div.append(c["code_divergence"])
                    dis.append(c["outcome_disagreement"])
            lines.append(
                f"| {condition} | {i + 1} | {_fmt(solved).split(' (')[0]}% | "
                f"{_fmt(peer, scale=1).split(' (')[0]} | {_fmt(pairs, scale=1).split(' (')[0]} | "
                f"{_fmt(peer_share).split(' (')[0] + '%' if peer_share else '–'} | "
                f"{_fmt(div).split(' (')[0] + '%' if div else '–'} | "
                f"{_fmt(dis).split(' (')[0] + '%' if dis else '–'} |"
            )
    return "\n".join(lines)


def coordinator_stats(run: dict[str, Any]) -> list[dict[str, Any]]:
    """Per cycle: choice shares, and how often the chosen proposal was right
    compared with the average proposal (only where correctness is known)."""
    stats = []
    for cycle in range(1, len(run["cycles"]) + 1):
        path = run["dir"] / "rounds" / f"cycle_{cycle}.jsonl"
        choices: Counter[str] = Counter()
        chosen_right, chosen_n, any_right, any_n = 0, 0, 0, 0
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                for agent, result in r["results"].items():
                    first = result["decisions"][0]
                    choices[str(first["choice"])] += 1
                    agreement = r["agreement"].get(agent, {})
                    known = [
                        agreement[f"1:{p['specialist']}"]
                        for p in first["proposals"]
                        if f"1:{p['specialist']}" in agreement
                    ]
                    if not known:
                        continue
                    any_right += sum(known)
                    any_n += len(known)
                    key = f"1:{first['choice']}"
                    if key in agreement:
                        chosen_right += agreement[key]
                        chosen_n += 1
        total = sum(choices.values())
        stats.append(
            {
                "cycle": cycle,
                "shares": {k: v / total for k, v in choices.items()},
                "chosen_accuracy": chosen_right / chosen_n if chosen_n else None,
                "proposal_accuracy": any_right / any_n if any_n else None,
            }
        )
    return stats


def coordinator_table(groups: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "| Condition | Cycle | Most chosen | Combined | Chosen proposal right | "
        "Average proposal right |",
        "|---|---|---|---|---|---|",
    ]
    for condition, runs in groups.items():
        per_run = [coordinator_stats(r) for r in runs]
        for i in range(max(len(s) for s in per_run)):
            shares: Counter[str] = Counter()
            chosen, base = [], []
            for stats in per_run:
                if i < len(stats):
                    shares.update(stats[i]["shares"])
                    if stats[i]["chosen_accuracy"] is not None:
                        chosen.append(stats[i]["chosen_accuracy"])
                        base.append(stats[i]["proposal_accuracy"])
            norm = sum(shares.values())
            top = max(
                ((k, v) for k, v in shares.items() if k not in ("combined", "None")),
                key=lambda kv: kv[1],
            )
            lines.append(
                f"| {condition} | {i + 1} | {top[0]} ({100 * top[1] / norm:.0f}%) | "
                f"{100 * shares.get('combined', 0) / norm:.0f}% | "
                f"{_fmt(chosen).split(' (')[0]}% | {_fmt(base).split(' (')[0]}% |"
            )
    return "\n".join(lines)


def lessons_section(groups: dict[str, list[dict[str, Any]]]) -> str:
    parts = []
    for condition, runs in groups.items():
        run = runs[0]
        last = run["cycles"][-1]
        for agent, lessons in last["lessons"].items():
            parts.append(f"**{run['name']} · agent {agent} · cycle {last['cycle']}**")
            parts.extend(f"- {lesson}" for lesson in lessons)
            parts.append("")
    return "\n".join(parts)


# ── figures ─────────────────────────────────────────────────────────────────


def _style(ax: Any) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#ecebe7", linewidth=0.8)
    ax.set_axisbelow(True)


def figures(groups: dict[str, list[dict[str, Any]]], out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for condition, runs in groups.items():
        series = validation_series(runs)
        xs = list(range(len(series)))
        means = [100 * statistics.mean(v) for v in series]
        sds = [100 * (statistics.stdev(v) if len(v) > 1 else 0) for v in series]
        ax.plot(
            xs,
            means,
            color=COLORS[condition],
            linewidth=2,
            marker="o",
            markersize=5,
            label=f"{condition} (n={len(series[0])} agents)",
        )
        ax.fill_between(
            xs,
            [m - s for m, s in zip(means, sds)],
            [m + s for m, s in zip(means, sds)],
            color=COLORS[condition],
            alpha=0.12,
            linewidth=0,
        )
    ax.set_xlabel("Cycle (0 = untrained)")
    ax.set_ylabel("Validation Pass@1 (%)")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "validation.png", dpi=200)
    plt.close(fig)
    written.append("validation.png")

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    labels = list(BENCH_NAMES.values())
    width = 0.36
    for j, (condition, runs) in enumerate(groups.items()):
        for i, ((bench, kind), _) in enumerate(BENCH_NAMES.items()):
            deltas = [
                100 * (e["scores"][bench][kind] - b["scores"][bench][kind])
                for run in runs
                for b, e in zip(run["baseline_eval"], run["final_eval"])
                if bench in b["scores"]
            ]
            if not deltas:
                continue
            x = i + (j - 0.5) * width
            ax.bar(
                x,
                statistics.mean(deltas),
                width * 0.9,
                color=COLORS[condition],
                label=condition if i == 0 else None,
            )
            ax.scatter([x] * len(deltas), deltas, s=12, color="#52514e", zorder=3)
    ax.axhline(0, color="#52514e", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Change, start → end (pp)")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "evalplus_change.png", dpi=200)
    plt.close(fig)
    written.append("evalplus_change.png")

    if "pair" in groups:
        runs = groups["pair"]
        fig, ax = plt.subplots(figsize=(6, 3.6))
        cycles = max(len(r["cycles"]) for r in runs)
        xs = list(range(1, cycles + 1))
        for key, label, color in (
            ("code_divergence", "Different code", "#2a78d6"),
            (
                "outcome_disagreement",
                "Different verdict (one right, one wrong)",
                "#eb6834",
            ),
        ):
            ys = [
                100
                * statistics.mean(
                    r["cycles"][i][key] for r in runs if i < len(r["cycles"])
                )
                for i in range(cycles)
            ]
            ax.plot(
                xs, ys, color=color, linewidth=2, marker="o", markersize=5, label=label
            )
        ax.set_xlabel("Cycle")
        ax.set_ylabel("% of validation problems")
        ax.legend(frameon=False)
        _style(ax)
        fig.tight_layout()
        fig.savefig(out / "divergence.png", dpi=200)
        plt.close(fig)
        written.append("divergence.png")

    fig, ax = plt.subplots(figsize=(6, 3.6))
    all_runs = [r for runs in groups.values() for r in runs]
    per_run = [coordinator_stats(r) for r in all_runs]
    cycles = max(len(s) for s in per_run)
    for choice, color in CHOICE_COLORS.items():
        ys = []
        for i in range(cycles):
            vals = [s[i]["shares"].get(choice, 0.0) for s in per_run if i < len(s)]
            ys.append(100 * statistics.mean(vals))
        ax.plot(
            range(1, cycles + 1),
            ys,
            color=color,
            linewidth=2,
            marker="o",
            markersize=5,
            label=choice,
        )
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Coordinator decisions (%)")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "coordinator_choices.png", dpi=200)
    plt.close(fig)
    written.append("coordinator_choices.png")
    return written


def analyze(runs_root: Path, out: Path) -> Path:
    runs = load_runs(runs_root)
    if not runs:
        raise SystemExit(f"No finished runs under {runs_root}")
    groups = by_condition(runs)
    out.mkdir(parents=True, exist_ok=True)
    written = figures(groups, out)
    summary = "\n\n".join(
        [
            "# Results",
            "Runs: " + ", ".join(r["name"] for r in runs),
            "Values are mean ± SD across agents (pair: 2 agents × replicates; "
            "solo: 1 agent × replicates).",
            "## EvalPlus Pass@1 (%)",
            evalplus_table(groups),
            "## Validation Pass@1 (%) by cycle",
            validation_table(groups),
            "## Learning during training",
            training_table(groups),
            "## Coordinator",
            coordinator_table(groups),
            "## Example sleep lessons (last cycle)",
            lessons_section(groups),
            "## Figures",
            "\n".join(f"![{w}]({w})" for w in written),
        ]
    )
    path = out / "summary.md"
    path.write_text(summary + "\n")
    return path
