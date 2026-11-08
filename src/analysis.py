"""Aggregate finished runs into paper tables and figures.

    python -m cli.main analyze            # writes data/analysis/

Reads the `baseline` run (untrained system) and every finished run, groups runs
by condition (pair = two agents, solo = one agent), and reports mean ± SD
across agents and replicates.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _ms(values: list[float], scale: float = 100.0) -> str:
    if not values:
        return "–"
    vals = [v * scale for v in values]
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{statistics.mean(vals):.1f} ± {sd:.1f}"


def _signed(values: list[float]) -> str:
    if not values:
        return "–"
    text = _ms(values)
    return text if text.startswith("-") else "+" + text


def load(
    runs_root: Path,
) -> tuple[dict[str, Any] | None, dict[str, list[dict[str, Any]]]]:
    baseline = None
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(runs_root.glob("*/report.json")):
        data = json.loads(path.read_text())
        data["name"], data["dir"] = path.parent.name, path.parent
        if path.parent.name == "baseline":
            baseline = data
            continue
        if not data["final_benchmarks"]:
            continue
        agents = data["config"]["agents"]["names"]
        data["agents"] = agents
        groups["solo" if len(agents) == 1 else "pair"].append(data)
    return baseline, dict(sorted(groups.items()))


def _entries(
    reports: list[dict[str, Any]], mode: str, stage: str
) -> list[dict[str, Any]]:
    key = "final_benchmarks" if stage == "final" else "baseline_benchmarks"
    return [e for r in reports for e in r[key] if e["mode"] == mode]


def benchmark_table(baseline: dict[str, Any] | None, groups: dict[str, list]) -> str:
    """Rows: benchmark × metric. Columns: untrained and trained, per mode."""
    columns: list[tuple[str, list[dict[str, Any]]]] = []
    if baseline:
        for mode, label in (
            ("agent", "Untrained · whole agent"),
            ("plain", "Untrained · base model (plain)"),
        ):
            columns.append((label, _entries([baseline], mode, "baseline")))
    for condition, runs in groups.items():
        columns.append(
            (f"Trained {condition} · whole agent", _entries(runs, "agent", "final"))
        )
        columns.append(
            (
                f"Trained {condition} · fine-tuned specialists (plain)",
                _entries(runs, "plain", "final"),
            )
        )
    columns = [(label, e) for label, e in columns if e]
    if not columns:
        return "_No benchmark results yet._"
    benches = list(columns[0][1][0]["scores"])
    lines = [
        "| Benchmark | " + " | ".join(label for label, _ in columns) + " |",
        "|---|" + "---|" * len(columns),
    ]
    for bench in benches:
        metrics = list(columns[0][1][0]["scores"][bench])
        for metric in metrics:
            cells = [
                _ms(
                    [
                        e["scores"][bench][metric]
                        for e in entries
                        if bench in e["scores"]
                    ]
                )
                for _, entries in columns
            ]
            label = bench if metric == "accuracy" else f"{bench} ({metric})"
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def specialist_benchmark_table(
    baseline: dict[str, Any] | None, groups: dict[str, list]
) -> str:
    """Each specialist alone (own persona) on every benchmark: untrained vs trained."""
    columns = []
    if baseline:
        columns.append(("untrained", _entries([baseline], "specialist", "baseline")))
    for condition, runs in groups.items():
        columns.append((f"trained {condition}", _entries(runs, "specialist", "final")))
    columns = [(c, e) for c, e in columns if e]
    if not columns:
        return "_No specialist benchmark results yet._"
    benches = list(columns[0][1][0]["scores"])
    lines = [
        "| Specialist | Benchmark | " + " | ".join(c for c, _ in columns) + " |",
        "|---|---|" + "---|" * len(columns),
    ]
    specialists = sorted({e["system"].split("/")[1] for _, es in columns for e in es})
    for s in specialists:
        for bench in benches:
            metric = (
                "plus" if "plus" in columns[0][1][0]["scores"][bench] else "accuracy"
            )
            cells = [
                _ms(
                    [
                        e["scores"][bench][metric]
                        for e in es
                        if e["system"].endswith("/" + s)
                    ]
                )
                for _, es in columns
            ]
            lines.append(f"| {s} | {bench} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def specialisation_table(groups: dict[str, list]) -> str:
    """Validation accuracy of each specialist alone, per domain: start → end (change),
    and how often it ended as its domain's owner across agents."""
    lines = [
        "| Condition | Specialist | Domain | Start | End | Change | Owner (share of agents) |",
        "|---|---|---|---|---|---|---|",
    ]
    for condition, runs in groups.items():
        start: dict[tuple[str, str], list[float]] = defaultdict(list)
        end: dict[tuple[str, str], list[float]] = defaultdict(list)
        change: dict[tuple[str, str], list[float]] = defaultdict(list)
        owner: dict[tuple[str, str], int] = defaultdict(int)
        agents_total = 0
        for run in runs:
            last = run["cycles"][-1]
            for a in run["agents"]:
                agents_total += 1
                records = last["domain_records"].get(a, {})
                for s, by in run["baseline_validation_specialists"][a].items():
                    for d, v0 in by.items():
                        v1 = last["validation_specialists"][a][s].get(d)
                        start[(s, d)].append(v0)
                        if v1 is not None:
                            end[(s, d)].append(v1)
                            change[(s, d)].append(v1 - v0)
                domains = {d for by in records.values() for d in by}
                for d in domains:
                    best = max(
                        records, key=lambda name: float(records[name].get(d, 0.5))
                    )  # noqa: B023
                    owner[(best, d)] += 1
        for s, d in sorted(start):
            lines.append(
                f"| {condition} | {s} | {d} | {_ms(start[(s, d)])} | {_ms(end[(s, d)])} | "
                f"{_signed(change[(s, d)])} | {owner[(s, d)]}/{agents_total} |"
            )
    return "\n".join(lines)


def learning_table(groups: dict[str, list]) -> str:
    lines = [
        "| Condition | Cycle | Solved in training | Learned from partner | "
        "Discussion changed | Changed answers fixed | Answers differ | Verdicts differ |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for condition, runs in groups.items():
        for i in range(max(len(r["cycles"]) for r in runs)):
            solved, peer, changed, fixed, div, dis = [], [], [], [], [], []
            for run in runs:
                if i >= len(run["cycles"]):
                    continue
                c = run["cycles"][i]
                for a in run["agents"]:
                    rates = list(c["solved_rate"][a].values())
                    solved.append(sum(rates) / len(rates))
                    peer.append(c["learned_from_peer"].get(a, 0))
                    changed.append(c["discussion_changed"].get(a, 0.0))
                    fixed.append(c["discussion_fixed"].get(a, 0.0))
                if c["answer_divergence"] is not None:
                    div.append(c["answer_divergence"])
                    dis.append(c["outcome_disagreement"])
            lines.append(
                f"| {condition} | {i + 1} | {_ms(solved)} | {_ms(peer, 1)} | {_ms(changed)} | "
                f"{_ms(fixed)} | {_ms(div)} | {_ms(dis)} |"
            )
    return "\n".join(lines)


def validation_series(runs: list[dict[str, Any]]) -> list[list[float]]:
    """[cycle 0..N][agent mean-over-domains values]."""
    n = max(len(r["cycles"]) for r in runs)
    series: list[list[float]] = [[] for _ in range(n + 1)]
    for run in runs:
        for a in run["agents"]:
            v = run["baseline_validation"][a]
            series[0].append(sum(v.values()) / len(v))
            for i, c in enumerate(run["cycles"], start=1):
                v = c["validation"][a]
                series[i].append(sum(v.values()) / len(v))
    return series


def analyze(runs_root: Path, out: Path) -> Path:
    from src import figures as extra

    baseline, groups = load(runs_root)
    if not groups and baseline is None:
        raise SystemExit(f"No finished runs under {runs_root}")
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    comparison = "_Needs finished pair and solo runs._"
    if groups:
        written.append(extra.fig_validation_overall(groups, out))
        written.append(extra.fig_responsibility(groups, out))
        changes = extra.run_level_changes(groups, baseline)
        comparison = extra.comparison_table(changes)
        written.append(extra.fig_benchmark_change(changes, out))
        written.append(extra.fig_validation_by_domain(groups, out))
        written.append(extra.fig_specialisation_heatmaps(groups, out))
        written.append(extra.fig_dynamics(groups, out))
        for optional in (
            extra.fig_specialist_benchmarks(groups, baseline, out),
            extra.fig_training(groups, out),
        ):
            if optional:
                written.append(optional)
    names = [r["name"] for runs in groups.values() for r in runs]
    summary = "\n\n".join(
        [
            "# Results",
            "Runs: "
            + (", ".join(names) or "none yet")
            + (" · baseline: yes" if baseline else ""),
            "Values are mean ± SD across agents (pair: 2 agents × replicates; solo: 1 agent × "
            "replicates). Agent and specialist modes use a fixed subset per benchmark "
            "(EvalPlus code benchmarks in full); plain mode uses full benchmarks.",
            "## Benchmarks",
            benchmark_table(baseline, groups),
            "## Pair vs solo (run level: each run's agents averaged first)",
            "Change = trained − untrained, whole agent, percentage points. CI: bootstrap "
            "95%; p: exact two-sided permutation test over runs.",
            comparison,
            "## Each specialist alone on the benchmarks (own persona)",
            specialist_benchmark_table(baseline, groups),
            "## Specialisation during training (validation, specialist alone)",
            specialisation_table(groups) if groups else "_No runs yet._",
            "## Learning dynamics",
            learning_table(groups) if groups else "_No runs yet._",
            "## Figures",
            "\n".join(f"![{w}]({w})" for w in written),
        ]
    )
    path = out / "summary.md"
    path.write_text(summary + "\n")
    (out / "tables.tex").write_text(extra.markdown_tables_to_latex(summary))
    return path
