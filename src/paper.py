"""The paper's main table and figures, in the style of the closest related work.

  main_table          systems × benchmarks, best bold / second underlined
                      (Multiagent Finetuning, Table 1)
  fig_headline        1.5B-based systems as bars, larger single models as
                      dashed lines (More Agents Is All You Need, Fig. 1)
  fig_learning_curves pair solid vs solo dashed with 95% bands per domain
                      (Multiagent Finetuning, Fig. 1)
  fig_specialisation  share of each domain's training won by each specialist
                      vs the uniform line, version 1 vs version 2 (OLMoE, Fig. 22)
  cycle_table         accuracy per cycle (Iterative RPO, Table 1)
  fig_method          method overview diagram (Multiagent Finetuning, Fig. 2)

Every system in the main table is scored on exactly the same problems: the
fixed benchmark subsets the brain was evaluated on (EvalPlus code benchmarks in
full), recomputed for reference systems from their per-problem outputs.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from src.figures import (
    COLORS,
    FULL_WIDTH,
    ONE_COLUMN,
    SPECIALIST_COLORS,
    SPECIALISTS,
    _ci95,
    _plt,
    _save,
    _style,
)

CODE_BENCHMARKS = {"humaneval", "mbpp"}
BENCH_LABELS = {
    "humaneval": "HumanEval+",
    "mbpp": "MBPP+",
    "gsm8k": "GSM8K",
    "math500": "MATH-500",
    "arc_challenge": "ARC-C",
    "siqa": "SocialIQA",
    "emobench_eu": "EmoBench-U",
    "emobench_ea": "EmoBench-A",
}
REFERENCE_STYLE = {"Qwen2.5-3B": ":", "Qwen2.5-7B": "--"}


# ── scores on identical problem sets ────────────────────────────────────────


def _metric(score: dict[str, float]) -> str:
    return "plus" if "plus" in score else "accuracy"


def _subset_ids(baseline: dict[str, Any], bench: str) -> set[str] | None:
    """Problems the brain was benchmarked on (from the untrained brain's outputs)."""
    path = (
        Path(baseline["dir"])
        / "benchmarks"
        / "baseline"
        / "agent"
        / "agent a"
        / f"{bench}.jsonl"
    )
    if bench in CODE_BENCHMARKS or not path.exists():
        return None
    return {json.loads(line)["task_id"] for line in path.open()}


def _accuracy_on(path: Path, ids: set[str]) -> float | None:
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.open()]
    hits = [r["correct"] for r in rows if r["task_id"] in ids]
    return sum(hits) / len(hits) if hits else None


def _single_system(
    entry: dict[str, Any], jsonl_dir: Path, bench: str, ids: set[str] | None
) -> float | None:
    if bench not in entry["scores"]:
        return None
    if ids is None:
        score = entry["scores"][bench]
        return float(score[_metric(score)])
    return _accuracy_on(jsonl_dir / f"{bench}.jsonl", ids)


def systems(
    baseline: dict[str, Any] | None,
    references: dict[str, Any] | None,
    groups: dict[str, list[dict[str, Any]]],
) -> list[tuple[str, int, dict[str, list[float]]]]:
    """(label, model calls per problem, benchmark -> values) for every system."""
    if baseline is None:
        return []
    benches = list(baseline["config"]["evaluation"]["benchmarks"])
    ids = {b: _subset_ids(baseline, b) for b in benches}
    rows: list[tuple[str, int, dict[str, list[float]]]] = []

    def single(
        label: str, calls: int, entry: dict[str, Any] | None, jsonl_dir: Path
    ) -> None:
        if entry is None:
            return
        values = {}
        for b in benches:
            v = _single_system(entry, jsonl_dir, b, ids[b])
            if v is not None:
                values[b] = [v]
        rows.append((label, calls, values))

    base_dir = Path(baseline["dir"]) / "benchmarks" / "baseline"
    plain = next(
        (e for e in baseline["baseline_benchmarks"] if e["mode"] == "plain"), None
    )
    single("Qwen2.5-1.5B", 1, plain, base_dir / "plain" / "base")
    ref_entries = references["baseline_benchmarks"] if references else []
    ref_dir = Path(references["dir"]) / "benchmarks" if references else Path()
    for e in ref_entries:
        if e["mode"].startswith("majority"):
            n = int(e["mode"].split("@")[1])
            single(
                f"{e['system']}, majority of {n}",
                n,
                e,
                ref_dir / f"{e['system']}_{e['mode']}",
            )
    brain = next(
        (e for e in baseline["baseline_benchmarks"] if e["mode"] == "agent"), None
    )
    single("Brain, untrained", 9, brain, base_dir / "agent" / "agent a")
    for condition, label in (
        ("solo", "Brain, trained alone"),
        ("pair", "Brain, trained in pair"),
    ):
        values: dict[str, list[float]] = {}
        for run in groups.get(condition, []):
            for e in run["final_benchmarks"]:
                if e["mode"] != "agent":
                    continue
                for b, score in e["scores"].items():
                    values.setdefault(b, []).append(float(score[_metric(score)]))
        if values:
            rows.append((label, 9, values))
    for e in sorted(ref_entries, key=lambda e: e["system"]):
        if e["mode"] == "plain":
            single(e["system"], 1, e, ref_dir / f"{e['system']}_{e['mode']}")
    return rows


# ── main table ──────────────────────────────────────────────────────────────


def main_table(
    rows: list[tuple[str, int, dict[str, list[float]]]], benches: list[str]
) -> tuple[str, str]:
    """Markdown and LaTeX versions; best value per column bold, second underlined."""
    cols = [*benches, "average"]

    def mean_of(values: dict[str, list[float]], b: str) -> float | None:
        if b == "average":
            means = [statistics.mean(values[x]) for x in benches if x in values]
            return statistics.mean(means) if len(means) == len(benches) else None
        return statistics.mean(values[b]) if b in values else None

    ranks: dict[str, list[float]] = {}
    for b in cols:
        vals = sorted(
            {round(100 * v, 1) for _, _, r in rows if (v := mean_of(r, b)) is not None},
            reverse=True,
        )
        ranks[b] = vals[:2]

    def cell(values: dict[str, list[float]], b: str, latex: bool) -> str:
        m = mean_of(values, b)
        if m is None:
            return "–"
        text = f"{100 * m:.1f}"
        if b != "average" and len(values.get(b, [])) > 1:
            sd = 100 * statistics.stdev(values[b])
            text += f" ± {sd:.1f}" if not latex else rf"\,{{\scriptsize$\pm${sd:.1f}}}"
        r = round(100 * m, 1)
        if ranks[b] and r == ranks[b][0]:
            return rf"\textbf{{{text}}}" if latex else f"**{text}**"
        if len(ranks[b]) > 1 and r == ranks[b][1]:
            return rf"\underline{{{text}}}" if latex else f"<u>{text}</u>"
        return text

    head = ["System", "Calls", *[BENCH_LABELS.get(b, b) for b in benches], "Avg."]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    tex_rows = []
    for label, calls, values in rows:
        md.append(
            f"| {label} | {calls} | "
            + " | ".join(cell(values, b, False) for b in cols)
            + " |"
        )
        tex_rows.append(
            f"{label} & {calls} & "
            + " & ".join(cell(values, b, True) for b in cols)
            + r" \\"
        )
    tex = (
        "\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{l"
        + "c" * (len(head) - 1)
        + "}\n\\toprule\n"
        + " & ".join(head)
        + r" \\"
        + "\n\\midrule\n"
        + "\n".join(tex_rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{Accuracy (\\%) on held-out benchmarks. All systems are scored on the same "
        "problems (code: EvalPlus base + plus tests). Trained rows: mean $\\pm$ SD over agents. "
        "Best in bold, second underlined. Calls: model calls per problem.}\n"
        "\\label{tab:main}\n\\end{table*}\n"
    )
    return "\n".join(md), tex


# ── figures ─────────────────────────────────────────────────────────────────

BAR_COLORS = {
    "Qwen2.5-1.5B": "#b8b6ae",
    "majority": "#8a8984",
    "Brain, untrained": "#9bc1ec",
    "Brain, trained alone": COLORS["solo"],
    "Brain, trained in pair": COLORS["pair"],
}


def fig_headline(
    rows: list[tuple[str, int, dict[str, list[float]]]], benches: list[str], out: Path
) -> str | None:
    bars = [r for r in rows if r[1] > 1 or r[0] == "Qwen2.5-1.5B"]
    lines = [r for r in rows if r[0] in REFERENCE_STYLE]
    if not bars:
        return None
    plt = _plt()
    panels = [*benches, "average"]
    ncols = 5
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(FULL_WIDTH, 1.55 * nrows), sharey=False
    )
    axes = [ax for row in (axes if nrows > 1 else [axes]) for ax in row]

    def value(values: dict[str, list[float]], b: str) -> list[float]:
        if b != "average":
            return [100 * v for v in values.get(b, [])]
        means = [statistics.mean(values[x]) for x in benches if x in values]
        return [100 * statistics.mean(means)] if len(means) == len(benches) else []

    for ax, b in zip(axes, panels):
        for i, (label, _, values) in enumerate(bars):
            vals = value(values, b)
            if not vals:
                continue
            color = BAR_COLORS.get(label, BAR_COLORS["majority"])
            ax.bar(
                i,
                statistics.mean(vals),
                0.75,
                color=color,
                yerr=_ci95(vals) or None,
                error_kw={"elinewidth": 0.6, "capsize": 1.5},
                label=label,
            )
        for label, _, values in lines:
            vals = value(values, b)
            if vals:
                ax.axhline(
                    vals[0],
                    color="#3a3936",
                    linewidth=0.8,
                    linestyle=REFERENCE_STYLE[label],
                    label=f"{label} (single)",
                )
        ax.set_title(BENCH_LABELS.get(b, "Average"))
        ax.set_xticks([])
        _style(ax)
        lo = min(
            [statistics.mean(value(v, b)) for _, _, v in rows if value(v, b)] or [0]
        )
        ax.set_ylim(max(0, lo - 12), None)
    for ax in axes[len(panels) :]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
    )
    fig.supylabel("Accuracy (%)", fontsize=8)
    fig.tight_layout()
    return _save(fig, out, "headline")


def fig_learning_curves(groups: dict[str, list[dict[str, Any]]], out: Path) -> str:
    plt = _plt()
    domains = list(
        next(r for rs in groups.values() for r in rs)["config"]["data"]["domains"]
    )
    fig, axes = plt.subplots(
        1, len(domains) + 1, figsize=(FULL_WIDTH, 1.8), sharey=False
    )
    style = {"pair": "-", "solo": "--"}
    for ax, d in zip(axes, [*domains, "mean"]):
        for condition, runs in groups.items():
            n = max(len(r["cycles"]) for r in runs)
            means, cis = [], []
            for i in range(n + 1):
                vals = []
                for r in runs:
                    for a in r["agents"]:
                        if i == 0:
                            v = r["baseline_validation"][a]
                        elif i - 1 < len(r["cycles"]):
                            v = r["cycles"][i - 1]["validation"][a]
                        else:
                            continue
                        vals.append(
                            100 * (statistics.mean(v.values()) if d == "mean" else v[d])
                        )
                means.append(statistics.mean(vals))
                cis.append(_ci95(vals))
            xs = list(range(n + 1))
            ax.plot(
                xs,
                means,
                linestyle=style[condition],
                color=COLORS[condition],
                marker="o",
                label=(
                    "Brain, trained in pair"
                    if condition == "pair"
                    else "Brain, trained alone"
                ),
            )
            ax.fill_between(
                xs,
                [m - c for m, c in zip(means, cis)],
                [m + c for m, c in zip(means, cis)],
                color=COLORS[condition],
                alpha=0.15,
                linewidth=0,
            )
        ax.set_title(d)
        ax.set_xlabel("Cycle")
        _style(ax)
    axes[0].set_ylabel("Validation accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
    )
    fig.tight_layout()
    return _save(fig, out, "learning_curves")


def _training_shares(
    runs: list[dict[str, Any]], domains: list[str]
) -> dict[str, dict[str, float]]:
    counts = {(s, d): 0 for s in SPECIALISTS for d in domains}
    for r in runs:
        for c in r["cycles"]:
            for a in r["agents"]:
                for s in SPECIALISTS:
                    for d, n in c["pairs_by_domain"].get(a, {}).get(s, {}).items():
                        if (s, d) in counts:
                            counts[(s, d)] += n
    shares: dict[str, dict[str, float]] = {}
    for d in domains:
        total = sum(counts[(s, d)] for s in SPECIALISTS)
        shares[d] = {
            s: 100 * counts[(s, d)] / total if total else 0.0 for s in SPECIALISTS
        }
    return shares


def fig_specialisation(
    versions: dict[str, list[dict[str, Any]]], out: Path
) -> str | None:
    """versions: label (e.g. 'v1: no competition') -> runs (pair and solo pooled)."""
    versions = {k: v for k, v in versions.items() if v}
    if not versions:
        return None
    plt = _plt()
    domains = list(next(iter(versions.values()))[0]["config"]["data"]["domains"])
    fig, axes = plt.subplots(
        len(versions), 1, figsize=(ONE_COLUMN * 1.6, 1.6 * len(versions)), sharex=True
    )
    axes = axes if len(versions) > 1 else [axes]
    width = 0.2
    for ax, (label, runs) in zip(axes, versions.items()):
        shares = _training_shares(runs, domains)
        for k, s in enumerate(SPECIALISTS):
            xs = [i + (k - 1.5) * width for i in range(len(domains))]
            ax.bar(
                xs,
                [shares[d][s] for d in domains],
                width * 0.92,
                color=SPECIALIST_COLORS[s],
                label=s,
            )
        ax.axhline(25, color="#8a8984", linewidth=0.8, linestyle="--")
        ax.set_ylabel("Share (%)")
        ax.set_title(label)
        _style(ax)
    axes[-1].set_xticks(range(len(domains)), domains)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
    )
    fig.tight_layout()
    return _save(fig, out, "specialisation")


def cycle_table(groups: dict[str, list[dict[str, Any]]]) -> str:
    domains = list(
        next(r for rs in groups.values() for r in rs)["config"]["data"]["domains"]
    )
    lines = [
        "| System | Cycle | " + " | ".join(domains) + " | Mean |",
        "|---|---|" + "---|" * (len(domains) + 1),
    ]
    for condition, runs in groups.items():
        label = (
            "Brain, trained in pair" if condition == "pair" else "Brain, trained alone"
        )
        n = max(len(r["cycles"]) for r in runs)
        for i in range(n + 1):
            per: dict[str, list[float]] = {d: [] for d in [*domains, "mean"]}
            for r in runs:
                for a in r["agents"]:
                    v = (
                        r["baseline_validation"][a]
                        if i == 0
                        else (
                            r["cycles"][i - 1]["validation"][a]
                            if i - 1 < len(r["cycles"])
                            else None
                        )
                    )
                    if v is None:
                        continue
                    for d in domains:
                        per[d].append(100 * v[d])
                    per["mean"].append(100 * statistics.mean(v.values()))
            cells = [
                f"{statistics.mean(per[d]):.1f}" if per[d] else "–"
                for d in [*domains, "mean"]
            ]
            lines.append(
                f"| {label if i == 0 else ''} | {'untrained' if i == 0 else i} | "
                + " | ".join(cells)
                + " |"
            )
    return "\n".join(lines)


def fig_method(out: Path) -> str:
    """Method overview: one agent's deliberation, the world, sharing, and sleep."""
    plt = _plt()
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    def box(
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        color: str = "#f0efec",
        bold: bool = False,
    ) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.3,rounding_size=0.8",
                linewidth=0.6,
                edgecolor="#52514e",
                facecolor=color,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=6.5,
            fontweight="bold" if bold else "normal",
        )

    def arrow(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        text: str = "",
        rad: float = 0.0,
        tx: float | None = None,
        ty: float | None = None,
    ) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=7,
                linewidth=0.6,
                color="#52514e",
                connectionstyle=f"arc3,rad={rad}",
            )
        )
        if text:
            ax.text(
                tx if tx is not None else (x1 + x2) / 2,
                ty if ty is not None else (y1 + y2) / 2 + 1.2,
                text,
                ha="center",
                fontsize=5.5,
                color="#52514e",
            )

    ax.text(1, 37.5, "Agent (×2 in a pair)", fontsize=7, fontweight="bold")
    box(1, 18, 11, 6, "Task\n(code · math ·\nreasoning · social)")
    for i, s in enumerate(SPECIALISTS):
        box(
            16,
            31 - i * 7.5,
            13,
            5,
            f"{s}\nround 1 (alone)",
            SPECIALIST_COLORS[s] + "40",
        )
        box(
            33,
            31 - i * 7.5,
            13,
            5,
            f"{s}\nround 2 (discuss)",
            SPECIALIST_COLORS[s] + "40",
        )
        arrow(12, 21, 16, 33.5 - i * 7.5)
        arrow(29, 33.5 - i * 7.5, 33, 33.5 - i * 7.5)
        arrow(46, 33.5 - i * 7.5, 50, 20)
    ax.text(
        31,
        37.6,
        "each sees the others' round-1 answers",
        ha="center",
        fontsize=5.5,
        color="#52514e",
    )
    box(50, 17, 12, 6, "Coordinator\nweighs arguments,\ndecides", "#e6e3f7", bold=True)
    box(66, 17, 10, 6, 'World\nsays only\n"wrong"', "#fbe3d6")
    arrow(62, 20, 66, 20)
    arrow(71, 23, 71, 31, "accepted")
    box(64, 31, 15, 5, "Answer shared\nwith partner agent", "#dcebfa")
    arrow(
        71,
        17,
        22,
        8.5,
        'wrong → retry (previous answer + "wrong")',
        rad=-0.25,
        tx=47,
        ty=3.2,
    )
    box(
        81,
        17,
        18,
        15,
        "Sleep (each cycle)\n\n1 reflect → lessons\n2 who learns: solo winner\n   or domain owner\n3 DPO + NLL on each\n   specialist's own LoRA\n4 validate / roll back",
        "#f0efec",
    )
    arrow(79, 33.5, 88, 32.2)
    box(1, 2, 14, 7, "Memory\nworking · episodic ·\nlessons", "#f0efec")
    arrow(8, 9, 8, 18)
    fig.tight_layout()
    return _save(fig, out, "method")
