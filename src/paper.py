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
    "Qwen2.5-1.5B": "#c9c7bf",
    "majority": "#9a9890",
    "Brain, untrained": "#a9c8ee",
    "Brain, trained alone": COLORS["solo"],
    "Brain, trained in pair": COLORS["pair"],
}
SHORT = {
    "Qwen2.5-1.5B": "1.5B",
    "Brain, untrained": "Brain, untrained",
    "Brain, trained alone": "Brain, trained alone",
    "Brain, trained in pair": "Brain, trained in pair",
    "Qwen2.5-3B": "3B (single)",
    "Qwen2.5-7B": "7B (single)",
}


def _short(label: str) -> str:
    if ", majority of " in label:
        return f"1.5B, vote of {label.rsplit(' ', 1)[1]}"
    return SHORT.get(label, label)


DOMAIN_GROUPS = {
    "Code": ["humaneval", "mbpp"],
    "Math": ["gsm8k", "math500"],
    "Reasoning": ["arc_challenge"],
    "Social & emotion": ["siqa", "emobench_eu", "emobench_ea"],
}


def _group_mean(values: dict[str, list[float]], benches: list[str]) -> list[float]:
    """Per-agent (or single) means over a group of benchmarks, in percent."""
    present = [values[b] for b in benches if b in values]
    if len(present) != len(benches):
        return []
    n = min(len(v) for v in present)
    return [100 * statistics.mean(v[i] for v in present) for i in range(n)]


def fig_headline(
    rows: list[tuple[str, int, dict[str, list[float]]]], benches: list[str], out: Path
) -> str | None:
    """(a) Average accuracy of every system, larger models as labelled dashed
    lines. (b) Gain over the plain 1.5B model per domain."""
    small = [r for r in rows if r[0] not in REFERENCE_STYLE]
    big = [r for r in rows if r[0] in REFERENCE_STYLE]
    if not small:
        return None
    plt = _plt()
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(FULL_WIDTH, 2.3), gridspec_kw={"width_ratios": [1, 1.7]}
    )

    # (a) average over all benchmarks
    for i, (label, _, values) in enumerate(small):
        vals = _group_mean(values, benches)
        if not vals:
            continue
        m = statistics.mean(vals)
        ax_a.bar(
            i,
            m,
            0.72,
            color=BAR_COLORS.get(label, BAR_COLORS["majority"]),
            yerr=_ci95(vals) if len(vals) > 1 else None,
            error_kw={"elinewidth": 0.6, "capsize": 1.5},
            label=_short(label),
        )
        ax_a.text(
            i, m - 2.5, f"{m:.1f}", ha="center", va="top", fontsize=6, color="#2b2a28"
        )
    for label, _, values in big:
        vals = _group_mean(values, benches)
        if vals:
            ax_a.axhline(
                vals[0],
                color="#3a3936",
                linewidth=0.8,
                linestyle=REFERENCE_STYLE[label],
            )
            ax_a.text(
                len(small) - 0.45,
                vals[0] + 0.8,
                _short(label),
                ha="right",
                va="bottom",
                fontsize=6,
                color="#3a3936",
            )
    ax_a.set_xticks([])
    ax_a.set_ylim(0, 100)
    ax_a.set_ylabel("Accuracy (%)")
    ax_a.set_xlim(-0.6, len(small) - 0.4)
    ax_a.set_title("a  Average over 8 benchmarks", loc="left")

    # (b) gain over the plain 1.5B model, per domain
    base = next((r for r in rows if r[0] == "Qwen2.5-1.5B"), None)
    if base is not None:
        others = [r for r in rows if r[0] != "Qwen2.5-1.5B"]
        groups = list(DOMAIN_GROUPS)
        width = 0.8 / max(1, len(others))
        for k, (label, _, values) in enumerate(others):
            gains, errs, multi = [], [], False
            for g in groups:
                vals = _group_mean(values, DOMAIN_GROUPS[g])
                ref = _group_mean(base[2], DOMAIN_GROUPS[g])
                gains.append(statistics.mean(vals) - ref[0] if vals and ref else 0.0)
                errs.append(_ci95(vals) if len(vals) > 1 else 0.0)
                multi = multi or len(vals) > 1
            xs = [i + (k - (len(others) - 1) / 2) * width for i in range(len(groups))]
            is_big = label in REFERENCE_STYLE
            ax_b.bar(
                xs,
                gains,
                width * 0.9,
                color=(
                    "white" if is_big else BAR_COLORS.get(label, BAR_COLORS["majority"])
                ),
                edgecolor="#3a3936" if is_big else "none",
                hatch="////" if label == "Qwen2.5-7B" else ("...." if is_big else None),
                linewidth=0.5,
                yerr=errs if multi else None,
                error_kw={"elinewidth": 0.6, "capsize": 1.2},
                label=_short(label) if is_big else None,
            )
        ax_b.axhline(0, color="#3a3936", linewidth=0.6)
        ax_b.set_xticks(range(len(groups)), groups)
        ax_b.set_ylabel("Gain over Qwen2.5-1.5B (pp)")
        ax_b.set_title("b  Gain by domain", loc="left")

    handles, labels = ax_a.get_legend_handles_labels()
    h2, l2 = ax_b.get_legend_handles_labels()
    fig.legend(
        handles + h2,
        labels + l2,
        ncol=len(labels + l2),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        columnspacing=1.2,
        handlelength=1.4,
    )
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
        ax.set_title(d.capitalize())
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
        ax.text(
            len(domains) - 0.5,
            25.6,
            "uniform",
            ha="right",
            va="bottom",
            fontsize=5.8,
            color="#8a8984",
        )
        ax.set_ylabel("Share of training (%)")
        ax.set_ylim(0, max(50.0, ax.get_ylim()[1]))
        ax.set_title(label, loc="left")
        _style(ax)
    axes[-1].set_xticks(range(len(domains)), [d.capitalize() for d in domains])
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles, labels, ncol=4, loc="upper left", handlelength=1.0, columnspacing=0.9
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


DIAGRAM_INK, DIAGRAM_LINE = "#2b2a28", "#5c5b57"
DIAGRAM_FILLS = {
    "spec": "#e8f0fb",
    "coord": "#efeaf8",
    "world": "#fbece4",
    "plain": "#f4f3f0",
}


def _diagram(ax: Any) -> tuple[Any, Any, Any]:
    """Box, dashed-panel and right-angle-arrow helpers shared by the diagrams."""
    from matplotlib.patches import FancyBboxPatch, Rectangle

    ink, line = DIAGRAM_INK, DIAGRAM_LINE

    def box(
        cx: float,
        cy: float,
        w: float,
        h: float,
        text: str,
        fill: str,
        bold: bool = False,
        size: float = 6.3,
    ) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (cx - w / 2, cy - h / 2),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=0.9",
                linewidth=0.7,
                edgecolor=line,
                facecolor=fill,
            )
        )
        ax.text(
            cx,
            cy,
            text,
            ha="center",
            va="center",
            fontsize=size,
            color=ink,
            fontweight="bold" if bold else "normal",
            linespacing=1.15,
        )

    def panel(x: float, y: float, w: float, h: float, title: str) -> None:
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                fill=False,
                linestyle=(0, (3, 2)),
                linewidth=0.7,
                edgecolor="#8a8984",
            )
        )
        ax.text(
            x + w / 2,
            y + h + 1.0,
            title,
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
            color=ink,
        )

    def arrow(
        points: list[tuple[float, float]],
        label: str = "",
        at: int = 0,
        dx: float = 0.0,
        dy: float = 1.0,
    ) -> None:
        xs, ys = zip(*points)
        ax.plot(
            xs[:-1] + (xs[-1],),
            ys[:-1] + (ys[-1],),
            color=line,
            linewidth=0.7,
            solid_capstyle="butt",
        )
        ax.annotate(
            "",
            xy=points[-1],
            xytext=points[-2],
            arrowprops={
                "arrowstyle": "-|>",
                "color": line,
                "lw": 0.7,
                "mutation_scale": 6.5,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )
        if label:
            (x1, y1), (x2, y2) = points[at], points[at + 1]
            ax.text(
                (x1 + x2) / 2 + dx,
                (y1 + y2) / 2 + dy,
                label,
                ha="center",
                va="bottom",
                fontsize=5.8,
                color="#5c5b57",
                style="italic",
            )

    return box, panel, arrow


def fig_method(out: Path) -> str:
    """Method overview in the style of Multiagent Finetuning (Fig. 2): grouped
    panels with dashed outlines, equal boxes, right-angle arrows, few colours."""
    plt = _plt()
    fills = DIAGRAM_FILLS
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 47)
    ax.axis("off")
    ax.grid(False)

    box, panel, arrow = _diagram(ax)

    names = ["Logical", "Creative", "Skeptical", "Empathetic"]
    cols = [12.0, 24.5, 37.0, 49.5]
    coord_x = 30.75
    arrow_r = "$\\rightarrow$"

    # ── waking: one agent on one task ──
    panel(2, 3, 58, 40, "Waking: one agent works on a task")
    box(
        coord_x,
        38.0,
        48.5,
        4.0,
        "Task:  code  ·  math  ·  reasoning  ·  social",
        fills["plain"],
    )
    for x, n in zip(cols, names):
        box(x, 29.0, 10.5, 4.8, f"{n}\nround 1: alone", fills["spec"], size=6.0)
        box(x, 19.5, 10.5, 4.8, f"{n}\nround 2: discuss", fills["spec"], size=6.0)
        arrow([(x, 36.0), (x, 31.4)])
        arrow([(x, 26.6), (x, 21.9)])
    box(
        coord_x,
        9.0,
        22,
        4.8,
        "Coordinator\nweighs the arguments, decides",
        fills["coord"],
    )
    for x in cols:
        tx = coord_x + (x - coord_x) * 0.38
        arrow([(x, 17.1), (x, 14.0), (tx, 14.0), (tx, 11.4)])
    box(11.0, 9.0, 11.5, 4.8, "Memory\nepisodes, lessons", fills["plain"], size=5.8)
    arrow([(16.75, 9.0), (coord_x - 11, 9.0)])

    # ── environment and partner ──
    box(71, 9.0, 13, 4.8, 'World\nsays only "wrong"', fills["world"])
    arrow([(coord_x + 11, 9.0), (64.5, 9.0)])
    arrow(
        [(71, 6.6), (71, 4.6), (4.2, 4.6), (4.2, 29.0), (6.75, 29.0)],
        f"wrong {arrow_r} retry",
        at=1,
        dx=8,
        dy=0.35,
    )
    box(71, 23.0, 13, 4.8, "Partner agent\n(same design)", fills["plain"])
    arrow([(71, 11.4), (71, 20.6)], f"accepted {arrow_r}\nshared", dx=-5.2, dy=-2.2)
    arrow([(60, 38.0), (81, 38.0)], "end of cycle\n(100 tasks)", dy=0.5)

    # ── sleep ──
    panel(81, 3, 17, 40, "Sleep: after each cycle")
    steps = [
        "Reflect\nedit lessons",
        "Credit\nsolo winner, else\ndomain owner",
        "Train\nDPO + NLL on each\npart's own LoRA",
        "Validate\nroll back if worse",
    ]
    ys = [36.0, 26.5, 17.0, 7.5]
    for y, t in zip(ys, steps):
        box(89.5, y, 14.5, 6.4, t, fills["plain"], size=5.9)
    for y1, y2 in zip(ys[:-1], ys[1:]):
        arrow([(89.5, y1 - 3.2), (89.5, y2 + 3.2)])
    fig.tight_layout()
    return _save(fig, out, "method")


def fig_memory(out: Path) -> str:
    """The agent's three memories: what is recalled before an answer, what is
    written after it, and how sleep edits the lessons."""
    plt = _plt()
    fills = DIAGRAM_FILLS
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.35))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")
    ax.grid(False)
    box, _, arrow = _diagram(ax)

    for x, title in [(13, "Before answering"), (48, "Memory"), (85, "After answering")]:
        ax.text(
            x,
            38.2,
            title,
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
            color=DIAGRAM_INK,
        )

    # the three stores
    stores = [
        (
            31.0,
            "Working memory  (short-term)\nlast 10 events · cleared every cycle",
            fills["spec"],
        ),
        (
            19.5,
            "Episodic memory  (hippocampus)\nevery past task · 3 most similar recalled",
            fills["coord"],
        ),
        (
            8.0,
            "Semantic memory  (cortex)\nlessons · 5 most trusted recalled",
            fills["world"],
        ),
    ]
    for y, text, fill in stores:
        box(48, y, 27, 7.4, text, fill, size=6.0)

    # recall: task + stores -> prompt
    box(13, 31.0, 19, 6.0, "New task", fills["plain"])
    box(
        13,
        8.0,
        19,
        7.4,
        "Prompt for each\nspecialist and the\ncoordinator",
        fills["plain"],
        size=6.0,
    )
    arrow([(13, 28.0), (13, 11.7)], "task text", dx=-4.6, dy=-1.2)
    for y, _, _ in stores:
        arrow([(34.5, y), (28.5, y), (28.5, 8.0), (22.5, 8.0)])
    ax.text(
        28.0,
        24.6,
        "recalled",
        ha="right",
        va="center",
        fontsize=5.8,
        color=DIAGRAM_LINE,
        style="italic",
    )

    # writing: after each task, and in sleep
    box(
        85,
        31.0,
        22,
        6.0,
        "After each task\nrecord what happened",
        fills["plain"],
        size=6.0,
    )
    arrow([(74, 31.0), (61.5, 31.0)], "a one-line note", dy=0.5)
    arrow(
        [(68, 31.0), (68, 21.0), (61.5, 21.0)],
        "the full\nepisode",
        at=0,
        dx=4.2,
        dy=-2.4,
    )
    box(
        85,
        8.0,
        22,
        7.4,
        "Sleep: reflect\ncompare wrong and accepted\nanswers; add, agree,\ndisagree or edit lessons",
        fills["plain"],
        size=5.7,
    )
    arrow(
        [(61.5, 18.0), (78, 18.0), (78, 11.7)],
        "this cycle's episodes",
        at=0,
        dx=-1.0,
        dy=-2.6,
    )
    arrow([(74, 8.0), (61.5, 8.0)], "edited lessons", dy=0.5)

    fig.tight_layout()
    return _save(fig, out, "memory")


def fig_sleep(out: Path) -> str:
    """One frozen model with five small adapters, and what sleep does to them."""
    plt = _plt()
    fills = DIAGRAM_FILLS
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.25))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 38)
    ax.axis("off")
    ax.grid(False)
    box, panel, arrow = _diagram(ax)

    # ── left: one model, five adapters ──
    panel(1, 2, 44, 31, "One brain: a frozen model with five adapters")
    box(23, 12.0, 39, 6.0, "Qwen2.5-1.5B  (frozen, shared)", fills["plain"], size=6.6)
    parts = [
        ("Logical", "spec"),
        ("Creative", "spec"),
        ("Skeptical", "spec"),
        ("Empathetic", "spec"),
        ("Coordinator", "coord"),
    ]
    xs = [7.0, 15.0, 23.0, 31.0, 39.0]
    for x, (name, fill) in zip(xs, parts):
        box(x, 25.5, 7.6, 8.0, f"{name}\nLoRA", fills[fill], size=5.6)
        arrow([(x, 21.5), (x, 15.0)])
    ax.text(
        23,
        5.6,
        "only the small adapters change",
        ha="center",
        va="center",
        fontsize=5.8,
        color=DIAGRAM_LINE,
        style="italic",
    )

    # ── right: sleep ──
    panel(49, 2, 50, 31, "Sleep: after every 100 tasks")
    steps = [
        (58.5, "Reflect\nedit lessons\n(Sec. 3.3)"),
        (69.5, "Pick who learns\nfive rules\n(Sec. 3.4)"),
        (80.5, "Train each part\nDPO + NLL on its\nown adapter"),
        (91.5, "Validate\nkeep, or\nroll back"),
    ]
    for x, text in steps:
        box(x, 22.0, 9.8, 10.0, text, fills["plain"], size=5.6)
    for (x1, _), (x2, _) in zip(steps[:-1], steps[1:]):
        arrow([(x1 + 4.9, 22.0), (x2 - 4.9, 22.0)])
    ax.text(
        80.5,
        11.0,
        "reference = the part's previous adapter,\nso each step stays small",
        ha="center",
        va="center",
        fontsize=5.6,
        color=DIAGRAM_LINE,
        style="italic",
    )
    arrow([(80.5, 17.0), (80.5, 13.4)])
    arrow(
        [(91.5, 17.0), (91.5, 4.6), (55.0, 4.6), (55.0, 17.0)],
        "next cycle with the updated parts",
        at=1,
        dx=0,
        dy=0.4,
    )

    fig.tight_layout()
    return _save(fig, out, "sleep")
