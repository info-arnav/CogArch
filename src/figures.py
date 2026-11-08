"""Paper figures and statistics beyond the basic tables in analysis.py.

Statistics are computed at the run level (each run's agents averaged first),
because the two agents of a pair run are not independent. With few runs per
condition, pair vs solo uses an exact permutation test and a bootstrap
confidence interval on the difference in means.
"""

from __future__ import annotations

import itertools
import random
import statistics
from pathlib import Path
from typing import Any

COLORS = {"pair": "#2a78d6", "solo": "#eb6834"}
SPECIALISTS = ["logical", "creative", "skeptical", "empathetic"]
SPECIALIST_COLORS = dict(zip(SPECIALISTS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))


def _domains(groups: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Training domains, in config order (same for every run)."""
    runs = [r for rs in groups.values() for r in rs]
    return list(runs[0]["config"]["data"]["domains"])


def _metric(score: dict[str, float]) -> str:
    return "plus" if "plus" in score else "accuracy"


def _entry(
    entries: list[dict[str, Any]], mode: str, system: str
) -> dict[str, Any] | None:
    return next(
        (e for e in entries if e["mode"] == mode and e["system"] == system), None
    )


# ── statistics ──────────────────────────────────────────────────────────────


def compare(a: list[float], b: list[float], resamples: int = 10000) -> dict[str, float]:
    """Difference in means (a − b), bootstrap 95% CI, exact two-sided permutation p."""
    diff = statistics.mean(a) - statistics.mean(b)
    rng = random.Random(0)
    boots = sorted(
        statistics.mean(rng.choices(a, k=len(a)))
        - statistics.mean(rng.choices(b, k=len(b)))
        for _ in range(resamples)
    )
    pooled, n = a + b, len(a)
    extreme = total = 0
    for idx in itertools.combinations(range(len(pooled)), n):
        chosen = set(idx)
        x = [pooled[i] for i in chosen]
        y = [pooled[i] for i in range(len(pooled)) if i not in chosen]
        total += 1
        extreme += abs(statistics.mean(x) - statistics.mean(y)) >= abs(diff) - 1e-12
    return {
        "diff": diff,
        "ci_low": boots[int(0.025 * resamples)],
        "ci_high": boots[int(0.975 * resamples) - 1],
        "p": extreme / total,
    }


def run_level_changes(
    groups: dict[str, list[dict[str, Any]]], baseline: dict[str, Any] | None
) -> dict[str, dict[str, list[float]]]:
    """benchmark -> condition -> per-run mean change of the whole agent
    (trained minus untrained), plus 'validation' (mean over domains)."""
    out: dict[str, dict[str, list[float]]] = {}
    base_agent = (
        _entry(baseline["baseline_benchmarks"], "agent", "agent a")
        if baseline
        else None
    )
    for condition, runs in groups.items():
        for run in runs:
            if base_agent:
                for bench, score in base_agent["scores"].items():
                    metric = _metric(score)
                    vals = [
                        e["scores"][bench][metric] - score[metric]
                        for e in run["final_benchmarks"]
                        if e["mode"] == "agent" and bench in e["scores"]
                    ]
                    if vals:
                        out.setdefault(bench, {}).setdefault(condition, []).append(
                            statistics.mean(vals)
                        )
            vals = []
            for a in run["agents"]:
                start = run["baseline_validation"][a]
                end = run["cycles"][-1]["validation"][a]
                vals.append(
                    statistics.mean(end[d] for d in end)
                    - statistics.mean(start[d] for d in start)
                )
            out.setdefault("validation", {}).setdefault(condition, []).append(
                statistics.mean(vals)
            )
    return out


def comparison_table(changes: dict[str, dict[str, list[float]]]) -> str:
    lines = [
        "| Measure | Pair change | Solo change | Pair − solo | 95% CI | p (permutation) |",
        "|---|---|---|---|---|---|",
    ]
    for bench, by in changes.items():
        if "pair" not in by or "solo" not in by:
            continue
        a, b = by["pair"], by["solo"]
        c = compare(a, b)
        lines.append(
            f"| {bench} | {100 * statistics.mean(a):+.1f} (n={len(a)}) | "
            f"{100 * statistics.mean(b):+.1f} (n={len(b)}) | {100 * c['diff']:+.1f} | "
            f"[{100 * c['ci_low']:+.1f}, {100 * c['ci_high']:+.1f}] | {c['p']:.3f} |"
        )
    return "\n".join(lines)


# ── figures ─────────────────────────────────────────────────────────────────


# Conference column widths in inches (NeurIPS/ICLR: ~3.25 single, ~6.75 full).
ONE_COLUMN, FULL_WIDTH = 3.25, 6.75


def _plt() -> Any:
    """Matplotlib with a publication style: serif text like the paper body,
    small sizes for figures printed at column width, thin axes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "lines.linewidth": 1.4,
            "lines.markersize": 3.5,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
        }
    )
    return plt


def _save(fig: Any, out: Path, name: str) -> str:
    """Vector PDF for the paper, PNG for quick viewing."""
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=300)
    _plt().close(fig)
    return f"{name}.png"


def _ci95(values: list[float]) -> float:
    """Half-width of a normal-approximation 95% confidence interval of the mean."""
    if len(values) < 2:
        return 0.0
    return float(1.96 * statistics.stdev(values) / len(values) ** 0.5)


def _style(ax: Any) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#ecebe7", linewidth=0.8)
    ax.set_axisbelow(True)


def _diverging() -> Any:
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("div", ["#eb6834", "#f0efec", "#2a78d6"])


def fig_benchmark_change(changes: dict[str, dict[str, list[float]]], out: Path) -> str:
    plt = _plt()
    benches = [b for b in changes if b != "validation"]
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.4))
    width = 0.38
    for j, condition in enumerate(["pair", "solo"]):
        for i, bench in enumerate(benches):
            vals = [100 * v for v in changes[bench].get(condition, [])]
            if not vals:
                continue
            x = i + (j - 0.5) * width
            ax.bar(
                x,
                statistics.mean(vals),
                width * 0.9,
                color=COLORS[condition],
                yerr=_ci95(vals),
                error_kw={"elinewidth": 0.7, "capsize": 2},
                label=condition if i == 0 else None,
            )
            ax.scatter([x] * len(vals), vals, s=5, color="#52514e", zorder=3)
    ax.axhline(0, color="#52514e", linewidth=0.8)
    ax.set_xticks(range(len(benches)))
    ax.set_xticklabels(benches, rotation=30, ha="right")
    ax.set_ylabel("Change, trained − untrained (pp)")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    return _save(fig, out, "benchmark_change")


def fig_validation_by_domain(groups: dict[str, list[dict[str, Any]]], out: Path) -> str:
    DOMAINS = _domains(groups)  # noqa: N806
    plt = _plt()
    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(FULL_WIDTH, 1.9), sharey=True)
    for ax, d in zip(axes, DOMAINS):
        for condition, runs in groups.items():
            n = max(len(r["cycles"]) for r in runs)
            means, sds = [], []
            for i in range(n + 1):
                vals = [
                    100
                    * (
                        r["baseline_validation"][a][d]
                        if i == 0
                        else r["cycles"][i - 1]["validation"][a][d]
                    )
                    for r in runs
                    for a in r["agents"]
                    if i == 0 or i - 1 < len(r["cycles"])
                ]
                means.append(statistics.mean(vals))
                sds.append(_ci95(vals))
            xs = range(n + 1)
            ax.plot(
                xs,
                means,
                color=COLORS[condition],
                linewidth=2,
                marker="o",
                markersize=4,
                label=condition,
            )
            ax.fill_between(
                xs,
                [m - s for m, s in zip(means, sds)],
                [m + s for m, s in zip(means, sds)],
                color=COLORS[condition],
                alpha=0.12,
                linewidth=0,
            )
        ax.set_title(d)
        ax.set_xlabel("Cycle")
        _style(ax)
    axes[0].set_ylabel("Whole agent, validation (%)")
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    return _save(fig, out, "validation_by_domain")


def _specialist_change(runs: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    acc: dict[tuple[str, str], list[float]] = {}
    for r in runs:
        last = r["cycles"][-1]
        for a in r["agents"]:
            for s, by in r["baseline_validation_specialists"][a].items():
                for d, v0 in by.items():
                    acc.setdefault((s, d), []).append(
                        last["validation_specialists"][a][s][d] - v0
                    )
    return {k: statistics.mean(v) for k, v in acc.items()}


def _owners(runs: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    DOMAINS = list(runs[0]["config"]["data"]["domains"])  # noqa: N806
    counts: dict[tuple[str, str], int] = {}
    for r in runs:
        for a in r["agents"]:
            rec = r["cycles"][-1]["domain_records"].get(a, {})
            for d in DOMAINS:
                if rec:
                    best = max(
                        SPECIALISTS,
                        key=lambda s: (
                            rec.get(s, {}).get(d, 0.5),
                            -SPECIALISTS.index(s),
                        ),
                    )  # noqa: B023
                    counts[(best, d)] = counts.get((best, d), 0) + 1
    return counts


def fig_specialisation_heatmaps(
    groups: dict[str, list[dict[str, Any]]], out: Path
) -> str:
    DOMAINS = _domains(groups)  # noqa: N806
    plt = _plt()
    fig, axes = plt.subplots(1, len(groups), figsize=(FULL_WIDTH, 2.4))
    axes = axes if len(groups) > 1 else [axes]
    for ax, (condition, runs) in zip(axes, groups.items()):
        change, owners = _specialist_change(runs), _owners(runs)
        agents = sum(len(r["agents"]) for r in runs)
        grid = [[100 * change.get((s, d), 0.0) for d in DOMAINS] for s in SPECIALISTS]
        bound = max(5.0, max(abs(v) for row in grid for v in row))
        im = ax.imshow(grid, cmap=_diverging(), vmin=-bound, vmax=bound)
        for i, s in enumerate(SPECIALISTS):
            for j, d in enumerate(DOMAINS):
                own = owners.get((s, d), 0)
                ax.text(
                    j,
                    i,
                    f"{grid[i][j]:+.0f}\n{own}/{agents}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="#0b0b0b",
                )
        ax.set_xticks(range(len(DOMAINS)), DOMAINS)
        ax.set_yticks(range(len(SPECIALISTS)), SPECIALISTS)
        ax.set_title(f"{condition} (cell: change in pp / agents owning)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _save(fig, out, "specialisation_heatmap")


def fig_specialist_benchmarks(
    groups: dict[str, list[dict[str, Any]]], baseline: dict[str, Any] | None, out: Path
) -> str | None:
    if not baseline:
        return None
    plt = _plt()
    base = {
        e["system"].split("/")[1]: e
        for e in baseline["baseline_benchmarks"]
        if e["mode"] == "specialist"
    }
    if not base:
        return None
    benches = list(next(iter(base.values()))["scores"])
    fig, axes = plt.subplots(len(groups), 1, figsize=(FULL_WIDTH, 2.2 * len(groups)))
    axes = axes if len(groups) > 1 else [axes]
    for ax, (condition, runs) in zip(axes, groups.items()):
        grid = []
        for s in SPECIALISTS:
            row = []
            for b in benches:
                metric = _metric(base[s]["scores"][b])
                vals = [
                    e["scores"][b][metric] - base[s]["scores"][b][metric]
                    for r in runs
                    for e in r["final_benchmarks"]
                    if e["mode"] == "specialist" and e["system"].endswith("/" + s)
                ]
                row.append(100 * statistics.mean(vals) if vals else 0.0)
            grid.append(row)
        bound = max(5.0, max(abs(v) for row in grid for v in row))
        im = ax.imshow(grid, cmap=_diverging(), vmin=-bound, vmax=bound)
        for i in range(len(SPECIALISTS)):
            for j in range(len(benches)):
                ax.text(
                    j, i, f"{grid[i][j]:+.0f}", ha="center", va="center", fontsize=6.5
                )
        ax.set_xticks(range(len(benches)), benches, rotation=35, ha="right")
        ax.set_yticks(range(len(SPECIALISTS)), SPECIALISTS)
        ax.set_title(condition)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _save(fig, out, "specialist_benchmarks")


def fig_dynamics(groups: dict[str, list[dict[str, Any]]], out: Path) -> str:
    plt = _plt()
    panels = [
        ("learned_from_peer", "Learned from partner\n(problems per agent)", 1.0),
        ("discussion_changed", "Answers changed\nby discussion (%)", 100.0),
        ("discussion_fixed", "Changed answers\nwrong → right (%)", 100.0),
        ("answer_divergence", "Agents' answers\ndiffer (%)", 100.0),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(FULL_WIDTH, 1.8))
    for ax, (key, title, scale) in zip(axes, panels):
        for condition, runs in groups.items():
            n = max(len(r["cycles"]) for r in runs)
            ys = []
            for i in range(n):
                vals: list[float] = []
                for r in runs:
                    if i >= len(r["cycles"]):
                        continue
                    v = r["cycles"][i][key]
                    if isinstance(v, dict):
                        vals.extend(float(x) for x in v.values())
                    elif v is not None:
                        vals.append(float(v))
                ys.append(scale * statistics.mean(vals) if vals else float("nan"))
            ax.plot(
                range(1, n + 1),
                ys,
                color=COLORS[condition],
                linewidth=2,
                marker="o",
                markersize=4,
                label=condition,
            )
        ax.set_title(title)
        ax.set_xlabel("Cycle")
        _style(ax)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    return _save(fig, out, "dynamics")


def fig_training(groups: dict[str, list[dict[str, Any]]], out: Path) -> str | None:
    plt = _plt()
    roles = [*SPECIALISTS, "coordinator"]
    colors = {**SPECIALIST_COLORS, "coordinator": "#6250d6"}
    runs = [r for rs in groups.values() for r in rs]
    if not any(c.get("training") for r in runs for c in r["cycles"]):
        return None
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 1.9))
    for ax, (key, title, scale) in zip(
        axes,
        [
            ("pairs", "Training pairs", 1.0),
            ("reward_accuracy", "DPO reward accuracy (%)", 100.0),
            ("loss", "Final loss", 1.0),
        ],
    ):
        n = max(len(r["cycles"]) for r in runs)
        for role in roles:
            ys = []
            for i in range(n):
                vals = [
                    float(t[key])
                    for r in runs
                    if i < len(r["cycles"])
                    for t in [
                        r["cycles"][i].get("training", {}).get(a, {}).get(role)
                        for a in r["agents"]
                    ]
                    if t and t.get(key) is not None
                ]
                ys.append(scale * statistics.mean(vals) if vals else float("nan"))
            ax.plot(
                range(1, n + 1),
                ys,
                color=colors[role],
                linewidth=2,
                marker="o",
                markersize=4,
                label=role,
            )
        ax.set_title(title)
        ax.set_xlabel("Cycle")
        _style(ax)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save(fig, out, "training")


def fig_validation_overall(groups: dict[str, list[dict[str, Any]]], out: Path) -> str:
    """Learning curve: whole agent, mean over domains, 95% CI band."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(ONE_COLUMN, 2.1))
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
                    vals.append(100 * statistics.mean(v.values()))
            means.append(statistics.mean(vals))
            cis.append(_ci95(vals))
        xs = range(n + 1)
        agents = sum(len(r["agents"]) for r in runs)
        ax.plot(
            xs,
            means,
            color=COLORS[condition],
            marker="o",
            label=f"{condition} ({agents} agents)",
        )
        ax.fill_between(
            xs,
            [m - c for m, c in zip(means, cis)],
            [m + c for m, c in zip(means, cis)],
            color=COLORS[condition],
            alpha=0.15,
            linewidth=0,
        )
    ax.set_xlabel("Cycle (0 = untrained)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    return _save(fig, out, "validation")


def fig_responsibility(groups: dict[str, list[dict[str, Any]]], out: Path) -> str:
    """Who learned each domain: share of a domain's training pairs that went to
    each specialist (all cycles pooled), against the uniform 25% line — the same
    view MoE papers use for expert specialisation (e.g. OLMoE)."""
    DOMAINS = _domains(groups)  # noqa: N806
    plt = _plt()
    fig, axes = plt.subplots(1, len(groups), figsize=(FULL_WIDTH, 2.0), sharey=True)
    axes = axes if len(groups) > 1 else [axes]
    width = 0.2
    for ax, (condition, runs) in zip(axes, groups.items()):
        counts = {(s, d): 0 for s in SPECIALISTS for d in DOMAINS}
        for r in runs:
            for c in r["cycles"]:
                for a in r["agents"]:
                    for s in SPECIALISTS:
                        for d, n in c["pairs_by_domain"].get(a, {}).get(s, {}).items():
                            counts[(s, d)] = counts.get((s, d), 0) + n
        for k, s in enumerate(SPECIALISTS):
            shares = []
            for d in DOMAINS:
                total = sum(counts[(o, d)] for o in SPECIALISTS)
                shares.append(100 * counts[(s, d)] / total if total else 0.0)
            xs = [i + (k - 1.5) * width for i in range(len(DOMAINS))]
            ax.bar(xs, shares, width * 0.92, color=SPECIALIST_COLORS[s], label=s)
        ax.axhline(25, color="#8a8984", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(len(DOMAINS)), DOMAINS)
        ax.set_title(condition)
        _style(ax)
    axes[0].set_ylabel("Share of domain's training (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=len(SPECIALISTS),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.tight_layout()
    return _save(fig, out, "responsibility")


def markdown_tables_to_latex(markdown: str) -> str:
    """Every markdown table in the summary as a booktabs LaTeX table."""
    blocks, current, heading = [], [], ""
    for line in markdown.splitlines():
        if line.startswith("## "):
            heading = line[3:]
        if line.startswith("|"):
            current.append(line)
        elif current:
            blocks.append((heading, current))
            current = []
    if current:
        blocks.append((heading, current))

    def cells(row: str) -> list[str]:
        return [
            c.strip().replace("±", r"$\pm$").replace("%", r"\%").replace("_", r"\_")
            for c in row.strip("|").split("|")
        ]

    out = []
    for heading, rows in blocks:
        header, body = cells(rows[0]), [cells(r) for r in rows[2:]]
        out.append(
            "% " + heading + "\n\\begin{table}[t]\n\\centering\n\\small\n"
            "\\begin{tabular}{"
            + "l" * len(header)
            + "}\n\\toprule\n"
            + " & ".join(header)
            + r" \\"
            + "\n\\midrule\n"
            + "\n".join(" & ".join(r) + r" \\" for r in body)
            + "\n\\bottomrule\n\\end{tabular}\n\\caption{"
            + heading
            + "}\n\\end{table}\n"
        )
    return "\n".join(out)
