"""A close look at what each run is doing now, from the end of its event log:
the latest problems with every specialist's answer, the coordinator's letter
and the verdict; how often memory was shown; empty answers; letter picks;
training pairs; validation; and whether the run's settings match its config.

    python3 scripts/runpod/inspect_runs.py [run ...]
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
from src.config import load_config

RUNS = Path("data/runs")
TAIL = 3_000_000


def tail_events(path: Path) -> list[dict]:
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - TAIL))
        lines = f.read().decode("utf-8", "replace").splitlines()[1:]
    out = []
    for line in lines:
        if '"type": "delta"' in line[:40]:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def settings_drift(run: Path) -> list[str]:
    report = json.loads((run / "report.json").read_text())["config"]
    name = report["experiment"]["name"].rsplit("-r", 1)[0]
    expected = load_config(f"config/runs/{name}.yaml").model_dump(mode="json")
    drift = []
    for section in (
        "agents",
        "training",
        "memory",
        "evaluation",
        "environment",
        "experiment",
        "data",
    ):
        for key, value in expected.get(section, {}).items():
            if key in ("name", "output_dir", "slots_dir", "seeds"):
                continue
            got = report.get(section, {}).get(key)
            if got != value:
                drift.append(f"{section}.{key}: {got!r} != {value!r}")
    return drift


def short(text: str, n: int = 60) -> str:
    return " ".join(str(text).split())[:n]


for run in (
    sorted(RUNS.iterdir()) if len(sys.argv) < 2 else [RUNS / a for a in sys.argv[1:]]
):
    ev_path = run / "events.jsonl"
    if (
        run.name.startswith("check")
        or not ev_path.exists()
        or not (run / "report.json").exists()
    ):
        continue
    ev = tail_events(ev_path)
    phases = [e for e in ev if e["type"] == "phase"]
    decisions = [e for e in ev if e["type"] == "decision"]
    verdicts = {
        (e.get("task_id"), e.get("agent"), e.get("attempt")): e
        for e in ev
        if e["type"] == "verdict"
    }
    rounds = {
        (e.get("task_id"), e.get("agent"), e.get("attempt")): e
        for e in ev
        if e["type"] == "round"
    }
    recalls = [e for e in ev if e["type"] == "recall"]
    pairs = [e for e in ev if e["type"] == "pairs"]
    vals = [e for e in ev if e["type"] == "validation_done"]
    picks = Counter(e.get("chosen") for e in decisions)
    empty = sum(
        1 for e in rounds.values() for a in (e.get("answers") or {}).values() if not a
    )
    total_answers = sum(len(e.get("answers") or {}) for e in rounds.values())
    print(
        f"== {run.name}  phase={phases[-1]['phase'] + str(phases[-1]['cycle']) if phases else '?'}"
    )
    drift = settings_drift(run)
    print(
        f"   settings: {'match config' if not drift else 'DRIFT ' + '; '.join(drift)}"
    )
    if recalls:
        shown = sum(1 for e in recalls if e.get("text"))
        print(f"   memory: shown on {shown}/{len(recalls)} recent problems")
    if total_answers:
        print(
            f"   answers: {empty}/{total_answers} recent specialist answers empty (no extractable answer)"
        )
    if decisions:
        right = sum(1 for e in verdicts.values() if e.get("passed"))
        print(
            f"   decisions: {len(decisions)} recent, picks {dict(picks)}, verdicts right {right}/{len(verdicts)}"
        )
        for d in decisions[-2:]:
            key = (d.get("task_id"), d.get("agent"), d.get("attempt"))
            answers = (rounds.get(key) or {}).get("answers") or {}
            v = verdicts.get(key, {})
            print(
                f"   e.g. {d.get('task_id')} team {d.get('agent')} try {d.get('attempt')}: "
                + ", ".join(f"{s[:3]}={short(a, 18)!r}" for s, a in answers.items())
                + f" -> coordinator {d.get('chosen_by')}:{d.get('chosen')} -> {'right' if v.get('passed') else 'wrong' if v else '?'}"
            )
    starts = {e["call"]: e for e in ev if e["type"] == "call_start"}
    ends = {e["call"]: e for e in ev if e["type"] == "call_end"}
    shown_teams = set()
    for d in reversed(decisions):
        team = d.get("agent")
        if team in shown_teams:
            continue
        key = (d.get("task_id"), team, d.get("attempt"))
        calls = [
            (starts[c], ends[c])
            for c in starts
            if c in ends
            and (
                starts[c].get("task_id"),
                starts[c].get("agent"),
                starts[c].get("attempt"),
            )
            == key
        ]
        if not calls:
            continue
        shown_teams.add(team)
        v = verdicts.get(key, {})
        print(
            f"   example team {team}, {d.get('task_id')} try {d.get('attempt')} -> {'right' if v.get('passed') else 'wrong' if v else '?'}"
        )
        for start, end in sorted(calls, key=lambda x: x[0]["t"]):
            model = start.get("model", "")
            weights = (
                "trained " + model.rsplit("-", 1)[-1]
                if model.count("-") > 3
                else "base"
            )
            reply = " ".join(str(end.get("response", "")).split())
            print(
                f"      {start.get('role')}/{start.get('stage')} [{weights}]: ...{reply[-150:]!r}"
            )
        if len(shown_teams) == 2:
            break
    for p in pairs[-2:]:
        print(
            f"   night {p.get('cycle')} pairs team {p.get('agent')}: "
            + ", ".join(f"{r}={len(v)}" for r, v in p["pairs"].items())
        )
    for v in vals[-2:]:
        print(
            "   validation: "
            + ", ".join(
                f"team {a} {round(100 * sum(x.values()) / len(x), 1)}%"
                for a, x in (v.get("whole") or {}).items()
                if x
            )
        )
