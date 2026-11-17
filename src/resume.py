"""Restarting a run that stopped (a crash or a machine restart) from the end
of its last finished cycle.

The report records every finished cycle. Everything the run wrote after that
point belongs to the unfinished cycle (or the unfinished final benchmarks) and
is set aside under `<run>/interrupted/<time>/`: the tail of the event and call
logs, the cycle's rounds, adapters, training data and memory snapshot. Memory
goes back to its snapshot from the end of the last finished cycle and the
adapter registry forgets versions trained after it, so the run continues
exactly as if it had never stopped.
"""

from __future__ import annotations

import gzip
import json
import shutil
import time
from pathlib import Path

from src.models.core import ExperimentReport
from src.training.registry import AdapterRegistry


def finished_cycles(run_dir: Path) -> int:
    report = ExperimentReport.model_validate_json((run_dir / "report.json").read_text())
    return len(report.cycles)


def _cutoff(events: Path, done: int) -> float | None:
    """Time the unfinished part began: the next cycle's wake or the final
    benchmarks."""
    with open(events, "rb") as f:
        for raw in f:
            if b'"type": "phase"' not in raw[:60]:
                continue
            try:
                e = json.loads(raw)
            except ValueError:
                continue
            if (e["phase"] == "wake" and e["cycle"] == done + 1) or e[
                "phase"
            ] == "benchmarks":
                return float(e["t"])
    return None


def _split_events(path: Path, cutoff: float | None, aside: Path) -> int:
    kept, moved = path.with_suffix(".kept"), aside / path.name
    count = 0
    with open(path, "rb") as f, open(kept, "wb") as k, open(moved, "wb") as m:
        for raw in f:
            if not raw.endswith(b"\n"):
                m.write(raw)
                continue
            try:
                t = json.loads(raw)["t"]
            except ValueError:
                m.write(raw)
                continue
            if cutoff is not None and t >= cutoff:
                m.write(raw)
                count += 1
            else:
                k.write(raw)
    kept.replace(path)
    return count


def _split_calls(path: Path, cutoff: float | None, aside: Path) -> int:
    """Rewrites the call log as one complete gzip stream: a stopped run leaves
    its last gzip member cut short, and records appended after it would be
    unreadable."""
    kept, moved = path.with_suffix(".kept"), aside / path.name
    count = 0
    with (
        gzip.open(path, "rt", encoding="utf-8") as f,
        gzip.open(kept, "wt", compresslevel=1, encoding="utf-8") as k,
        gzip.open(moved, "wt", compresslevel=1, encoding="utf-8") as m,
    ):
        try:
            for line in f:
                if not line.endswith("\n") or not line.strip():
                    continue
                if cutoff is not None and json.loads(line)["t"] >= cutoff:
                    m.write(line)
                    count += 1
                else:
                    k.write(line)
        except EOFError:
            pass
    kept.replace(path)
    return count


def _move(path: Path, aside: Path, run_dir: Path) -> None:
    if path.exists():
        target = aside / path.relative_to(run_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))


def prepare(run_dir: Path, memory_enabled: bool) -> dict:
    """Sets aside everything after the last finished cycle; returns what was
    done, for the run's event log."""
    done = finished_cycles(run_dir)
    aside = run_dir / "interrupted" / time.strftime("%Y%m%d-%H%M%S")
    aside.mkdir(parents=True)
    cutoff = _cutoff(run_dir / "events.jsonl", done)
    events = _split_events(run_dir / "events.jsonl", cutoff, aside)
    calls = _split_calls(run_dir / "calls.jsonl.gz", cutoff, aside)
    later = [f"cycle_{c}" for c in range(done + 1, 100)]
    for name in later:
        _move(run_dir / "rounds" / f"{name}.jsonl", aside, run_dir)
        _move(run_dir / "memory_snapshots" / name, aside, run_dir)
        for role in (run_dir / "adapters").glob("*/*"):
            for suffix in ("", "_new.jsonl", "_pairs.jsonl"):
                _move(role / f"{name}{suffix}", aside, run_dir)
    _move(run_dir / "benchmarks" / "final", aside, run_dir)
    AdapterRegistry(run_dir / "adapters" / "registry.json").drop_after(done)
    memory = run_dir / "memory"
    if memory_enabled:
        _move(memory, aside, run_dir)
        if done:
            shutil.copytree(run_dir / "memory_snapshots" / f"cycle_{done}", memory)
    return {
        "finished_cycles": done,
        "set_aside": str(aside.relative_to(run_dir)),
        "events_set_aside": events,
        "calls_set_aside": calls,
    }


def records(run_dir: Path, cycle: int) -> dict[str, dict]:
    """Each agent's competition records and win rates as they stood at the end
    of `cycle`, from its credit_rules event."""
    out: dict[str, dict] = {}
    with open(run_dir / "events.jsonl", "rb") as f:
        for raw in f:
            if b'"type": "credit_rules"' not in raw[:60]:
                continue
            e = json.loads(raw)
            if e["cycle"] == cycle:
                out[e["agent"]] = {"values": e["records"], "win_rate": e["win_rate"]}
    return out
