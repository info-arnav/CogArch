"""Live dashboard for running experiments.

    python -m cli.main dashboard            # serves http://127.0.0.1:8080

Reads every run under data/runs/ on each request: report.json (written after
the baseline and every cycle), the current cycle's rounds log (live progress)
and the run's log file. Binds to localhost; reach it from your machine with
    ssh -L 8080:localhost:8080 root@<server>
"""

from __future__ import annotations

import json
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

PAGE = Path(__file__).with_name("dashboard.html")


def _tail(path: Path, lines: int) -> list[str]:
    if not path.exists():
        return []
    with open(path, errors="replace") as f:
        return [line.rstrip() for line in deque(f, maxlen=lines) if line.strip()]


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def run_summary(run_dir: Path, logs_dir: Path) -> dict[str, Any] | None:
    report_path = run_dir / "report.json"
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text())
    cfg = report["config"]
    total_cycles = cfg["experiment"]["cycles"]
    done = len(report["cycles"])
    current = min(done + 1, total_cycles)
    log = logs_dir / f"{run_dir.name}.log"
    if report["final_eval"]:
        stage = "finished"
    elif done == total_cycles:
        stage = "final evaluation"
    else:
        stage = f"cycle {current}"
    return {
        "name": run_dir.name,
        "agents": cfg["agents"]["names"],
        "condition": "solo" if len(cfg["agents"]["names"]) == 1 else "pair",
        "stage": stage,
        "cycles_done": done,
        "cycles_total": total_cycles,
        "rounds_done": (
            _count_lines(run_dir / "rounds" / f"cycle_{current}.jsonl")
            if stage.startswith("cycle")
            else None
        ),
        "rounds_total": cfg["experiment"]["rounds_per_cycle"],
        "baseline_validation": report["baseline_validation"],
        "baseline_eval": report["baseline_eval"],
        "final_eval": report["final_eval"],
        "cycles": report["cycles"],
        "log": _tail(log, 12),
    }


def collect(runs_root: Path) -> list[dict[str, Any]]:
    logs = runs_root / "logs"
    runs = []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir() and p != logs):
        summary = run_summary(run_dir, logs)
        if summary is not None:
            runs.append(summary)
    return runs


def serve(runs_root: Path, host: str = "127.0.0.1", port: int = 8080) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 — http.server API
            if self.path.startswith("/api/runs"):
                body = json.dumps(collect(runs_root)).encode()
                content_type = "application/json"
            elif self.path in ("/", "/index.html"):
                body = PAGE.read_bytes()
                content_type = "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    ThreadingHTTPServer((host, port), Handler).serve_forever()
