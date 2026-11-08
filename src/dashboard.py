"""Live dashboard for running experiments.

    python -m cli.main dashboard            # serves http://127.0.0.1:8080
    scripts/tunnel.sh                       # public https link via Cloudflare

Everything is read from disk on request, so it shows runs as they progress:
  /api/runs                     overview of every run
  /api/run?name=R               one run's report, logs and available cycles
  /api/rounds?name=R&cycle=N    one line per problem in a cycle
  /api/round?name=R&cycle=N&i=K one problem in full: every prompt and reply
  /api/pairs?name=R&agent=A&role=X&cycle=N   DPO pairs a part trained on
"""

from __future__ import annotations

import json
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

PAGE = Path(__file__).with_name("dashboard.html")
MAX_PAIRS_SHOWN = 30


def _tail(path: Path, lines: int) -> list[str]:
    if not path.exists():
        return []
    with open(path, errors="replace") as f:
        return [
            line.rstrip()
            for line in deque(f, maxlen=lines)
            if line.strip() and "it/s]" not in line and "s/it]" not in line
        ]


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


class RunStore:
    """Reads run folders; caches per-cycle round summaries by file size."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._summary_cache: dict[Path, tuple[int, list[dict[str, Any]]]] = {}

    def run_dir(self, name: str) -> Path:
        path = (self.root / name).resolve()
        if path.parent != self.root.resolve() or not (path / "report.json").exists():
            raise KeyError(name)
        return path

    def _report(self, run_dir: Path) -> dict[str, Any]:
        return dict(json.loads((run_dir / "report.json").read_text()))

    def overview(self, run_dir: Path) -> dict[str, Any]:
        report = self._report(run_dir)
        cfg = report["config"]
        total = cfg["experiment"]["cycles"]
        done = len(report["cycles"])
        current = min(done + 1, total)
        is_baseline = run_dir.name in ("baseline", "references")
        if report["final_benchmarks"] or (
            is_baseline and report["baseline_benchmarks"]
        ):
            stage = "finished"
        elif is_baseline:
            stage = "benchmarking the untrained system"
        elif done == total:
            stage = "final benchmarks"
        elif not report["baseline_validation"]:
            stage = "baseline validation"
        else:
            stage = f"cycle {current}"
        return {
            "name": run_dir.name,
            "kind": "baseline" if is_baseline else "run",
            "agents": cfg["agents"]["names"],
            "condition": "solo" if len(cfg["agents"]["names"]) == 1 else "pair",
            "model": cfg["model"]["base"],
            "domains": cfg["data"]["domains"],
            "stage": stage,
            "cycles_done": done,
            "cycles_total": total,
            "rounds_done": (
                _count_lines(run_dir / "rounds" / f"cycle_{current}.jsonl")
                if stage.startswith("cycle")
                else None
            ),
            "rounds_total": cfg["experiment"]["rounds_per_cycle"],
            "baseline_validation": report["baseline_validation"],
            "baseline_benchmarks": report["baseline_benchmarks"],
            "final_benchmarks": report["final_benchmarks"],
            "cycles": [
                {k: v for k, v in c.items() if k not in ("reflections", "training")}
                for c in report["cycles"]
            ],
            "log": _tail(self.root / "logs" / f"{run_dir.name}.log", 12),
        }

    def runs(self) -> list[dict[str, Any]]:
        return [
            self.overview(p)
            for p in sorted(self.root.iterdir())
            if p.is_dir() and (p / "report.json").exists()
        ]

    def run(self, name: str) -> dict[str, Any]:
        run_dir = self.run_dir(name)
        rounds_dir = run_dir / "rounds"
        cycles = (
            sorted(int(p.stem.split("_")[1]) for p in rounds_dir.glob("cycle_*.jsonl"))
            if rounds_dir.exists()
            else []
        )
        return {
            "overview": self.overview(run_dir),
            "report": self._report(run_dir),
            "round_cycles": cycles,
            "log": _tail(self.root / "logs" / f"{name}.log", 60),
        }

    def rounds(self, name: str, cycle: int) -> list[dict[str, Any]]:
        path = self.run_dir(name) / "rounds" / f"cycle_{cycle}.jsonl"
        if not path.exists():
            return []
        size = path.stat().st_size
        cached = self._summary_cache.get(path)
        if cached and cached[0] == size:
            return cached[1]
        summaries = []
        with open(path) as f:
            for i, line in enumerate(f):
                r = json.loads(line)
                summaries.append(
                    {
                        "i": i,
                        "task_id": r["task"]["task_id"],
                        "domain": r["task"]["domain"],
                        "agents": {
                            agent: {
                                "solved": any(
                                    d["result"]["passed"] for d in res["decisions"]
                                ),
                                "attempts": len(res["decisions"]),
                                "answers": [
                                    (
                                        d["answer"][:40]
                                        if r["task"]["kind"] != "code"
                                        else "code"
                                    )
                                    for d in res["decisions"]
                                ],
                                "learned_from": (
                                    r["known"][agent]["found_by"]
                                    if agent in r["known"]
                                    else None
                                ),
                            }
                            for agent, res in r["results"].items()
                        },
                    }
                )
        self._summary_cache[path] = (size, summaries)
        return summaries

    def round(self, name: str, cycle: int, index: int) -> dict[str, Any]:
        path = self.run_dir(name) / "rounds" / f"cycle_{cycle}.jsonl"
        with open(path) as f:
            for i, line in enumerate(f):
                if i == index:
                    return dict(json.loads(line))
        raise KeyError(index)

    def pairs(
        self, name: str, agent: str, role: str, cycle: int
    ) -> list[dict[str, Any]]:
        if not (agent.isalnum() and role.isalpha()):
            raise KeyError(role)
        path = (
            self.run_dir(name)
            / "adapters"
            / agent
            / role
            / f"cycle_{cycle}_pairs.jsonl"
        )
        if not path.exists():
            return []
        with open(path) as f:
            return [json.loads(line) for _, line in zip(range(MAX_PAIRS_SHOWN), f)]


def serve(runs_root: Path, host: str = "127.0.0.1", port: int = 8080) -> None:
    store = RunStore(runs_root)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 — http.server API
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/index.html"):
                    self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                    return
                if url.path == "/api/runs":
                    data: Any = store.runs()
                elif url.path == "/api/run":
                    data = store.run(q["name"])
                elif url.path == "/api/rounds":
                    data = store.rounds(q["name"], int(q["cycle"]))
                elif url.path == "/api/round":
                    data = store.round(q["name"], int(q["cycle"]), int(q["i"]))
                elif url.path == "/api/pairs":
                    data = store.pairs(
                        q["name"], q["agent"], q["role"], int(q["cycle"])
                    )
                else:
                    self._send(404, b"not found", "text/plain")
                    return
            except (KeyError, ValueError, FileNotFoundError):
                self._send(404, b"not found", "text/plain")
                return
            self._send(200, json.dumps(data).encode(), "application/json")

        def log_message(self, format: str, *args: Any) -> None:
            return

    ThreadingHTTPServer((host, port), Handler).serve_forever()
