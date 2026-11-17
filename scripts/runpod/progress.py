"""Where every run is: phase, cycle, progress and timings, read incrementally
from each run's event log, with GPU and grader status. Prints one JSON object.

    python3 scripts/runpod/progress.py [runs_dir]
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

RUNS = Path(sys.argv[1] if len(sys.argv) > 1 else "data/runs")
STATE = Path(os.environ.get("PROGRESS_STATE", "/workspace/progress_state.json"))
SKIP = (
    '"type": "delta"',
    '"type": "call_start"',
    '"type": "call_end"',
    '"type": "pairs"',
    '"type": "round"',
    '"type": "credit"',
    '"type": "recall"',
    '"type": "memory_write"',
    '"type": "decision"',
    '"type": "verdict"',
    '"type": "known"',
    '"type": "rationalize"',
    '"type": "reflection"',
)
KEEP = {
    "run_start",
    "phase",
    "progress",
    "measure_start",
    "measure_progress",
    "training",
    "train_progress",
    "adapter",
    "validation_done",
    "benchmark",
    "rollback",
    "run_end",
}


def load_state() -> dict:
    try:
        state: dict = json.loads(STATE.read_text())
        return state
    except (OSError, ValueError):
        return {}


def update(run: Path, st: dict) -> dict:
    path = run / "events.jsonl"
    s = st.setdefault(
        run.name,
        {
            "offset": 0,
            "phases": [],
            "train_jobs": {},
            "benchmarks": 0,
            "rollbacks": 0,
            "validation": [],
            "progress": None,
            "measure": None,
            "last_t": None,
            "started": None,
        },
    )
    with open(path, "rb") as f:
        f.seek(s["offset"])
        chunk = f.read()
    end = chunk.rfind(b"\n") + 1
    for raw in chunk[:end].splitlines():
        line = raw.decode("utf-8", "replace")
        if any(k in line[:80] for k in SKIP):
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        t = e.get("type")
        if t not in KEEP:
            continue
        s["last_t"] = e["t"]
        if t == "run_start":
            s["started"] = e["t"]
        elif t == "phase":
            s["phases"].append([e["phase"], e["cycle"], e["t"]])
            s["measure"] = None
            s["progress"] = None
        elif t == "progress":
            s["progress"] = [e["done"], e["total"]]
        elif t == "measure_start":
            s["measure"] = [0, e["total"]]
        elif t == "measure_progress" and s["measure"]:
            s["measure"][0] = e["done"]
        elif t == "training":
            s["train_jobs"] = {j: [0, 0] for j in e["jobs"]}
        elif t == "train_progress":
            s["train_jobs"][e["job"]] = [e["step"], e["of"]]
        elif t == "validation_done":
            whole = e.get("whole") or {}
            s["validation"].append(
                {
                    a: round(100 * sum(v.values()) / len(v), 1)
                    for a, v in whole.items()
                    if v
                }
            )
        elif t == "benchmark":
            s["benchmarks"] += 1
        elif t == "rollback":
            s["rollbacks"] += 1
    s["offset"] += end
    run_state: dict = s
    return run_state


def gpus() -> list:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [
        [int(x) for x in line.split(",")] for line in out.strip().splitlines() if line
    ]


def grader() -> dict:
    url = os.environ.get("GRADER_URL", "http://localhost:8300/v1")
    req = urllib.request.Request(
        f"{url}/models",
        headers={"Authorization": f"Bearer {os.environ.get('VERIFIER_API_KEY', '')}"},
    )
    start = time.time()
    try:
        urllib.request.urlopen(req, timeout=10).read()
        return {"ok": True, "ms": round(1000 * (time.time() - start))}
    except OSError as error:
        return {"ok": False, "error": str(error)[:120]}


def errors(name: str) -> str | None:
    log = Path("/workspace/logs") / f"{name}.log"
    if not log.exists():
        return None
    tail = log.read_text(errors="replace")[-4000:]
    return tail[tail.rfind("Traceback") :][:600] if "Traceback" in tail else None


state = load_state()
runs = {}
for run in sorted(RUNS.iterdir()):
    if (run / "events.jsonl").exists() and not run.name.startswith("check"):
        runs[run.name] = update(run, state)
        runs[run.name]["error"] = errors(run.name)
STATE.write_text(json.dumps(state))
print(
    json.dumps({"now": time.time(), "runs": runs, "gpus": gpus(), "grader": grader()})
)
