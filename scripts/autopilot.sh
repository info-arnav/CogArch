#!/usr/bin/env bash
# Runs the whole experiment unattended on the GPU server, detached from any
# SSH session:
#   setsid nohup scripts/autopilot.sh [replicates] > /dev/null 2>&1 &
#
# Steps: public dashboard link → setup → vLLM → dashboard → smoke test (stops
# here if it fails) → per-domain calibration → untrained baseline → pair vs
# solo runs → analysis. Progress: data/runs/logs/autopilot.log and the dashboard.
#
# Steps can be skipped when they already ran on this machine:
#   SKIP_SETUP=1 SKIP_SMOKE=1 SKIP_CALIBRATION=1 SKIP_BASELINE=1
set -uo pipefail

REPLICATES="${1:-4}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
mkdir -p data/runs/logs
LOG=data/runs/logs/autopilot.log
EXP="docker exec -w $REPO cogarch-exp"

step() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
fail() { step "FAILED: $*"; exit 1; }

step "autopilot start: model=$MODEL replicates=$REPLICATES"

# 1. Public link first, so it is known as early as possible (dashboard comes later).
#    An already-running tunnel is kept, so the link stays the same.
if pgrep -f "cloudflared tunnel" > /dev/null && [ -s data/dashboard_url.txt ]; then
  url=$(cat data/dashboard_url.txt)
else
  url=$(scripts/tunnel.sh) || fail "tunnel"
  echo "$url" > data/dashboard_url.txt
fi
step "dashboard link: $url"

# 2. Environment.
if [ -z "${SKIP_SETUP:-}" ]; then
  step "setup"
  scripts/setup_server.sh >> "$LOG" 2>&1 || fail "setup_server.sh"
  step "vLLM"
  MODEL="$MODEL" scripts/serve_vllm.sh >> "$LOG" 2>&1 || fail "vLLM"
fi
if ! $EXP pgrep -f "cli.main dashboard" > /dev/null 2>&1; then
  docker exec -d -w "$REPO" cogarch-exp sh -c "python -m cli.main dashboard > /tmp/dashboard.log 2>&1"
fi

# 3. Smoke test: every component, then a tiny end-to-end run. Stop if it fails.
if [ -z "${SKIP_SMOKE:-}" ]; then
step "smoke: components"
$EXP python -m cli.main smoke config/smoke.yaml >> "$LOG" 2>&1 || fail "smoke components"
step "smoke: end-to-end"
$EXP python -m cli.main run config/smoke.yaml > data/runs/logs/smoke-r1.log 2>&1 \
  || fail "smoke end-to-end (see data/runs/logs/smoke-r1.log)"
mkdir -p data/smoke-archive && rm -rf data/smoke-archive/smoke-r1
mv data/runs/smoke-r1 data/smoke-archive/ 2>/dev/null || true
fi

# 4. Calibration and the untrained baseline.
if [ -z "${SKIP_CALIBRATION:-}" ]; then
  step "calibration"
  $EXP python -m cli.main calibrate config/general.yaml >> "$LOG" 2>&1 || step "calibration failed (continuing)"
fi
if [ -z "${SKIP_BASELINE:-}" ]; then
  step "baseline benchmarks"
  $EXP python -m cli.main baseline config/general.yaml > data/runs/logs/baseline.log 2>&1 \
    || step "baseline failed (continuing; see data/runs/logs/baseline.log)"
fi

# 5. The experiment.
step "launching pair vs solo, $REPLICATES replicates each"
scripts/launch_pilot.sh "$REPLICATES" >> "$LOG" 2>&1
sleep 60
while $EXP pgrep -f "cli.main run" > /dev/null 2>&1; do
  sleep 300
  step "running: $($EXP sh -c 'ls data/runs/*/report.json 2>/dev/null | wc -l') reports"
done
step "all runs finished"
grep -l "Traceback" data/runs/logs/v2-*.log data/runs/logs/general-*.log >> "$LOG" 2>/dev/null \
  && step "some runs had errors (listed above)"

# 6. Analysis.
step "analysis"
$EXP python -m cli.main analyze >> "$LOG" 2>&1 || fail "analysis"
step "done: data/analysis/summary.md"
