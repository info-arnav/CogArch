#!/usr/bin/env bash
# Run on YOUR LAPTOP: copies run data off the GPU server every few minutes,
# over one persistent SSH connection (repeated logins can get rate-limited).
#   scripts/sync_to_laptop.sh root@<server-ip> [interval-seconds]
set -euo pipefail

HOST="$1"
INTERVAL="${2:-300}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SOCKET="$HOME/.ssh/cogarch-sync-%r@%h"

ssh -fNM -S "$SOCKET" -o ServerAliveInterval=15 "$HOST"
trap 'ssh -S "$SOCKET" -O exit "$HOST" 2>/dev/null' EXIT

while true; do
  rsync -a --partial --timeout=120 -e "ssh -S $SOCKET" \
    "$HOST:/root/CogArch/data/runs" "$HOST:/root/CogArch/data/calibration" \
    "$REPO/data/" 2>/dev/null || echo "$(date +%T) sync failed, retrying"
  echo "$(date +%T) synced $(du -sh "$REPO/data/runs" | cut -f1)"
  sleep "$INTERVAL"
done
