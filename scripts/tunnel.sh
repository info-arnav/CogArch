#!/usr/bin/env bash
# Public HTTPS link to the dashboard via a Cloudflare quick tunnel (no account).
# The link is long and random; anyone who has it can view the dashboard
# (read-only: prompts, code, scores). Stop it with: pkill -f "cloudflared tunnel"
set -euo pipefail

BIN=/usr/local/bin/cloudflared
if [ ! -x "$BIN" ]; then
  curl -fsSL -o "$BIN" \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x "$BIN"
fi

LOG=/tmp/cloudflared.log
pkill -f "cloudflared tunnel" >/dev/null 2>&1 || true
nohup "$BIN" tunnel --no-autoupdate --url http://localhost:8080 >"$LOG" 2>&1 &

for _ in $(seq 1 30); do
  url=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" | head -1 || true)
  if [ -n "$url" ]; then echo "$url"; exit 0; fi
  sleep 2
done
echo "Tunnel did not start; see $LOG" >&2
exit 1
