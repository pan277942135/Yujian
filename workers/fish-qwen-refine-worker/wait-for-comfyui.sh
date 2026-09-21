#!/usr/bin/env bash
set -u

COMFYUI_URL="${COMFYUI_URL:-http://127.0.0.1:8188}"
READY_TIMEOUT="${COMFYUI_READY_TIMEOUT_SECONDS:-120}"
READY_POLL="${COMFYUI_READY_POLL_SECONDS:-2}"
DEADLINE=$((SECONDS + READY_TIMEOUT))

while (( SECONDS < DEADLINE )); do
  HTTP_STATUS="$(
    curl --connect-timeout 2 --max-time 5 -sS       -o /dev/null -w '%{http_code}'       "${COMFYUI_URL%/}/system_stats" 2>/dev/null || true
  )"
  if [[ "$HTTP_STATUS" =~ ^[234][0-9]{2}$ ]]; then
    echo "ComfyUI ready url=$COMFYUI_URL http_status=$HTTP_STATUS"
    exit 0
  fi
  sleep "$READY_POLL"
done

echo "ComfyUI readiness timeout url=$COMFYUI_URL timeout=${READY_TIMEOUT}s" >&2
exit 1
