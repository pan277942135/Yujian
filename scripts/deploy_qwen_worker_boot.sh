#!/usr/bin/env bash
set -euo pipefail

: "${SERVICE_URL:?SERVICE_URL is required}"
: "${PROJECT_ID:?PROJECT_ID is required}"
: "${GPU_PROJECT_ID:?GPU_PROJECT_ID is required}"
: "${GPU_ZONE:?GPU_ZONE is required}"
: "${GPU_INSTANCE:?GPU_INSTANCE is required}"

WORKER_SOURCE="${WORKER_SOURCE:-workers/fish-qwen-refine-worker}"
REMOTE_PARENT="/tmp/yujian-qwen-worker-${GITHUB_RUN_ID:-$$}"
OUT_DIR="$(mktemp -d -t yujian-qwen-boot.XXXXXX)"
COOKIE_JAR="$OUT_DIR/cookies.txt"
STATUS_JSON="$OUT_DIR/status.json"
START_REQUESTED=false

cleanup() {
  local rc=$?
  if [[ "$rc" -ne 0 && "$START_REQUESTED" == "true" ]]; then
    echo "Bootstrap failed; stopping the VM through Compute API to avoid leaving GPU running."
    gcloud compute instances stop "$GPU_INSTANCE"       --project "$GPU_PROJECT_ID" --zone "$GPU_ZONE" --quiet || true
  fi
  rm -rf "$OUT_DIR"
  exit "$rc"
}
trap cleanup EXIT

request() {
  local method="$1"
  local url="$2"
  local output="$3"
  shift 3
  curl --retry 3 --retry-all-errors --retry-delay 2     --connect-timeout 10 --max-time 60 -sS     -b "$COOKIE_JAR" -c "$COOKIE_JAR"     -o "$output" -w '%{http_code}'     -X "$method" "$url" "$@"
}

display_status() {
  python3 - "$STATUS_JSON" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(str(json.load(handle).get("display_status") or "ERROR").upper())
PY
}

CONSOLE_KEY="$(gcloud secrets versions access latest --secret=yujian-console-access-key --project="$PROJECT_ID")"
test -n "$CONSOLE_KEY"
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then echo "::add-mask::$CONSOLE_KEY"; fi
LOGIN_HTTP="$(curl --retry 3 --retry-all-errors --retry-delay 2   --connect-timeout 10 --max-time 30 -sS   -b "$COOKIE_JAR" -c "$COOKIE_JAR" -o /dev/null -w '%{http_code}'   -X POST "$SERVICE_URL/login" --data-urlencode "access_key=$CONSOLE_KEY")"
test "$LOGIN_HTTP" = "303"
unset CONSOLE_KEY

STATUS_HTTP="$(request GET "$SERVICE_URL/api/qwen-lab/gpu/status?boot_ts=$RANDOM" "$STATUS_JSON")"
test "$STATUS_HTTP" = "200"
INITIAL_STATUS="$(display_status)"
echo "initial display_status=$INITIAL_STATUS"

if [[ "$INITIAL_STATUS" == "STOPPED" ]]; then
  START_JSON="$OUT_DIR/start.json"
  START_HTTP="$(request POST "$SERVICE_URL/api/qwen-lab/gpu/start" "$START_JSON")"
  test "$START_HTTP" = "200"
  START_REQUESTED=true
  python3 - "$START_JSON" <<'PY'
import json
import sys
payload=json.load(open(sys.argv[1], encoding="utf-8"))
assert payload.get("accepted") is True, payload
assert payload.get("display_status") == "STARTING", payload
PY
elif [[ "$INITIAL_STATUS" != "READY" && "$INITIAL_STATUS" != "LOADING" && "$INITIAL_STATUS" != "STARTING" && "$INITIAL_STATUS" != "BUSY" ]]; then
  cat "$STATUS_JSON"
  echo "Unexpected GPU state before worker bootstrap: $INITIAL_STATUS" >&2
  exit 1
fi

for attempt in $(seq 1 60); do
  VM_STATUS="$(gcloud compute instances describe "$GPU_INSTANCE"     --project "$GPU_PROJECT_ID" --zone "$GPU_ZONE" --format='value(status)' 2>/dev/null || true)"
  echo "vm_status=$VM_STATUS"
  [[ "$VM_STATUS" == "RUNNING" ]] && break
  sleep 5
done
[[ "${VM_STATUS:-}" == "RUNNING" ]]

remote_ssh() {
  timeout 120s gcloud compute ssh "$GPU_INSTANCE"     --project "$GPU_PROJECT_ID" --zone "$GPU_ZONE" --quiet     --command="$1"
}

for attempt in $(seq 1 24); do
  if remote_ssh "mkdir -p '$REMOTE_PARENT'"; then
    break
  fi
  echo "waiting for SSH attempt=$attempt"
  sleep 5
done
[[ "$(remote_ssh "test -d '$REMOTE_PARENT' && echo ready")" == "ready" ]]

for attempt in $(seq 1 6); do
  if gcloud compute scp --recurse "$WORKER_SOURCE"       "$GPU_INSTANCE:$REMOTE_PARENT/"       --project "$GPU_PROJECT_ID" --zone "$GPU_ZONE" --quiet; then
    break
  fi
  echo "waiting for SCP attempt=$attempt"
  sleep 5
done

remote_ssh "
set -euo pipefail
STAGE='$REMOTE_PARENT/fish-qwen-refine-worker'
sudo install -d -o pan277942135 -g pan277942135 /opt/fish-qwen-refine-worker
sudo install -o pan277942135 -g pan277942135 -m 0644 "$STAGE/worker.py" /opt/fish-qwen-refine-worker/worker.py
sudo install -o pan277942135 -g pan277942135 -m 0644 "$STAGE/config.yaml" /opt/fish-qwen-refine-worker/config.yaml
sudo install -o pan277942135 -g pan277942135 -m 0644 "$STAGE/qwen2511_api.json" /opt/fish-qwen-refine-worker/qwen2511_api.json
sudo install -o root -g root -m 0644 "$STAGE/fish-qwen-comfyui.service" /etc/systemd/system/fish-qwen-comfyui.service
sudo install -o root -g root -m 0644 "$STAGE/fish-qwen-refine-worker.service" /etc/systemd/system/fish-qwen-refine-worker.service
sudo install -o pan277942135 -g pan277942135 -m 0755 "$STAGE/wait-for-comfyui.sh" /opt/fish-qwen-refine-worker/wait-for-comfyui.sh
sudo install -d -o pan277942135 -g pan277942135 /opt/fish-qwen-refine-worker/output
sudo systemctl daemon-reload
sudo systemctl enable fish-qwen-comfyui.service fish-qwen-refine-worker.service
sudo systemctl restart fish-qwen-comfyui.service
sudo systemctl restart fish-qwen-refine-worker.service
sudo systemctl is-enabled fish-qwen-comfyui.service fish-qwen-refine-worker.service
"

for attempt in $(seq 1 180); do
  STATUS_HTTP="$(request GET "$SERVICE_URL/api/qwen-lab/gpu/status?boot_ready_ts=$RANDOM" "$STATUS_JSON" || true)"
  if [[ "$STATUS_HTTP" == "200" ]]; then
    CURRENT_STATUS="$(display_status)"
    echo "worker bootstrap attempt=$attempt display_status=$CURRENT_STATUS"
    if [[ "$CURRENT_STATUS" == "READY" ]]; then
      echo "Qwen worker boot configuration installed and Ready."
      break
    fi
    if [[ "$CURRENT_STATUS" == "ERROR" ]]; then
      cat "$STATUS_JSON"
      remote_ssh "sudo systemctl status fish-qwen-comfyui.service fish-qwen-refine-worker.service --no-pager -l; sudo journalctl -u fish-qwen-comfyui.service -u fish-qwen-refine-worker.service -n 120 --no-pager" || true
      exit 1
    fi
  else
    echo "worker bootstrap status_http=$STATUS_HTTP"
  fi
  sleep 5
done
[[ "${CURRENT_STATUS:-}" == "READY" ]]

{
  echo "### Qwen VM boot dependency bootstrap"
  echo "- ComfyUI service installed: **PASS**"
  echo "- Qwen Worker Requires/After ComfyUI: **PASS**"
  echo "- Worker readiness gate: **PASS**"
  echo "- Worker health: **READY**"
  echo "- VM intentionally left RUNNING for the following page-driven UAT"
} >> "${GITHUB_STEP_SUMMARY:-/dev/stdout}"
