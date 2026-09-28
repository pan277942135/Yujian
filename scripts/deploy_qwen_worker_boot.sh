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
  if [[ "$rc" -ne 0 ]]; then
    echo "Bootstrap failed; persistent GPU policy leaves the VM unchanged for diagnosis and interactive use."
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

worker_loaded() {
  python3 - "$STATUS_JSON" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
print("true" if payload.get("model_loaded") is True else "false")
PY
}

wait_for_idle_before_restart() {
  local attempt
  local http_code
  local state
  local loaded
  for attempt in $(seq 1 720); do
    http_code="$(request GET "$SERVICE_URL/api/qwen-lab/gpu/status?restart_wait_ts=$RANDOM" "$STATUS_JSON" || true)"
    if [[ "$http_code" == "200" ]]; then
      state="$(display_status)"
      loaded="$(worker_loaded)"
      echo "worker-update wait attempt=$attempt display_status=$state model_loaded=$loaded"
      if [[ "$state" == "READY" && "$loaded" == "true" ]]; then
        return 0
      fi
      if [[ "$state" == "ERROR" ]]; then
        cat "$STATUS_JSON"
        return 1
      fi
    fi
    sleep 5
  done
  echo "Timed out waiting for live Qwen workload to drain; worker restart was NOT performed." >&2
  cat "$STATUS_JSON" || true
  return 1
}

wait_for_worker_healthy() {
  local attempt
  local http_code
  local state
  local loaded
  for attempt in $(seq 1 180); do
    http_code="$(request GET "$SERVICE_URL/api/qwen-lab/gpu/status?healthy_ts=$RANDOM" "$STATUS_JSON" || true)"
    if [[ "$http_code" == "200" ]]; then
      state="$(display_status)"
      loaded="$(worker_loaded)"
      echo "worker-health attempt=$attempt display_status=$state model_loaded=$loaded"
      if [[ "$loaded" == "true" && ( "$state" == "READY" || "$state" == "BUSY" ) ]]; then
        return 0
      fi
      if [[ "$state" == "ERROR" ]]; then
        cat "$STATUS_JSON"
        return 1
      fi
    fi
    sleep 5
  done
  echo "Timed out waiting for healthy Qwen worker." >&2
  return 1
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
elif [[ "$INITIAL_STATUS" != "READY" && "$INITIAL_STATUS" != "LOADING" && "$INITIAL_STATUS" != "STARTING" && "$INITIAL_STATUS" != "BUSY" && "$INITIAL_STATUS" != "ERROR" ]]; then
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

STAGE_PATH="$REMOTE_PARENT/fish-qwen-refine-worker"
WORKER_CHANGED="$(remote_ssh "
set -euo pipefail
same=true
for pair in \
  '$STAGE_PATH/worker.py:/opt/fish-qwen-refine-worker/worker.py' \
  '$STAGE_PATH/config.yaml:/opt/fish-qwen-refine-worker/config.yaml' \
  '$STAGE_PATH/qwen2511_api.json:/opt/fish-qwen-refine-worker/qwen2511_api.json' \
  '$STAGE_PATH/fish-qwen-comfyui.service:/etc/systemd/system/fish-qwen-comfyui.service' \
  '$STAGE_PATH/fish-qwen-refine-worker.service:/etc/systemd/system/fish-qwen-refine-worker.service' \
  '$STAGE_PATH/wait-for-comfyui.sh:/opt/fish-qwen-refine-worker/wait-for-comfyui.sh'
do
  src=\${pair%%:*}
  dst=\${pair#*:}
  if [[ ! -f "\$dst" ]] || ! cmp -s "\$src" "\$dst"; then
    same=false
    break
  fi
done
if [[ "\$same" == "true" ]]; then echo false; else echo true; fi
")"
echo "worker_changed=$WORKER_CHANGED"

if [[ "$WORKER_CHANGED" == "true" ]]; then
  echo "Worker files changed. Waiting for live Image Studio/Qwen workload to drain before restart."
  wait_for_idle_before_restart

  remote_ssh "
set -euo pipefail
sudo install -d -o pan277942135 -g pan277942135 /opt/fish-qwen-refine-worker
sudo install -o pan277942135 -g pan277942135 -m 0644 $STAGE_PATH/worker.py /opt/fish-qwen-refine-worker/worker.py
sudo install -o pan277942135 -g pan277942135 -m 0644 $STAGE_PATH/config.yaml /opt/fish-qwen-refine-worker/config.yaml
sudo install -o pan277942135 -g pan277942135 -m 0644 $STAGE_PATH/qwen2511_api.json /opt/fish-qwen-refine-worker/qwen2511_api.json
sudo install -o root -g root -m 0644 $STAGE_PATH/fish-qwen-comfyui.service /etc/systemd/system/fish-qwen-comfyui.service
sudo install -o root -g root -m 0644 $STAGE_PATH/fish-qwen-refine-worker.service /etc/systemd/system/fish-qwen-refine-worker.service
sudo install -o pan277942135 -g pan277942135 -m 0755 $STAGE_PATH/wait-for-comfyui.sh /opt/fish-qwen-refine-worker/wait-for-comfyui.sh
sudo install -d -o pan277942135 -g pan277942135 /opt/fish-qwen-refine-worker/output
sudo systemctl daemon-reload
sudo systemctl enable fish-qwen-comfyui.service fish-qwen-refine-worker.service
sudo systemctl restart fish-qwen-comfyui.service
sudo systemctl restart fish-qwen-refine-worker.service
sudo systemctl is-enabled fish-qwen-comfyui.service fish-qwen-refine-worker.service
"
else
  echo "Worker files are unchanged. Skipping ComfyUI/Qwen restart to preserve live generation."
fi

wait_for_worker_healthy
echo "Qwen worker bootstrap is healthy. Restart performed only when worker files changed and queue was idle."

{
  echo "### Qwen VM boot dependency bootstrap"
  echo "- ComfyUI service installed: **PASS**"
  echo "- Qwen Worker Requires/After ComfyUI: **PASS**"
  echo "- Worker change detection: **PASS** (`$WORKER_CHANGED`)"
  echo "- Non-invasive live-queue restart policy: **PASS**"
  echo "- Worker health: **READY/BUSY with model loaded**"
  echo "- VM intentionally left RUNNING for the following page-driven UAT"
} >> "${GITHUB_STEP_SUMMARY:-/dev/stdout}"
