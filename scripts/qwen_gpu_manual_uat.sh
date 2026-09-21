#!/usr/bin/env bash
set -euo pipefail

: "${SERVICE_URL:?SERVICE_URL is required}"
: "${PROJECT_ID:?PROJECT_ID is required}"
: "${GPU_PROJECT_ID:?GPU_PROJECT_ID is required}"
: "${GPU_ZONE:?GPU_ZONE is required}"
: "${GPU_INSTANCE:?GPU_INSTANCE is required}"

OUT_DIR="${RUNNER_TEMP:-/tmp}/qwen-gpu-manual-uat"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
COOKIE_JAR="$OUT_DIR/cookies.txt"
STATUS_JSON="$OUT_DIR/status.json"

START_REQUESTED=false
WORKER_URL="\${QWEN_WORKER_BASE_URL:-http://34.69.75.199:8002}"

diagnose_worker() {
  echo "### Qwen worker diagnostic (read-only)"
  echo "--- runner -> worker /health ---"
  curl --connect-timeout 10 --max-time 30 -sS "\${WORKER_URL%/}/health" || true
  echo
  echo "--- GCE instance status and external IP ---"
  gcloud compute instances describe "$GPU_INSTANCE" \
    --project "$GPU_PROJECT_ID" --zone "$GPU_ZONE" \
    --format='value(status,networkInterfaces[0].accessConfigs[0].natIP)' || true
  echo
  echo "--- VM systemd/journal/GPU ---"
  timeout 120s gcloud compute ssh "$GPU_INSTANCE" \
    --project "$GPU_PROJECT_ID" --zone "$GPU_ZONE" --quiet \
    --command='
      echo "health_local="
      curl --connect-timeout 5 --max-time 20 -sS http://127.0.0.1:8002/health || true
      echo
      echo "systemd_enabled="
      systemctl is-enabled fish-qwen-refine-worker || true
      echo "systemd_active="
      systemctl is-active fish-qwen-refine-worker || true
      echo "systemd_status="
      sudo systemctl status fish-qwen-refine-worker --no-pager -l || true
      echo "journal="
      sudo journalctl -u fish-qwen-refine-worker -n 120 --no-pager || true
      echo "nvidia_smi="
      nvidia-smi || true
    ' 2>&1 || true
}

cleanup_uat_vm() {
  local rc=$?
  if [[ "$START_REQUESTED" == "true" && "$rc" -ne 0 ]]; then
    diagnose_worker
    echo "UAT failed; stopping the exact test VM to leave it TERMINATED"
    gcloud compute instances stop "$GPU_INSTANCE" \
      --project "$GPU_PROJECT_ID" --zone "$GPU_ZONE" --quiet || true
    for _ in $(seq 1 36); do
      VM_STATUS="$(gcloud compute instances describe "$GPU_INSTANCE" \
        --project "$GPU_PROJECT_ID" --zone "$GPU_ZONE" --format='value(status)' 2>/dev/null || true)"
      echo "failure-cleanup vm_status=$VM_STATUS"
      [[ "$VM_STATUS" == "TERMINATED" ]] && break
      sleep 5
    done
  fi
  trap - EXIT
  exit "$rc"
}
trap cleanup_uat_vm EXIT


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
    payload = json.load(handle)
print(str(payload.get("display_status") or "ERROR").upper())
PY
}

wait_for_display() {
  local expected="$1"
  local attempts="$2"
  local label="$3"
  local attempt
  local http_code
  local current
  for attempt in $(seq 1 "$attempts"); do
    http_code="$(request GET "$SERVICE_URL/api/qwen-lab/gpu/status?uat_ts=$RANDOM" "$STATUS_JSON")"
    test "$http_code" = "200"
    current="$(display_status)"
    echo "$label attempt=$attempt display_status=$current"
    if [[ "$current" == "$expected" ]]; then
      return 0
    fi
    if [[ "$current" == "ERROR" ]]; then
      cat "$STATUS_JSON"
      return 1
    fi
    sleep 5
  done
  echo "Timed out waiting for $expected"
  cat "$STATUS_JSON"
  return 1
}

CONSOLE_KEY="$(gcloud secrets versions access latest --secret=yujian-console-access-key --project="$PROJECT_ID")"
test -n "$CONSOLE_KEY"
echo "::add-mask::$CONSOLE_KEY"
LOGIN_HTTP="$(curl --retry 3 --retry-all-errors --retry-delay 2   --connect-timeout 10 --max-time 30 -sS   -b "$COOKIE_JAR" -c "$COOKIE_JAR" -o /dev/null -w '%{http_code}'   -X POST "$SERVICE_URL/login" --data-urlencode "access_key=$CONSOLE_KEY")"
test "$LOGIN_HTTP" = "303"
unset CONSOLE_KEY

wait_for_display READY 12 "initial-ready"

STOP_JSON="$OUT_DIR/stop.json"
STOP_HTTP="$(request POST "$SERVICE_URL/api/qwen-lab/gpu/stop" "$STOP_JSON")"
test "$STOP_HTTP" = "200"
python3 - "$STOP_JSON" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
assert payload.get("accepted") is True, payload
assert payload.get("display_status") == "STOPPING", payload
PY
wait_for_display STOPPED 48 "stop"
VM_STATUS="$(gcloud compute instances describe "$GPU_INSTANCE"   --project "$GPU_PROJECT_ID" --zone "$GPU_ZONE" --format='value(status)')"
test "$VM_STATUS" = "TERMINATED"

START_JSON="$OUT_DIR/start.json"
START_HTTP="$(request POST "$SERVICE_URL/api/qwen-lab/gpu/start" "$START_JSON")"
START_REQUESTED=true
test "$START_HTTP" = "200"
python3 - "$START_JSON" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
assert payload.get("accepted") is True, payload
assert payload.get("display_status") == "STARTING", payload
PY
wait_for_display READY 180 "start-worker-ready"
VM_STATUS="$(gcloud compute instances describe "$GPU_INSTANCE"   --project "$GPU_PROJECT_ID" --zone "$GPU_ZONE" --format='value(status)')"
test "$VM_STATUS" = "RUNNING"

DATASETS_JSON="$OUT_DIR/datasets.json"
DATASETS_HTTP="$(request GET "$SERVICE_URL/api/platform/datasets" "$DATASETS_JSON")"
test "$DATASETS_HTTP" = "200"
read -r DATASET_ID DATASET_ITEM_ID < <(python3 - "$DATASETS_JSON" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
rows = payload if isinstance(payload, list) else payload.get("datasets") or []
frozen = [row for row in rows if str(row.get("status") or "").upper() == "FROZEN"]
if not frozen:
    raise SystemExit("no frozen dataset available")
dataset = frozen[0]
print(str(dataset.get("id") or dataset.get("dataset_id") or dataset.get("dataset_version") or ""), end=" ")
PY
)"
test -n "$DATASET_ID"
ITEMS_JSON="$OUT_DIR/items.json"
ITEMS_HTTP="$(request GET "$SERVICE_URL/api/platform/datasets/$DATASET_ID/items?page=1&size=10" "$ITEMS_JSON")"
test "$ITEMS_HTTP" = "200"
read -r DATASET_ID DATASET_ITEM_ID < <(python3 - "$DATASETS_JSON" "$ITEMS_JSON" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    datasets = json.load(handle)
with open(sys.argv[2], encoding="utf-8") as handle:
    items = json.load(handle)
rows = datasets if isinstance(datasets, list) else datasets.get("datasets") or []
frozen = [row for row in rows if str(row.get("status") or "").upper() == "FROZEN"]
dataset = frozen[0]
dataset_id = str(dataset.get("id") or dataset.get("dataset_id") or dataset.get("dataset_version") or "")
item_rows = items.get("items") if isinstance(items, dict) else items
if not item_rows:
    raise SystemExit("frozen dataset has no items")
item = item_rows[0]
item_id = str(item.get("id") or item.get("image_id") or "")
if not item_id:
    raise SystemExit("dataset item has no id")
print(dataset_id, item_id)
PY
)"
test -n "$DATASET_ID"
test -n "$DATASET_ITEM_ID"

GENERATE_JSON="$OUT_DIR/generate.json"
GENERATE_HTTP_FILE="$OUT_DIR/generate.http"
(
  curl --retry 2 --retry-all-errors --retry-delay 2     --connect-timeout 10 --max-time 1500 -sS     -b "$COOKIE_JAR" -c "$COOKIE_JAR"     -o "$GENERATE_JSON" -w '%{http_code}'     -X POST "$SERVICE_URL/api/fish-portrait/qwen-lab/generate"     -F "dataset_id=$DATASET_ID"     -F "dataset_item_id=$DATASET_ITEM_ID"
) > "$GENERATE_HTTP_FILE" 2>&1 &
GENERATE_PID=$!

BUSY_SEEN=false
for attempt in $(seq 1 18); do
  if ! kill -0 "$GENERATE_PID" 2>/dev/null; then
    break
  fi
  STATUS_HTTP="$(request GET "$SERVICE_URL/api/qwen-lab/gpu/status?uat_ts=$RANDOM" "$STATUS_JSON")"
  test "$STATUS_HTTP" = "200"
  CURRENT_STATUS="$(display_status)"
  echo "generate attempt=$attempt display_status=$CURRENT_STATUS"
  if [[ "$CURRENT_STATUS" == "BUSY" ]]; then
    BUSY_SEEN=true
    BUSY_STOP_JSON="$OUT_DIR/busy-stop.json"
    BUSY_STOP_HTTP="$(request POST "$SERVICE_URL/api/qwen-lab/gpu/stop" "$BUSY_STOP_JSON")"
    test "$BUSY_STOP_HTTP" = "409"
    python3 - "$BUSY_STOP_JSON" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
detail = payload.get("detail") or {}
assert detail.get("error_code") == "QWEN_GPU_BUSY", payload
PY
    break
  fi
  sleep 5
done
test "$BUSY_SEEN" = "true"

set +e
wait "$GENERATE_PID"
GENERATE_WAIT_RC=$?
set -e
test "$GENERATE_WAIT_RC" -eq 0
GENERATE_HTTP="$(tr -d '
' < "$GENERATE_HTTP_FILE")"
test "$GENERATE_HTTP" = "200"
python3 - "$GENERATE_JSON" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
assert payload.get("status") == "SUCCESS", payload
assert payload.get("run_id"), payload
assert payload.get("output_image_url") or payload.get("output_image_uri"), payload
PY
wait_for_display READY 24 "post-generate"

FINAL_STOP_JSON="$OUT_DIR/final-stop.json"
FINAL_STOP_HTTP="$(request POST "$SERVICE_URL/api/qwen-lab/gpu/stop" "$FINAL_STOP_JSON")"
test "$FINAL_STOP_HTTP" = "200"
python3 - "$FINAL_STOP_JSON" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
assert payload.get("accepted") is True, payload
assert payload.get("display_status") == "STOPPING", payload
PY
wait_for_display STOPPED 48 "final-stop"
VM_STATUS="$(gcloud compute instances describe "$GPU_INSTANCE"   --project "$GPU_PROJECT_ID" --zone "$GPU_ZONE" --format='value(status)')"
test "$VM_STATUS" = "TERMINATED"

{
  echo "### Qwen Lab GPU Manual Control Runtime UAT"
  echo "- STOPPED -> STARTING -> LOADING -> READY: **PASS**"
  echo "- READY -> STOPPING -> STOPPED: **PASS**"
  echo "- BUSY blocks stop: **PASS**"
  echo "- Dataset Qwen generation: **PASS**"
  echo "- Final VM status: `$VM_STATUS`"
} >> "$GITHUB_STEP_SUMMARY"
