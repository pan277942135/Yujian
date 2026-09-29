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
WORKER_URL="${QWEN_WORKER_BASE_URL:-http://34.69.75.199:8002}"

diagnose_worker() {
  echo "### Qwen worker diagnostic (read-only)"
  echo "--- runner -> worker /health ---"
  curl --connect-timeout 10 --max-time 30 -sS "${WORKER_URL%/}/health" || true
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
  if [[ "$rc" -ne 0 ]]; then
    diagnose_worker
    echo "UAT failed; persistent GPU policy keeps the VM unchanged for diagnosis and interactive use."
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


fetch_identity_fixture() {
  local uri="$1"
  local destination="$2"
  if [[ "$uri" == gs://* ]]; then
    gcloud storage cp "$uri" "$destination"
  elif [[ "$uri" == http://* || "$uri" == https://* ]]; then
    curl --retry 3 --retry-all-errors --retry-delay 2 \
      --connect-timeout 10 --max-time 120 -fsS "$uri" -o "$destination"
  elif [[ -f "$uri" ]]; then
    cp "$uri" "$destination"
  else
    echo "Unsupported or missing identity UAT fixture: $uri" >&2
    return 1
  fi
  test -s "$destination"
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

collect_boot_evidence() {
  echo "### VM boot evidence"
  timeout 120s gcloud compute ssh "$GPU_INSTANCE"     --project "$GPU_PROJECT_ID" --zone "$GPU_ZONE" --quiet     --command='
      echo "comfy_active=$(systemctl is-active fish-qwen-comfyui.service || true)"
      echo "comfy_enabled=$(systemctl is-enabled fish-qwen-comfyui.service || true)"
      echo "worker_active=$(systemctl is-active fish-qwen-refine-worker.service || true)"
      echo "worker_enabled=$(systemctl is-enabled fish-qwen-refine-worker.service || true)"
      echo "comfy_active_timestamp=$(systemctl show -p ActiveEnterTimestamp --value fish-qwen-comfyui.service || true)"
      echo "worker_active_timestamp=$(systemctl show -p ActiveEnterTimestamp --value fish-qwen-refine-worker.service || true)"
      ss -lntp | grep -E ":8188|:8002" || true
      sudo journalctl -u fish-qwen-comfyui.service -u fish-qwen-refine-worker.service -n 160 --no-pager | grep -E "ComfyUI ready|Loading Qwen|Qwen warmup started|Model loaded successfully|Worker ready|warmup failed" || true
      curl --connect-timeout 5 --max-time 20 -sS http://127.0.0.1:8002/health || true
    ' 2>&1 || true
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

wait_for_worker_healthy() {
  local attempts="$1"
  local label="$2"
  local attempt
  local http_code
  local current
  local model_loaded
  for attempt in $(seq 1 "$attempts"); do
    http_code="$(request GET "$SERVICE_URL/api/qwen-lab/gpu/status?uat_ts=$RANDOM" "$STATUS_JSON")"
    test "$http_code" = "200"
    current="$(display_status)"
    model_loaded="$(python3 - "$STATUS_JSON" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
print("true" if payload.get("model_loaded") is True else "false")
PY
)"
    echo "$label attempt=$attempt display_status=$current model_loaded=$model_loaded"
    if [[ "$model_loaded" == "true" && ( "$current" == "READY" || "$current" == "BUSY" ) ]]; then
      return 0
    fi
    if [[ "$current" == "ERROR" ]]; then
      cat "$STATUS_JSON"
      return 1
    fi
    sleep 5
  done
  echo "Timed out waiting for healthy Qwen worker (READY or BUSY with model_loaded=true)"
  cat "$STATUS_JSON"
  return 1
}

CONSOLE_KEY="$(gcloud secrets versions access latest --secret=yujian-console-access-key --project="$PROJECT_ID")"
test -n "$CONSOLE_KEY"
echo "::add-mask::$CONSOLE_KEY"
LOGIN_HTTP="$(curl --retry 3 --retry-all-errors --retry-delay 2   --connect-timeout 10 --max-time 30 -sS   -b "$COOKIE_JAR" -c "$COOKIE_JAR" -o /dev/null -w '%{http_code}'   -X POST "$SERVICE_URL/login" --data-urlencode "access_key=$CONSOLE_KEY")"
test "$LOGIN_HTTP" = "303"
unset CONSOLE_KEY

INITIAL_STATUS_HTTP="$(request GET "$SERVICE_URL/api/qwen-lab/gpu/status?uat_initial=$RANDOM" "$STATUS_JSON")"
test "$INITIAL_STATUS_HTTP" = "200"
INITIAL_DISPLAY="$(display_status)"
echo "initial display_status=$INITIAL_DISPLAY"
if [[ "$INITIAL_DISPLAY" == "READY" ]]; then
  echo "initial GPU is already READY; persistent GPU policy leaves it running"
elif [[ "$INITIAL_DISPLAY" == "STOPPED" ]]; then
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
elif [[ "$INITIAL_DISPLAY" == "STARTING" || "$INITIAL_DISPLAY" == "LOADING" || "$INITIAL_DISPLAY" == "BUSY" ]]; then
  wait_for_display READY 180 "initial-ready"
else
  cat "$STATUS_JSON"
  echo "Unexpected initial GPU state: $INITIAL_DISPLAY" >&2
  exit 1
fi
collect_boot_evidence
VM_STATUS="$(gcloud compute instances describe "$GPU_INSTANCE"   --project "$GPU_PROJECT_ID" --zone "$GPU_ZONE" --format='value(status)')"
test "$VM_STATUS" = "RUNNING"

DATASETS_JSON="$OUT_DIR/datasets.json"
DATASETS_HTTP="$(request GET "$SERVICE_URL/api/platform/datasets" "$DATASETS_JSON")"
test "$DATASETS_HTTP" = "200"
DATASET_ID="$(python3 - "$DATASETS_JSON" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
rows = payload if isinstance(payload, list) else payload.get("datasets") or []
frozen = [
    row for row in rows
    if isinstance(row, dict) and str(row.get("status") or "").upper() == "FROZEN"
]
if not frozen:
    raise SystemExit("no frozen dataset available")
dataset = frozen[0]
dataset_id = str(
    dataset.get("id")
    or dataset.get("dataset_id")
    or dataset.get("dataset_version")
    or ""
).strip()
if not dataset_id:
    raise SystemExit("frozen dataset has no id")
print(dataset_id)
PY
)"
test -n "${DATASET_ID:-}"
test -n "$DATASET_ID"
ITEMS_JSON="$OUT_DIR/items.json"
ITEMS_HTTP="$(request GET "$SERVICE_URL/api/platform/datasets/$DATASET_ID/items?page=1&size=10" "$ITEMS_JSON")"
test "$ITEMS_HTTP" = "200"
DATASET_ITEM_ID="$(python3 - "$ITEMS_JSON" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
item_rows = (
    payload.get("items") or payload.get("data") or []
    if isinstance(payload, dict)
    else payload
)
if not item_rows:
    raise SystemExit("frozen dataset has no items")
item = item_rows[0]
if not isinstance(item, dict):
    raise SystemExit("dataset item is not an object")
item_id = str(item.get("id") or item.get("image_id") or "").strip()
if not item_id:
    raise SystemExit("dataset item has no id")
print(item_id)
PY
)"
test -n "${DATASET_ITEM_ID:-}"
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
wait_for_worker_healthy 24 "post-generate"

# Image Studio V1 runtime gate: reuse the successful Qwen Lab result as both
# Base and an IDENTITY reference. This proves the new worker mode, reference
# multipart protocol, isolated image_studio_run persistence and media route on
# the exact deployed L4 worker before the VM is stopped.
IMAGE_STUDIO_BASE="$OUT_DIR/image_studio_base.png"
IMAGE_STUDIO_RUN_JSON="$OUT_DIR/image_studio_run.json"
QWEN_OUTPUT_URL="$(python3 - "$GENERATE_JSON" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
value = str(payload.get("output_image_url") or "").strip()
if not value:
    raise SystemExit("Qwen Lab did not return output_image_url")
print(value)
PY
)"
curl --retry 3 --retry-all-errors --retry-delay 2 \
  --connect-timeout 10 --max-time 120 -fsS \
  -b "$COOKIE_JAR" -c "$COOKIE_JAR" \
  "$SERVICE_URL$QWEN_OUTPUT_URL" -o "$IMAGE_STUDIO_BASE"

IMAGE_STUDIO_HTTP="$(
  curl --retry 2 --retry-all-errors --retry-delay 2 \
    --connect-timeout 10 --max-time 180 -sS \
    -b "$COOKIE_JAR" -c "$COOKIE_JAR" \
    -o "$IMAGE_STUDIO_RUN_JSON" -w '%{http_code}' \
    -X POST "$SERVICE_URL/api/image-studio/v1/edit" \
    -F "base_image=@$IMAGE_STUDIO_BASE;type=image/png" \
    -F "references=@$IMAGE_STUDIO_BASE;type=image/png" \
    -F 'reference_roles=["IDENTITY"]' \
    -F "mode=IDENTITY_LOCK" \
    -F "preservation=MAX" \
    -F "steps=4" \
    -F "resolution_mode=current" \
    -F "prompt=Keep the same subject identity and preserve the base composition. This is a runtime protocol verification."
)"
test "$IMAGE_STUDIO_HTTP" = "200"
IMAGE_STUDIO_RUN_ID="$(python3 - "$IMAGE_STUDIO_RUN_JSON" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
assert payload.get("status") in {"QUEUED", "RUNNING", "SUCCESS"}, payload
assert payload.get("storage_type") == "IMAGE_STUDIO_V1", payload
assert payload.get("mode") == "IDENTITY_LOCK", payload
assert payload.get("reference_roles") == ["IDENTITY"], payload
queue = payload.get("queue") or {}
assert queue.get("concurrency") == 1, payload
assert queue.get("policy") == "FIFO_SINGLE_L4", payload
run_id = str(payload.get("run_id") or "").strip()
if not run_id:
    raise SystemExit("Image Studio did not return run_id")
print(run_id)
PY
)"
test -n "$IMAGE_STUDIO_RUN_ID"

IMAGE_STUDIO_READBACK="$OUT_DIR/image_studio_run_readback.json"
IMAGE_STUDIO_FINAL_STATUS=""
for attempt in $(seq 1 240); do
  READBACK_HTTP="$(request GET "$SERVICE_URL/api/image-studio/v1/runs/$IMAGE_STUDIO_RUN_ID?uat_ts=$RANDOM" "$IMAGE_STUDIO_READBACK")"
  test "$READBACK_HTTP" = "200"
  IMAGE_STUDIO_FINAL_STATUS="$(python3 - "$IMAGE_STUDIO_READBACK" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
print(str(payload.get("status") or "").upper())
PY
)"
  echo "image-studio queue attempt=$attempt status=$IMAGE_STUDIO_FINAL_STATUS"
  if [[ "$IMAGE_STUDIO_FINAL_STATUS" == "SUCCESS" ]]; then
    break
  fi
  if [[ "$IMAGE_STUDIO_FINAL_STATUS" == "FAILED" ]]; then
    cat "$IMAGE_STUDIO_READBACK"
    exit 1
  fi
  sleep 5
done
test "$IMAGE_STUDIO_FINAL_STATUS" = "SUCCESS"
python3 - "$IMAGE_STUDIO_READBACK" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
assert payload.get("status") == "SUCCESS", payload
assert payload.get("storage_type") == "IMAGE_STUDIO_V1", payload
assert payload.get("output_image_url"), payload
assert payload.get("queue_position") is None, payload
PY
wait_for_worker_healthy 24 "post-image-studio"

# Strict Identity Transfer V2 visual evidence gate.
# This gate intentionally requires two distinct, explicitly configured human
# fixtures. Without them we do NOT claim visual fidelity PASS.
IDENTITY_UAT_BASE_URI="${IMAGE_STUDIO_IDENTITY_UAT_BASE_URI:-}"
IDENTITY_UAT_REFERENCE_URI="${IMAGE_STUDIO_IDENTITY_UAT_REFERENCE_URI:-}"
IDENTITY_UAT_STEPS="${IMAGE_STUDIO_IDENTITY_UAT_STEPS:-25}"
IDENTITY_V2_STATUS="NOT_EVALUATED"
IDENTITY_VISUAL_STATUS="NOT_EVALUATED"

if [[ -n "$IDENTITY_UAT_BASE_URI" && -n "$IDENTITY_UAT_REFERENCE_URI" ]]; then
  IDENTITY_EVIDENCE_DIR="$OUT_DIR/identity-transfer-v2"
  mkdir -p "$IDENTITY_EVIDENCE_DIR"
  IDENTITY_BASE="$IDENTITY_EVIDENCE_DIR/01_base_input.png"
  IDENTITY_REFERENCE="$IDENTITY_EVIDENCE_DIR/02_identity_reference.png"

  fetch_identity_fixture "$IDENTITY_UAT_BASE_URI" "$IDENTITY_BASE"
  fetch_identity_fixture "$IDENTITY_UAT_REFERENCE_URI" "$IDENTITY_REFERENCE"

  BASE_SHA="$(sha256sum "$IDENTITY_BASE" | awk '{print $1}')"
  REFERENCE_SHA="$(sha256sum "$IDENTITY_REFERENCE" | awk '{print $1}')"
  test "$BASE_SHA" != "$REFERENCE_SHA"
  {
    echo "base_sha256=$BASE_SHA"
    echo "identity_sha256=$REFERENCE_SHA"
  } > "$IDENTITY_EVIDENCE_DIR/fixture_hashes.txt"

  IDENTITY_RUN_JSON="$IDENTITY_EVIDENCE_DIR/03_submit.json"
  IDENTITY_HTTP="$(
    curl --retry 2 --retry-all-errors --retry-delay 2 \
      --connect-timeout 10 --max-time 180 -sS \
      -b "$COOKIE_JAR" -c "$COOKIE_JAR" \
      -o "$IDENTITY_RUN_JSON" -w '%{http_code}' \
      -X POST "$SERVICE_URL/api/image-studio/v1/edit" \
      -F "base_image=@$IDENTITY_BASE" \
      -F "references=@$IDENTITY_REFERENCE" \
      -F 'reference_roles=["IDENTITY"]' \
      -F "mode=STRICT_HEAD_SWAP" \
      -F "preservation=MAX" \
      -F "identity_strength=HIGH" \
      -F "head_edit_tightness=MEDIUM" \
      -F "steps=$IDENTITY_UAT_STEPS" \
      -F "resolution_mode=current" \
      -F "prompt=Replace only the head identity with the configured Identity B. Preserve every non-head Base pixel."
  )"
  test "$IDENTITY_HTTP" = "200"

  IDENTITY_RUN_ID="$(python3 - "$IDENTITY_RUN_JSON" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
assert payload.get("mode") == "STRICT_HEAD_SWAP", payload
assert payload.get("pipeline_version") == "identity_transfer_v2", payload
assert payload.get("reference_roles") == ["IDENTITY"], payload
run_id = str(payload.get("run_id") or "").strip()
if not run_id:
    raise SystemExit("strict identity V2 submit did not return run_id")
print(run_id)
PY
)"
  test -n "$IDENTITY_RUN_ID"

  IDENTITY_READBACK="$IDENTITY_EVIDENCE_DIR/04_final_readback.json"
  IDENTITY_FINAL_STATUS=""
  for attempt in $(seq 1 240); do
    IDENTITY_READBACK_HTTP="$(request GET "$SERVICE_URL/api/image-studio/v1/runs/$IDENTITY_RUN_ID?uat_ts=$RANDOM" "$IDENTITY_READBACK")"
    test "$IDENTITY_READBACK_HTTP" = "200"
    IDENTITY_FINAL_STATUS="$(python3 - "$IDENTITY_READBACK" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
print(str(payload.get("status") or "").upper())
PY
)"
    echo "strict-identity-v2 attempt=$attempt status=$IDENTITY_FINAL_STATUS"
    if [[ "$IDENTITY_FINAL_STATUS" == "SUCCESS" ]]; then
      break
    fi
    if [[ "$IDENTITY_FINAL_STATUS" == "FAILED" ]]; then
      cat "$IDENTITY_READBACK"
      exit 1
    fi
    sleep 5
  done
  test "$IDENTITY_FINAL_STATUS" = "SUCCESS"

  python3 - "$IDENTITY_READBACK" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
assert payload.get("status") == "SUCCESS", payload
assert payload.get("mode") == "STRICT_HEAD_SWAP", payload
assert payload.get("pipeline_version") == "identity_transfer_v2", payload
geometry = payload.get("strict_geometry") or {}
assert len(geometry.get("base_head_box") or []) == 4, payload
validation = payload.get("strict_validation") or {}
assert validation.get("outside_roi_preserved") is True, validation
assert int(validation.get("outside_roi_changed_pixels") or 0) == 0, validation
assert float(validation.get("head_roi_mean_abs_diff") or 0.0) > 0.25, validation
stages = {item.get("name"): item.get("status") for item in payload.get("stages") or [] if isinstance(item, dict)}
for required in ("AUTO_CROP", "AUTO_MASK", "HEAD_SWAP", "COMPOSITE"):
    assert stages.get(required) == "DONE", (required, stages)
assets = {item.get("kind") for item in payload.get("intermediate_assets") or [] if isinstance(item, dict)}
required_assets = {
    "base_face_crop",
    "base_head_crop",
    "identity_face_crop",
    "identity_head_crop",
    "mask_binary",
    "mask_preview",
    "edited_head_roi",
}
missing = sorted(required_assets - assets)
assert not missing, missing
assert payload.get("output_image_url"), payload
PY

  for spec in \
    "base_face_crop:05_base_face_crop.png" \
    "base_head_crop:06_base_head_crop.png" \
    "identity_face_crop:07_identity_face_crop.png" \
    "identity_head_crop:08_identity_head_crop.png" \
    "mask_binary:09_mask_binary.png" \
    "mask_preview:10_mask_preview.png" \
    "edited_head_roi:11_edited_head_roi.png" \
    "output:12_final_output.png"; do
    kind="${spec%%:*}"
    filename="${spec#*:}"
    curl --retry 3 --retry-all-errors --retry-delay 2 \
      --connect-timeout 10 --max-time 120 -fsS \
      -b "$COOKIE_JAR" -c "$COOKIE_JAR" \
      "$SERVICE_URL/api/image-studio/v1/runs/$IDENTITY_RUN_ID/media/$kind" \
      -o "$IDENTITY_EVIDENCE_DIR/$filename"
  done

  python3 - "$IDENTITY_READBACK" "$IDENTITY_EVIDENCE_DIR/13_visual_review.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
review = {
    "gate": "STRICT_IDENTITY_TRANSFER_V2",
    "protocol_geometry": "PASS",
    "visual_identity_fidelity": "REVIEW_REQUIRED",
    "run_id": payload.get("run_id"),
    "mode": payload.get("mode"),
    "strict_validation": payload.get("strict_validation"),
    "required_human_checks": [
        "Final face is unmistakably Identity B",
        "Base clothing/body/pose remain unchanged outside head ROI",
        "No hybrid/averaged face",
        "Hair/neck boundary is visually natural",
    ],
}
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump(review, handle, ensure_ascii=False, indent=2)
PY

  IDENTITY_V2_STATUS="PASS"
  IDENTITY_VISUAL_STATUS="REVIEW_REQUIRED"
  wait_for_worker_healthy 24 "post-strict-identity-v2"
fi

VM_STATUS="$(gcloud compute instances describe "$GPU_INSTANCE"   --project "$GPU_PROJECT_ID" --zone "$GPU_ZONE" --format='value(status)')"
test "$VM_STATUS" = "RUNNING"

{
  echo "### Qwen / Image Studio persistent GPU Runtime UAT"
  echo "- STOPPED (if needed) -> STARTING -> LOADING -> READY: **PASS**"
  echo "- BUSY blocks manual stop: **PASS**"
  echo "- Dataset Qwen generation: **PASS**"
  echo "- Image Studio durable FIFO queue + Identity Lock protocol runtime: **PASS**"
  echo "- Strict Identity Transfer V2 protocol/geometry: **$IDENTITY_V2_STATUS**"
  echo "- Strict Identity Transfer V2 visual fidelity: **$IDENTITY_VISUAL_STATUS**"
  echo "- Post-Image Studio worker health accepts READY/BUSY when model is loaded: **PASS**"
  echo "- Automatic GPU stop during UAT: **DISABLED**"
  echo "- Final VM status: `$VM_STATUS` (expected RUNNING)"
} >> "$GITHUB_STEP_SUMMARY"
