#!/usr/bin/env bash
set -euo pipefail

# Explicitly opt-in end-to-end v0.3 trainer deployment and GPU execution.
# The production inference model version is not changed by this script.
if [[ "${CONFIRM_START_GPU:-}" != "YES" ]]; then
  echo "Set CONFIRM_START_GPU=YES to explicitly authorize GPU training." >&2
  exit 2
fi

PROJECT_ID="${PROJECT_ID:-gemini-api-project-503706}"
BUCKET="${BUCKET:-yujian-model-factory-571785698442}"
CONSOLE_REGION="${CONSOLE_REGION:-asia-east1}"
GPU_REGION="${DETECTOR_GPU_REGION:-asia-southeast1}"
CONSOLE_SERVICE="${CONSOLE_SERVICE:-yujian-model-factory-console}"
JOB_NAME="${DETECTOR_TRAINING_JOB_NAME:-yujian-detector-trainer}"
DATASET_VERSION="DET_DS_v0.3"
MODEL_VERSION="DET_FISH_v0.3"
GIT_SHA="$(git rev-parse HEAD 2>/dev/null || true)"

if [[ ! "${GIT_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "A full Git SHA is required; refusing deployment." >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "Working tree is not clean; refusing deployment of untracked bytes." >&2
  git status --short
  exit 2
fi

echo "========== Exact release target =========="
echo "Git SHA: ${GIT_SHA}"
echo "Dataset: ${DATASET_VERSION}"
echo "Model:   ${MODEL_VERSION}"
echo "Project: ${PROJECT_ID}"

echo ""
echo "========== Ensure target model version has no published final artifacts =========="
MODEL_OBJECTS="$(gcloud storage ls --recursive "gs://${BUCKET}/models/")"
if grep -Fq "/models/${MODEL_VERSION}/" <<<"${MODEL_OBJECTS}"; then
  echo "BLOCKED: ${MODEL_VERSION} already has objects. Refusing to overwrite immutable model artifacts." >&2
  grep -F "/models/${MODEL_VERSION}/" <<<"${MODEL_OBJECTS}" >&2
  exit 3
fi

echo ""
echo "========== Deploy console with v0.3 training target =========="
PROJECT_ID="${PROJECT_ID}" \
REGION="${CONSOLE_REGION}" \
SERVICE="${CONSOLE_SERVICE}" \
GCS_BUCKET="${BUCKET}" \
DETECTOR_DATASET_VERSION="${DATASET_VERSION}" \
DETECTOR_TRAINING_MODEL_VERSION="${MODEL_VERSION}" \
bash scripts/deploy_console_runtime.sh

echo ""
echo "========== Verify live console source and training target =========="
gcloud run services describe "${CONSOLE_SERVICE}" \
  --project="${PROJECT_ID}" --region="${CONSOLE_REGION}" --format=json \
  > /tmp/det_fish_v03_console_service.json
python3 - /tmp/det_fish_v03_console_service.json "${GIT_SHA}" "${DATASET_VERSION}" "${MODEL_VERSION}" <<'PY'
import json
import sys

path, expected_commit, expected_dataset, expected_model = sys.argv[1:]
with open(path, encoding="utf-8") as f:
    service = json.load(f)
template = ((service.get("spec") or {}).get("template") or {})
containers = ((template.get("spec") or {}).get("containers") or [])
if not containers:
    template = service.get("template") or {}
    containers = ((template.get("template") or {}).get("containers") or [])
if not containers:
    raise SystemExit("BLOCKED: could not inspect live console container")
env = {
    str(item.get("name") or ""): str(item.get("value") or "")
    for item in (containers[0].get("env") or [])
    if isinstance(item, dict)
}
expected = {
    "APP_GIT_COMMIT": expected_commit,
    "DETECTOR_DATASET_VERSION": expected_dataset,
    "DETECTOR_TRAINING_MODEL_VERSION": expected_model,
}
mismatches = {key: (env.get(key), value) for key, value in expected.items() if env.get(key) != value}
if mismatches:
    raise SystemExit("BLOCKED: live console configuration mismatch: " + json.dumps(mismatches))
print("CONSOLE_V03_CONFIG_PASS", json.dumps(expected))
PY

echo ""
echo "========== Build and deploy immutable GPU trainer =========="
PROJECT_ID="${PROJECT_ID}" \
DETECTOR_GPU_REGION="${GPU_REGION}" \
BUCKET="${BUCKET}" \
DETECTOR_DATASET_VERSION="${DATASET_VERSION}" \
DETECTOR_MODEL_VERSION="${MODEL_VERSION}" \
DETECTOR_TRAINING_JOB_NAME="${JOB_NAME}" \
bash scripts/deploy_detector_training_gcp.sh

echo ""
echo "========== Refuse duplicate active executions =========="
gcloud run jobs executions list \
  --job="${JOB_NAME}" --project="${PROJECT_ID}" --region="${GPU_REGION}" \
  --limit=100 --format=json > /tmp/det_fish_v03_executions_before.json
python3 - /tmp/det_fish_v03_executions_before.json <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    value = json.load(f)
items = value if isinstance(value, list) else value.get("executions", []) if isinstance(value, dict) else []
active = []
for item in items:
    status = item.get("status") or {}
    if not status.get("completionTime"):
        active.append(item.get("metadata", {}).get("name") or item.get("name") or "<unnamed>")
if active:
    raise SystemExit("BLOCKED: active or queued DET_FISH executions already exist: " + ", ".join(active))
print("PASS: no active or queued execution")
PY

echo ""
echo "========== Start DET_FISH_v0.3 GPU training =========="
gcloud run jobs execute "${JOB_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${GPU_REGION}" \
  --quiet

EXECUTION="$(gcloud run jobs executions list \
  --job="${JOB_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${GPU_REGION}" \
  --limit=1 \
  --sort-by='~metadata.creationTimestamp' \
  --format='value(metadata.name)')"
if [[ -z "${EXECUTION}" ]]; then
  echo "BLOCKED: Cloud Run did not return a discoverable execution ID." >&2
  exit 4
fi

echo "GPU execution ID: ${EXECUTION}"
gcloud run jobs executions describe "${EXECUTION}" \
  --job="${JOB_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${GPU_REGION}" \
  --format='yaml(status.conditions,status.completionTime,status.runningCount,status.succeededCount,status.failedCount)'

echo ""
echo "Training has been submitted. This script does not promote DET_FISH_v0.3 to production."
echo "Inspect the execution and the models/${MODEL_VERSION}/ artifacts before any promotion."
