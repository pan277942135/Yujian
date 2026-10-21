#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-${GOOGLE_CLOUD_PROJECT:-}}"
if [[ -z "$PROJECT_ID" ]]; then
  echo "Set PROJECT_ID or GOOGLE_CLOUD_PROJECT to the reviewed target project." >&2
  exit 2
fi
REGION="${DETECTOR_GPU_REGION:-asia-southeast1}"
BUCKET="${BUCKET:-yujian-model-factory-571785698442}"
JOB_NAME="${DETECTOR_TRAINING_JOB_NAME:-yujian-detector-trainer}"
AR_REPOSITORY="${DETECTOR_AR_REPOSITORY:-yujian-detector-training}"
RUNTIME_SA="${RUNTIME_SA:-yujian-model-factory@${PROJECT_ID}.iam.gserviceaccount.com}"
DATASET_VERSION="${DETECTOR_DATASET_VERSION:-DET_DS_v0.3}"
MODEL_VERSION="${DETECTOR_MODEL_VERSION:-DET_FISH_v0.3}"
DATASET_ROOT="/tmp/yujian-detector/dataset/${DATASET_VERSION}"
GIT_SHA="$(git rev-parse HEAD 2>/dev/null || true)"

case "${DATASET_VERSION}:${MODEL_VERSION}" in
  DET_DS_v0.2:DET_FISH_v0.2|DET_DS_v0.3:DET_FISH_v0.3) ;;
  *) echo "Unsupported or mismatched dataset/model pair: ${DATASET_VERSION}:${MODEL_VERSION}" >&2; exit 2 ;;
esac
if [[ ! "${GIT_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "A full Git commit SHA is required; refusing to build an unpinned trainer." >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "Working tree is not clean; refusing to build bytes that do not match APP_GIT_COMMIT." >&2
  git status --short
  exit 2
fi

log() { printf '\n==> %s\n' "$*"; }

gcloud config set project "$PROJECT_ID" >/dev/null
# Required APIs are already enabled. The deployer deliberately lacks serviceusage.services.enable.

log "Verifying frozen ${DATASET_VERSION} artifacts before any trainer build"
PREFLIGHT_DIR="$(mktemp -d)"
trap 'rm -rf "$PREFLIGHT_DIR"' EXIT
gcloud storage cp "gs://${BUCKET}/detector-datasets/${DATASET_VERSION}/bootstrap_report.json" "${PREFLIGHT_DIR}/bootstrap_report.json"
gcloud storage cp "gs://${BUCKET}/detector-datasets/${DATASET_VERSION}/bootstrap_manifest.json" "${PREFLIGHT_DIR}/bootstrap_manifest.json"
ARCHIVE_SHA256="$(gcloud storage cat "gs://${BUCKET}/detector-datasets/${DATASET_VERSION}/${DATASET_VERSION}.tar.gz" | sha256sum | awk '{print $1}')"
DETECTOR_DATASET_VERSION="${DATASET_VERSION}" PYTHONPATH="$PWD" python3 - "${PREFLIGHT_DIR}/bootstrap_report.json" "${PREFLIGHT_DIR}/bootstrap_manifest.json" "${ARCHIVE_SHA256}" <<'PY'
import json
import sys
from trainer.detector_dataset_gate import validate_v02_dataset

with open(sys.argv[1], encoding="utf-8") as f:
    report = json.load(f)
with open(sys.argv[2], "rb") as f:
    manifest = f.read()
archive_sha = sys.argv[3]
result = validate_v02_dataset(report, manifest, archive_sha)
if report.get("status") != "FROZEN_READY_FOR_TRAINING" or not result.get("ready") or not result.get("counts_verified"):
    print("BLOCKED: frozen dataset gate did not pass", file=sys.stderr)
    print(json.dumps({"report_status": report.get("status"), **result}, ensure_ascii=False, indent=2), file=sys.stderr)
    raise SystemExit(1)
print("DATASET_GATE_PASS", json.dumps({
    "dataset_version": report.get("dataset_version"),
    "manifest_sha256": report.get("manifest_sha256"),
    "archive_sha256": archive_sha,
    "counts": result.get("counts"),
    "negative_split_counts": result.get("negative_split_counts"),
}, ensure_ascii=False))
PY

if ! gcloud artifacts repositories describe "$AR_REPOSITORY" --project "$PROJECT_ID" --location "$REGION" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$AR_REPOSITORY" \
    --project "$PROJECT_ID" \
    --location "$REGION" \
    --repository-format=docker \
    --description="YuJian YOLOX detector training images"
fi

IMAGE_TAG="${GIT_SHA}-$(date -u +%Y%m%d-%H%M%S)"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPOSITORY}/detector-trainer:${IMAGE_TAG}"

log "Building a fresh trainer image from exact source SHA ${GIT_SHA}"
gcloud builds submit . --project "$PROJECT_ID" --config infra/cloudbuild.detector.yaml --substitutions="_IMAGE_URI=${IMAGE_URI}"

IMAGE_DIGEST="$(gcloud artifacts docker images describe "$IMAGE_URI" --project "$PROJECT_ID" --format='value(image_summary.digest)')"
if [[ ! "$IMAGE_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "Unable to resolve an immutable detector trainer image digest for ${IMAGE_URI}; refusing Job deployment." >&2
  exit 3
fi
PINNED_IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPOSITORY}/detector-trainer@${IMAGE_DIGEST}"

log "Deploying L4 detector training job ${JOB_NAME} in ${REGION}"
# Cloud Run Jobs caps GPU task timeout at one hour. The detector dataset is small enough
# for the complete train → evaluate → ONNX export path to fit this platform limit.
gcloud run jobs deploy "$JOB_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --image "$PINNED_IMAGE_URI" \
  --service-account "$RUNTIME_SA" \
  --set-env-vars="GCS_BUCKET=${BUCKET},DETECTOR_DATASET_VERSION=${DATASET_VERSION},DETECTOR_DATASET_ROOT=${DATASET_ROOT},DETECTOR_MODEL_VERSION=${MODEL_VERSION},APP_GIT_COMMIT=${GIT_SHA},DETECTOR_TRAINER_IMAGE_DIGEST=${IMAGE_DIGEST},DETECTOR_EPOCHS=${DETECTOR_EPOCHS:-30},DETECTOR_BATCH_SIZE=${DETECTOR_BATCH_SIZE:-16},DETECTOR_SEED=${DETECTOR_SEED:-20261020},DETECTOR_EVAL_INTERVAL=${DETECTOR_EVAL_INTERVAL:-2}" \
  --cpu=4 \
  --memory=16Gi \
  --gpu=1 \
  --gpu-type=nvidia-l4 \
  --no-gpu-zonal-redundancy \
  --tasks=1 \
  --parallelism=1 \
  --max-retries=0 \
  --task-timeout=3600s \
  --quiet

printf 'Detector trainer ready: %s -> %s\n' "$DATASET_VERSION" "$MODEL_VERSION"

log "Verifying deployed trainer Job configuration"
gcloud run jobs describe "$JOB_NAME" --project "$PROJECT_ID" --region "$REGION" --format=json > "${PREFLIGHT_DIR}/job.json"
python3 - "${PREFLIGHT_DIR}/job.json" "${PINNED_IMAGE_URI}" "${GIT_SHA}" "${DATASET_VERSION}" "${MODEL_VERSION}" "${IMAGE_DIGEST}" <<'PY'
import json
import re
import sys

path, expected_image, expected_commit, expected_dataset, expected_model, expected_digest = sys.argv[1:]
with open(path, encoding="utf-8") as f:
    job = json.load(f)

task = (
    (((job.get("spec") or {}).get("template") or {}).get("spec") or {}).get("template") or {}
).get("spec") or {}
if not task:
    task = ((job.get("template") or {}).get("template") or {})
containers = task.get("containers") or []
if len(containers) != 1:
    raise SystemExit("BLOCKED: expected exactly one trainer container")
container = containers[0]
image = str(container.get("image") or "")
env = {
    str(item.get("name") or ""): str(item.get("value") or "")
    for item in (container.get("env") or [])
    if isinstance(item, dict)
}
limits = (container.get("resources") or {}).get("limits") or {}
selector = task.get("nodeSelector") or {}
checks = {
    "image_digest_pinned": bool(re.search(r"@sha256:[0-9a-f]{64}$", image)),
    "image_matches_deployed_digest": image == expected_image and image.endswith("@" + expected_digest),
    "APP_GIT_COMMIT": env.get("APP_GIT_COMMIT") == expected_commit,
    "DETECTOR_DATASET_VERSION": env.get("DETECTOR_DATASET_VERSION") == expected_dataset,
    "DETECTOR_MODEL_VERSION": env.get("DETECTOR_MODEL_VERSION") == expected_model,
    "DETECTOR_TRAINER_IMAGE_DIGEST": env.get("DETECTOR_TRAINER_IMAGE_DIGEST") == expected_digest,
    "gpu_count": str(limits.get("nvidia.com/gpu") or "") == "1",
    "gpu_type": str(selector.get("run.googleapis.com/accelerator") or "").lower() == "nvidia-l4",
    "cpu": float(str(limits.get("cpu") or "0").removesuffix("m")) >= 4 if not str(limits.get("cpu") or "").endswith("m") else float(str(limits.get("cpu"))[:-1]) >= 4000,
    "memory": str(limits.get("memory") or "").lower() in {"16gi", "16384mi", "17g"},
    "max_retries": int(task.get("maxRetries") or 0) == 0,
}
print(json.dumps({
    "job": job.get("metadata", {}).get("name") or job.get("name"),
    "image": image,
    "commit": env.get("APP_GIT_COMMIT"),
    "dataset_version": env.get("DETECTOR_DATASET_VERSION"),
    "model_version": env.get("DETECTOR_MODEL_VERSION"),
    "image_digest": expected_digest,
    "checks": checks,
}, ensure_ascii=False, indent=2))
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit("BLOCKED: deployed trainer Job verification failed: " + ", ".join(failed))
print("TRAINER_JOB_CONFIG_PASS")
PY

