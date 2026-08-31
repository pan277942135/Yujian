#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-gemini-api-project-503706}"
REGION="${DETECTOR_GPU_REGION:-asia-southeast1}"
BUCKET="${BUCKET:-yujian-model-factory-571785698442}"
JOB_NAME="${DETECTOR_TRAINING_JOB_NAME:-yujian-detector-trainer}"
AR_REPOSITORY="${DETECTOR_AR_REPOSITORY:-yujian-detector-training}"
RUNTIME_SA="${RUNTIME_SA:-yujian-model-factory@${PROJECT_ID}.iam.gserviceaccount.com}"
DATASET_VERSION="${DETECTOR_DATASET_VERSION:-DET_DS_v0.1}"
MODEL_VERSION="${DETECTOR_MODEL_VERSION:-DET_FISH_v0.1}"

log() { printf '\n==> %s\n' "$*"; }

gcloud config set project "$PROJECT_ID" >/dev/null
# Required APIs are already enabled. The deployer deliberately lacks serviceusage.services.enable.

if ! gcloud artifacts repositories describe "$AR_REPOSITORY" --project "$PROJECT_ID" --location "$REGION" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$AR_REPOSITORY" \
    --project "$PROJECT_ID" \
    --location "$REGION" \
    --repository-format=docker \
    --description="YuJian YOLOX detector training images"
fi

GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPOSITORY}/detector-trainer:${GIT_SHA}"

log "Building/reusing detector trainer image ${IMAGE_URI}"
if ! gcloud artifacts docker images describe "$IMAGE_URI" --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud builds submit . --project "$PROJECT_ID" --config infra/cloudbuild.detector.yaml --substitutions="_IMAGE_URI=${IMAGE_URI}"
fi

log "Deploying L4 detector training job ${JOB_NAME} in ${REGION}"
gcloud run jobs deploy "$JOB_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --image "$IMAGE_URI" \
  --service-account "$RUNTIME_SA" \
  --set-env-vars="GCS_BUCKET=${BUCKET},DETECTOR_DATASET_VERSION=${DATASET_VERSION},DETECTOR_MODEL_VERSION=${MODEL_VERSION},APP_GIT_COMMIT=${GIT_SHA},DETECTOR_EPOCHS=${DETECTOR_EPOCHS:-30},DETECTOR_BATCH_SIZE=${DETECTOR_BATCH_SIZE:-16}" \
  --cpu=4 \
  --memory=16Gi \
  --gpu=1 \
  --gpu-type=nvidia-l4 \
  --no-gpu-zonal-redundancy \
  --tasks=1 \
  --parallelism=1 \
  --max-retries=0 \
  --task-timeout=7200s \
  --quiet

printf 'Detector trainer ready: %s -> %s\n' "$DATASET_VERSION" "$MODEL_VERSION"
