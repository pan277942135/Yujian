#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-gemini-api-project-503706}"
REGION="${REGION:-asia-east1}"
BUCKET="${BUCKET:-yujian-model-factory-571785698442}"
JOB_NAME="${DETECTOR_DATASET_JOB_NAME:-yujian-detector-dataset-builder}"
AR_REPOSITORY="${TRAINING_AR_REPOSITORY:-yujian-training}"
SQL_INSTANCE="${SQL_INSTANCE:-yujian-registry}"
DB_NAME="${DB_NAME:-yujian_registry}"
DB_USER="${DB_USER:-yujian_console}"
RUNTIME_SA="${RUNTIME_SA:-yujian-model-factory@${PROJECT_ID}.iam.gserviceaccount.com}"
DB_SECRET="${DB_SECRET:-yujian-console-db-password}"
DATASET_VERSION="${DETECTOR_DATASET_VERSION:-DET_DS_v0.1}"

log() { printf '\n==> %s\n' "$*"; }

gcloud config set project "$PROJECT_ID" >/dev/null
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com storage.googleapis.com sqladmin.googleapis.com --project "$PROJECT_ID"

if ! gcloud artifacts repositories describe "$AR_REPOSITORY" --project "$PROJECT_ID" --location "$REGION" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$AR_REPOSITORY" --project "$PROJECT_ID" --location "$REGION" --repository-format=docker --description="YuJian training images"
fi

GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPOSITORY}/classifier-trainer:${GIT_SHA}"

log "Building/reusing CPU data-builder image ${IMAGE_URI}"
if ! gcloud artifacts docker images describe "$IMAGE_URI" --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud builds submit . --project "$PROJECT_ID" --config infra/cloudbuild.trainer.yaml --substitutions="_IMAGE_URI=${IMAGE_URI}"
fi

CONNECTION_NAME="$(gcloud sql instances describe "$SQL_INSTANCE" --project "$PROJECT_ID" --format='value(connectionName)')"

log "Deploying detector dataset builder ${JOB_NAME}"
gcloud run jobs deploy "$JOB_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --image "$IMAGE_URI" \
  --command python \
  --args=-m,trainer.build_detector_dataset \
  --service-account "$RUNTIME_SA" \
  --set-cloudsql-instances "$CONNECTION_NAME" \
  --set-env-vars="GCS_BUCKET=${BUCKET},DETECTOR_DATASET_VERSION=${DATASET_VERSION},CLOUD_SQL_CONNECTION_NAME=${CONNECTION_NAME},DB_USER=${DB_USER},DB_NAME=${DB_NAME},APP_GIT_COMMIT=${GIT_SHA}" \
  --set-secrets="DB_PASSWORD=${DB_SECRET}:latest" \
  --cpu=2 \
  --memory=4Gi \
  --tasks=1 \
  --parallelism=1 \
  --max-retries=0 \
  --task-timeout=3600s \
  --quiet

printf 'Detector dataset builder ready: %s (%s)\n' "$JOB_NAME" "$DATASET_VERSION"
