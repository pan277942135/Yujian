#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-gemini-api-project-503706}"
REGION="${REGION:-asia-east1}"
BUCKET="${BUCKET:-yujian-model-factory-571785698442}"
JOB_NAME="${TRAINING_JOB_NAME:-yujian-classifier-trainer}"
AR_REPOSITORY="${TRAINING_AR_REPOSITORY:-yujian-training}"
SQL_INSTANCE="${SQL_INSTANCE:-yujian-registry}"
DB_NAME="${DB_NAME:-yujian_registry}"
DB_USER="${DB_USER:-yujian_console}"
RUNTIME_SA="${RUNTIME_SA:-yujian-model-factory@${PROJECT_ID}.iam.gserviceaccount.com}"
DB_SECRET="${DB_SECRET:-yujian-console-db-password}"

log() { printf '\n==> %s\n' "$*"; }

log "Using project ${PROJECT_ID}"
gcloud config set project "$PROJECT_ID" >/dev/null

gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com \
  sqladmin.googleapis.com \
  --project "$PROJECT_ID"

log "Ensuring Artifact Registry repository ${AR_REPOSITORY}"
if ! gcloud artifacts repositories describe "$AR_REPOSITORY" \
  --project "$PROJECT_ID" --location "$REGION" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$AR_REPOSITORY" \
    --project "$PROJECT_ID" \
    --location "$REGION" \
    --repository-format=docker \
    --description="YuJian classifier training images"
fi

GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPOSITORY}/classifier-trainer:${GIT_SHA}"

log "Building trainer image ${IMAGE_URI}"
if gcloud artifacts docker images describe "$IMAGE_URI" --project "$PROJECT_ID" >/dev/null 2>&1; then
  echo "Trainer image already exists; skipping build."
else
  gcloud builds submit . \
    --project "$PROJECT_ID" \
    --config infra/cloudbuild.trainer.yaml \
    --substitutions="_IMAGE_URI=${IMAGE_URI}"
fi

CONNECTION_NAME="$(gcloud sql instances describe "$SQL_INSTANCE" --project "$PROJECT_ID" --format='value(connectionName)')"

log "Deploying Cloud Run Job ${JOB_NAME}"
gcloud run jobs deploy "$JOB_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --image "$IMAGE_URI" \
  --service-account "$RUNTIME_SA" \
  --set-cloudsql-instances "$CONNECTION_NAME" \
  --set-env-vars="GCS_BUCKET=${BUCKET},CLOUD_SQL_CONNECTION_NAME=${CONNECTION_NAME},DB_USER=${DB_USER},DB_NAME=${DB_NAME},APP_GIT_COMMIT=${GIT_SHA}" \
  --set-secrets="DB_PASSWORD=${DB_SECRET}:latest" \
  --cpu=2 \
  --memory=4Gi \
  --tasks=1 \
  --parallelism=1 \
  --max-retries=0 \
  --task-timeout=3600s \
  --quiet

log "Granting Console service identity permission to execute job with per-run overrides"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/run.jobsExecutorWithOverrides" \
  --condition=None >/dev/null

cat <<EOF

============================================================
YuJian classifier training job deployed
============================================================
Job: ${JOB_NAME}
Image: ${IMAGE_URI}
Region: ${REGION}
CPU / Memory: 2 vCPU / 4 GiB
Task timeout: 3600s

日常训练不再需要 Cloud Shell：
  Model Factory Console -> 模型训练 -> 启动训练
============================================================
EOF
