#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-gemini-api-project-503706}"
REGION="${REGION:-asia-east1}"
BUCKET="${BUCKET:-yujian-model-factory-571785698442}"
SERVICE="${SERVICE:-yujian-model-factory-console}"
TRAINING_JOB_NAME="${TRAINING_JOB_NAME:-yujian-classifier-trainer}"
DEPLOY_TRAINER="${DEPLOY_TRAINER:-1}"
SQL_INSTANCE="${SQL_INSTANCE:-yujian-registry}"
DB_NAME="${DB_NAME:-yujian_registry}"
DB_USER="${DB_USER:-yujian_console}"
RUNTIME_SA="${RUNTIME_SA:-yujian-model-factory@${PROJECT_ID}.iam.gserviceaccount.com}"
DB_SECRET="${DB_SECRET:-yujian-console-db-password}"
CONSOLE_SECRET="${CONSOLE_SECRET:-yujian-console-access-key}"

log() { printf '\n==> %s\n' "$*"; }
exists_secret() { gcloud secrets describe "$1" --project "$PROJECT_ID" >/dev/null 2>&1; }

log "Using project ${PROJECT_ID}"
gcloud config set project "$PROJECT_ID" >/dev/null

log "Enabling required APIs"
gcloud services enable \
  run.googleapis.com \
  sqladmin.googleapis.com \
  secretmanager.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com \
  iam.googleapis.com \
  vision.googleapis.com \
  --project "$PROJECT_ID"

log "Ensuring runtime service account"
if ! gcloud iam service-accounts describe "$RUNTIME_SA" --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud iam service-accounts create yujian-model-factory \
    --display-name="YuJian Model Factory" \
    --project "$PROJECT_ID"
fi

log "Ensuring Cloud SQL PostgreSQL instance"
if ! gcloud sql instances describe "$SQL_INSTANCE" --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud sql instances create "$SQL_INSTANCE" \
    --project "$PROJECT_ID" \
    --database-version=POSTGRES_15 \
    --tier=db-f1-micro \
    --region="$REGION" \
    --storage-type=SSD \
    --storage-size=10 \
    --storage-auto-increase \
    --availability-type=ZONAL
fi

log "Ensuring database ${DB_NAME}"
if ! gcloud sql databases describe "$DB_NAME" --instance "$SQL_INSTANCE" --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud sql databases create "$DB_NAME" --instance "$SQL_INSTANCE" --project "$PROJECT_ID"
fi

log "Ensuring database password secret"
if exists_secret "$DB_SECRET"; then
  DB_PASSWORD="$(gcloud secrets versions access latest --secret "$DB_SECRET" --project "$PROJECT_ID")"
else
  DB_PASSWORD="$(openssl rand -base64 36 | tr -d '\n')"
  printf '%s' "$DB_PASSWORD" | gcloud secrets create "$DB_SECRET" \
    --project "$PROJECT_ID" \
    --replication-policy=automatic \
    --data-file=-
fi

log "Ensuring PostgreSQL application user"
if gcloud sql users list --instance "$SQL_INSTANCE" --project "$PROJECT_ID" --format='value(name)' | grep -Fxq "$DB_USER"; then
  gcloud sql users set-password "$DB_USER" \
    --instance "$SQL_INSTANCE" \
    --password "$DB_PASSWORD" \
    --project "$PROJECT_ID"
else
  gcloud sql users create "$DB_USER" \
    --instance "$SQL_INSTANCE" \
    --password "$DB_PASSWORD" \
    --project "$PROJECT_ID"
fi
unset DB_PASSWORD

log "Ensuring Console access-key secret"
if ! exists_secret "$CONSOLE_SECRET"; then
  CONSOLE_ACCESS_KEY="$(openssl rand -hex 18)"
  printf '%s' "$CONSOLE_ACCESS_KEY" | gcloud secrets create "$CONSOLE_SECRET" \
    --project "$PROJECT_ID" \
    --replication-policy=automatic \
    --data-file=-
  unset CONSOLE_ACCESS_KEY
fi

log "Granting runtime IAM"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/cloudsql.client" \
  --condition=None >/dev/null

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/serviceusage.serviceUsageConsumer" \
  --condition=None >/dev/null

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/run.jobsExecutorWithOverrides" \
  --condition=None >/dev/null

gcloud secrets add-iam-policy-binding "$DB_SECRET" \
  --project "$PROJECT_ID" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/secretmanager.secretAccessor" >/dev/null

gcloud secrets add-iam-policy-binding "$CONSOLE_SECRET" \
  --project "$PROJECT_ID" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/secretmanager.secretAccessor" >/dev/null

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/storage.objectAdmin" >/dev/null

CONNECTION_NAME="$(gcloud sql instances describe "$SQL_INSTANCE" --project "$PROJECT_ID" --format='value(connectionName)')"
GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

log "Deploying ${SERVICE} to Cloud Run"
gcloud run deploy "$SERVICE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --source . \
  --service-account "$RUNTIME_SA" \
  --add-cloudsql-instances "$CONNECTION_NAME" \
  --set-env-vars="GCS_BUCKET=${BUCKET},CLOUD_SQL_CONNECTION_NAME=${CONNECTION_NAME},DB_USER=${DB_USER},DB_NAME=${DB_NAME},APP_GIT_COMMIT=${GIT_SHA},CONSOLE_COOKIE_SECURE=1,GCP_PROJECT_ID=${PROJECT_ID},GCP_REGION=${REGION},TRAINING_JOB_NAME=${TRAINING_JOB_NAME}" \
  --set-secrets="DB_PASSWORD=${DB_SECRET}:latest,CONSOLE_ACCESS_KEY=${CONSOLE_SECRET}:latest" \
  --cpu=1 \
  --memory=1Gi \
  --concurrency=20 \
  --min-instances=0 \
  --max-instances=2 \
  --timeout=900 \
  --allow-unauthenticated \
  --quiet

SERVICE_URL="$(gcloud run services describe "$SERVICE" --project "$PROJECT_ID" --region "$REGION" --format='value(status.url)')"

log "Health check"
curl -fsS "${SERVICE_URL}/health"
printf '\n'

if [[ "$DEPLOY_TRAINER" == "1" ]]; then
  log "Deploying classifier training worker"
  PROJECT_ID="$PROJECT_ID" \
  REGION="$REGION" \
  BUCKET="$BUCKET" \
  TRAINING_JOB_NAME="$TRAINING_JOB_NAME" \
  SQL_INSTANCE="$SQL_INSTANCE" \
  DB_NAME="$DB_NAME" \
  DB_USER="$DB_USER" \
  RUNTIME_SA="$RUNTIME_SA" \
  DB_SECRET="$DB_SECRET" \
  bash scripts/deploy_training_gcp.sh
else
  log "Skipping classifier training worker because DEPLOY_TRAINER=${DEPLOY_TRAINER}"
fi

cat <<EOF

============================================================
YuJian Model Factory Console deployed
============================================================
URL: ${SERVICE_URL}
Training Job: ${TRAINING_JOB_NAME}

Retrieve the Console access key when needed:
  gcloud secrets versions access latest \\
    --secret=${CONSOLE_SECRET} \\
    --project=${PROJECT_ID}

日常操作：总览 -> 数据批次 -> 鱼体检测 -> 人工审核 -> 鱼种管理 -> 用户反馈 -> 数据集 -> 模型训练

Cloud Shell is no longer required for normal daily data / training operations.
============================================================
EOF
