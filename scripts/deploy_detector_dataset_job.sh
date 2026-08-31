#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-gemini-api-project-503706}"
REGION="${REGION:-asia-east1}"
BUCKET="${BUCKET:-yujian-model-factory-571785698442}"
JOB_NAME="${DETECTOR_DATASET_JOB_NAME:-yujian-detector-dataset-builder}"
AR_REPOSITORY="${TRAINING_AR_REPOSITORY:-yujian-training}"
CONSOLE_SERVICE="${CONSOLE_SERVICE:-yujian-model-factory-console}"
DB_NAME="${DB_NAME:-yujian_registry}"
DB_USER="${DB_USER:-yujian_console}"
RUNTIME_SA="${RUNTIME_SA:-yujian-model-factory@${PROJECT_ID}.iam.gserviceaccount.com}"
DB_SECRET="${DB_SECRET:-yujian-console-db-password}"
DATASET_VERSION="${DETECTOR_DATASET_VERSION:-DET_DS_v0.1}"

log() { printf '\n==> %s\n' "$*"; }

gcloud config set project "$PROJECT_ID" >/dev/null
# Required APIs are already enabled. The GitHub deployer intentionally has no
# serviceusage.services.enable and does not need Cloud SQL Admin read access.

if ! gcloud artifacts repositories describe "$AR_REPOSITORY" --project "$PROJECT_ID" --location "$REGION" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$AR_REPOSITORY" --project "$PROJECT_ID" --location "$REGION" --repository-format=docker --description="YuJian training images"
fi

GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPOSITORY}/classifier-trainer:${GIT_SHA}"

log "Building/reusing CPU data-builder image ${IMAGE_URI}"
if ! gcloud artifacts docker images describe "$IMAGE_URI" --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud builds submit . --project "$PROJECT_ID" --config infra/cloudbuild.trainer.yaml --substitutions="_IMAGE_URI=${IMAGE_URI}"
fi

log "Reusing Cloud SQL connection from ${CONSOLE_SERVICE}"
CONNECTION_NAME="$(
  gcloud run services describe "$CONSOLE_SERVICE" --project "$PROJECT_ID" --region "$REGION" --format=json \
  | python -c '
import json,re,sys
x=json.load(sys.stdin)
# First prefer the explicit runtime env var.
for container in (((x.get("spec") or {}).get("template") or {}).get("spec") or {}).get("containers", []):
    for env in container.get("env", []):
        if env.get("name") == "CLOUD_SQL_CONNECTION_NAME" and env.get("value"):
            print(env["value"]); raise SystemExit
# Fall back to the Cloud Run annotation used by Cloud SQL attachments.
ann=(((x.get("spec") or {}).get("template") or {}).get("metadata") or {}).get("annotations", {})
value=ann.get("run.googleapis.com/cloudsql-instances", "")
if value:
    print(value.split(",")[0]); raise SystemExit
raise SystemExit("CLOUD_SQL_CONNECTION_NOT_FOUND")
'
)"
if [[ -z "$CONNECTION_NAME" ]]; then
  echo "Unable to resolve Cloud SQL connection from ${CONSOLE_SERVICE}" >&2
  exit 1
fi

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
