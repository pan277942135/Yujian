#!/usr/bin/env bash
set -euo pipefail

# Run from a clean checkout of the reviewed PR #166 commit in the project
# owner's existing Cloud Shell. This creates/executes a dedicated CPU-only
# Cloud Run Job; it never starts the GPU training job.
PROJECT_ID="${PROJECT_ID:-gemini-api-project-503706}"
BUILDER_REGION="${DETECTOR_DATASET_BUILDER_REGION:-asia-east1}"
ARTIFACT_REGION="${DETECTOR_ARTIFACT_REGION:-${DETECTOR_GPU_REGION:-asia-southeast1}}"
BUCKET="${GCS_BUCKET:-yujian-model-factory-571785698442}"
CONSOLE_SERVICE="${CONSOLE_SERVICE:-yujian-model-factory-console}"
AR_REPOSITORY="${DETECTOR_AR_REPOSITORY:-yujian-detector-training}"
GIT_SHA="$(git rev-parse HEAD)"
SHORT_SHA="${GIT_SHA:0:8}"
PR_REF="${DETECTOR_DATASET_PR_REF:-refs/remotes/origin/pr-166-live}"
JOB_NAME="${DETECTOR_DATASET_BUILDER_JOB_NAME:-yujian-detector-dataset-v02-${SHORT_SHA}}"
BUILD_NONCE="$(date -u +%Y%m%dT%H%M%S)-${RANDOM}"
IMAGE_NAME="accepted-pool-dataset-builder-${GIT_SHA}-${BUILD_NONCE}"
IMAGE_URI="${ARTIFACT_REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPOSITORY}/${IMAGE_NAME}"

if [[ ! -f trainer/build_detector_dataset_v02.py || ! -f trainer/detector_dataset_gate.py ]]; then
  echo "Run this from the reviewed Yujian checkout containing the v0.2 builder." >&2
  exit 2
fi
if [[ ! "${GIT_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "A full Git SHA is required; run from the reviewed PR #166 commit." >&2
  exit 2
fi
command -v gcloud >/dev/null 2>&1 || { echo "gcloud is required in Cloud Shell." >&2; exit 2; }
git show-ref --verify --quiet "${PR_REF}" || {
  echo "Fetch PR #166 to ${PR_REF} before building; refusing to build an unpinned checkout." >&2
  exit 2
}
[[ "$(git rev-parse "${PR_REF}")" == "${GIT_SHA}" ]] || {
  echo "HEAD does not match the fetched PR #166 head ref; refusing to build stale or unrelated code." >&2
  exit 2
}
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || {
  echo "Checkout has uncommitted or untracked changes; refusing to build bytes that do not match APP_GIT_COMMIT." >&2
  exit 2
}

echo "Project=${PROJECT_ID} BuilderRegion=${BUILDER_REGION} ArtifactRegion=${ARTIFACT_REGION} Commit=${GIT_SHA} Job=${JOB_NAME}"
echo "Accepted Pool source=gs://${BUCKET}/datasets/accepted_pool/manifest.csv"
echo "Destination=gs://${BUCKET}/detector-datasets/DET_DS_v0.2/"

# Complete all read-only preflights before building or deploying. A permission
# failure aborts under set -e and is never interpreted as an absent resource.
gcloud artifacts repositories describe "${AR_REPOSITORY}" \
  --project="${PROJECT_ID}" --location="${ARTIFACT_REGION}" >/dev/null
JOB_NAMES="$(gcloud run jobs list \
  --project="${PROJECT_ID}" --region="${BUILDER_REGION}" \
  --format='value(metadata.name)')"
if grep -Fxq "${JOB_NAME}" <<<"${JOB_NAMES}"; then
  echo "Refusing to replace existing builder Job ${JOB_NAME}. Inspect its execution and select a new DETECTOR_DATASET_BUILDER_JOB_NAME if a retry is needed." >&2
  exit 3
fi

SERVICE_JSON="$(gcloud run services describe "${CONSOLE_SERVICE}" \
  --project="${PROJECT_ID}" --region="${BUILDER_REGION}" --format=json)"
SERVICE_CONFIG="$(python3 -c '
import json,sys
service=json.loads(sys.argv[1])
templates=[]
if isinstance(service.get("template"), dict):
    templates.append(service["template"])
if isinstance(service.get("spec"), dict) and isinstance(service["spec"].get("template"), dict):
    templates.append(service["spec"]["template"])
connection=runtime_sa=secret_name=""
values={}
for candidate in templates:
    template=candidate.get("template") or candidate
    spec=template.get("spec") or template
    runtime_sa=str(spec.get("serviceAccountName") or spec.get("serviceAccount") or runtime_sa)
    containers=spec.get("containers") or template.get("containers") or []
    for container in containers:
        for item in container.get("env") or []:
            name=str(item.get("name") or "")
            if name == "DB_PASSWORD":
                ref=((item.get("valueSource") or {}).get("secretKeyRef")
                     or (item.get("valueFrom") or {}).get("secretKeyRef") or {})
                secret_name=str(ref.get("secret") or ref.get("name") or secret_name)
            elif item.get("value") is not None:
                values[name]=str(item["value"])
    connection=values.get("CLOUD_SQL_CONNECTION_NAME", connection)
    annotations=(template.get("metadata") or {}).get("annotations") or {}
    if not annotations:
        annotations=(candidate.get("metadata") or {}).get("annotations") or {}
    instances=annotations.get("run.googleapis.com/cloudsql-instances", "")
    if not connection and instances:
        connection=instances.split(",")[0].strip()
if not all((connection,runtime_sa,secret_name,values.get("DB_USER"),values.get("DB_NAME"))):
    raise SystemExit("Live console lacks a readable Cloud SQL connection, runtime service account, DB secret reference, DB_USER, or DB_NAME")
service_bucket=values.get("GCS_BUCKET", "")
if service_bucket and service_bucket != sys.argv[2]:
    raise SystemExit("Live console GCS_BUCKET differs from the requested dataset bucket")
print("\n".join((connection,runtime_sa,secret_name,values["DB_USER"],values["DB_NAME"])))
' "${SERVICE_JSON}" "${BUCKET}")"
mapfile -t SERVICE_CONFIG <<<"${SERVICE_CONFIG}"
CONNECTION_NAME="${SERVICE_CONFIG[0]:-}"
RUNTIME_SA="${SERVICE_CONFIG[1]:-}"
DB_SECRET="${SERVICE_CONFIG[2]:-}"
DB_USER="${SERVICE_CONFIG[3]:-}"
DB_NAME="${SERVICE_CONFIG[4]:-}"

# These checks verify resource identity, not IAM grants. The dataset Job will
# still fail closed if its service account lacks runtime access to SQL or GCS.
gcloud iam service-accounts describe "${RUNTIME_SA}" --project="${PROJECT_ID}" >/dev/null
gcloud secrets describe "${DB_SECRET}" --project="${PROJECT_ID}" >/dev/null
gcloud storage buckets describe "gs://${BUCKET}" --project="${PROJECT_ID}" >/dev/null
echo "Live console configuration resolved; secret contents were not read."

echo "Building CPU-only reviewed v0.2 dataset builder image ${IMAGE_URI}"
gcloud builds submit . \
  --project="${PROJECT_ID}" \
  --config=infra/cloudbuild.trainer.yaml \
  --substitutions="_IMAGE_URI=${IMAGE_URI}"

IMAGE_DIGEST="$(gcloud artifacts docker images describe "${IMAGE_URI}" \
  --project="${PROJECT_ID}" --format='value(image_summary.digest)')"
if [[ ! "${IMAGE_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "Unable to resolve immutable builder image digest; refusing Job deployment." >&2
  exit 4
fi
PINNED_IMAGE_URI="${ARTIFACT_REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPOSITORY}/${IMAGE_NAME}@${IMAGE_DIGEST}"

echo "Deploying dedicated CPU-only dataset builder Job ${JOB_NAME}"
gcloud run jobs deploy "${JOB_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${BUILDER_REGION}" \
  --image="${PINNED_IMAGE_URI}" \
  --command=python \
  --args="-m,trainer.build_detector_dataset_v02,--publish-gcs,--bucket,${BUCKET}" \
  --service-account="${RUNTIME_SA}" \
  --set-cloudsql-instances="${CONNECTION_NAME}" \
  --set-env-vars="GCS_BUCKET=${BUCKET},DETECTOR_DATASET_VERSION=DET_DS_v0.2,CLOUD_SQL_CONNECTION_NAME=${CONNECTION_NAME},DB_USER=${DB_USER},DB_NAME=${DB_NAME},APP_GIT_COMMIT=${GIT_SHA}" \
  --set-secrets="DB_PASSWORD=${DB_SECRET}:latest" \
  --cpu=4 \
  --memory=16Gi \
  --tasks=1 \
  --parallelism=1 \
  --max-retries=0 \
  --task-timeout=86400s \
  --quiet

echo "Executing dataset build only (no GPU training)"
gcloud run jobs execute "${JOB_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${BUILDER_REGION}" \
  --wait

echo "Builder execution complete. Verify the job logs, then run:"
echo "gcloud storage ls gs://${BUCKET}/detector-datasets/DET_DS_v0.2/"
echo "gcloud storage cat gs://${BUCKET}/detector-datasets/DET_DS_v0.2/bootstrap_report.json"
