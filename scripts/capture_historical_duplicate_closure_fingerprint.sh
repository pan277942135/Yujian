#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-gemini-api-project-503706}"
REGION="${REGION:-asia-east1}"
SERVICE="${SERVICE:-yujian-model-factory-console}"
JOB_NAME="${JOB_NAME:-yujian-historical-duplicate-closure-fingerprint-v1}"
EXPECTED_FENCE="${EXPECTED_FENCE:?EXPECTED_FENCE is required}"
OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH is required}"
GCS_OUTPUT_URI="${GCS_OUTPUT_URI:?GCS_OUTPUT_URI is required}"
APP_GIT_COMMIT="${APP_GIT_COMMIT:-${GITHUB_SHA:-$(git rev-parse HEAD)}}"

SERVICE_JSON="$(mktemp)"
trap 'rm -f "$SERVICE_JSON"; gcloud run jobs delete "$JOB_NAME" --project "$PROJECT_ID" --region "$REGION" --quiet >/dev/null 2>&1 || true' EXIT

gcloud run services describe "$SERVICE" \
  --project "$PROJECT_ID" --region "$REGION" --format=json > "$SERVICE_JSON"

IFS=$'\t' read -r IMAGE RUNTIME_SA BUCKET CONNECTION DB_USER DB_NAME DB_SECRET < <(
  python3 - "$SERVICE_JSON" <<'PY'
import json
import sys

payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
template = (payload.get("spec") or {}).get("template") or {}
spec = template.get("spec") or {}
containers = spec.get("containers") or []
if not containers:
    raise SystemExit("live Console service has no container")
container = containers[0]
env = {str(item.get("name")): item for item in container.get("env") or []}

def value(name):
    return str((env.get(name) or {}).get("value") or "")

secret_item = env.get("DB_PASSWORD") or {}
secret_ref = (secret_item.get("valueSource") or {}).get("secretKeyRef") or {}
secret_ref = secret_ref or (secret_item.get("valueFrom") or {}).get("secretKeyRef") or {}
db_secret = str(secret_ref.get("secret") or secret_ref.get("name") or "")
annotations = (template.get("metadata") or {}).get("annotations") or {}
connection = value("CLOUD_SQL_CONNECTION_NAME") or str(
    annotations.get("run.googleapis.com/cloudsql-instances") or ""
).split(",")[0]
values = [
    str(container.get("image") or ""),
    str(spec.get("serviceAccountName") or ""),
    value("GCS_BUCKET"),
    connection,
    value("DB_USER"),
    value("DB_NAME"),
    db_secret,
]
if not all(values):
    raise SystemExit(f"incomplete live Console configuration: {values}")
print("\t".join(values))
PY
)

gcloud run jobs deploy "$JOB_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --image "$IMAGE" \
  --command python \
  --args=scripts/historical_duplicate_closure_fingerprint.py,--output,/tmp/historical-duplicate-closure-fingerprint.json \
  --service-account "$RUNTIME_SA" \
  --set-cloudsql-instances "$CONNECTION" \
  --set-env-vars="GCS_BUCKET=${BUCKET},CLOUD_SQL_CONNECTION_NAME=${CONNECTION},DB_USER=${DB_USER},DB_NAME=${DB_NAME},APP_GIT_COMMIT=${APP_GIT_COMMIT},HISTORICAL_DUPLICATE_CLOSURE_WRITE_FENCE=${EXPECTED_FENCE},CLOSURE_FINGERPRINT_GCS_URI=${GCS_OUTPUT_URI}" \
  --set-secrets="DB_PASSWORD=${DB_SECRET}:latest" \
  --cpu=1 --memory=1Gi --tasks=1 --parallelism=1 --max-retries=0 --task-timeout=600s --quiet

gcloud run jobs execute "$JOB_NAME" --project "$PROJECT_ID" --region "$REGION" --wait --quiet
mkdir -p "$(dirname "$OUTPUT_PATH")"
gcloud storage cp "$GCS_OUTPUT_URI" "$OUTPUT_PATH"

python3 - "$OUTPUT_PATH" "$EXPECTED_FENCE" <<'PY'
import json
import sys

payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
expected = sys.argv[2].strip().lower() in {"1", "true", "yes", "on"}
assert payload.get("write_fence_active") is expected, payload
print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
PY
