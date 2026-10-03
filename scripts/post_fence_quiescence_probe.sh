#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-gemini-api-project-503706}"
REGION="${REGION:-asia-east1}"
SERVICE="${SERVICE:-yujian-model-factory-console}"
JOB_NAME="${JOB_NAME:-yujian-historical-duplicate-post-fence-v2}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
GCS_OUTPUT_ROOT="${GCS_OUTPUT_ROOT:?GCS_OUTPUT_ROOT is required}"
EXPECTED_APP_GIT_COMMITS="${EXPECTED_APP_GIT_COMMITS:?EXPECTED_APP_GIT_COMMITS is required}"

mkdir -p "$OUTPUT_DIR"
SERVICE_JSON="$(mktemp)"
trap 'rm -f "$SERVICE_JSON"; gcloud run jobs delete "$JOB_NAME" --project "$PROJECT_ID" --region "$REGION" --quiet >/dev/null 2>&1 || true' EXIT

gcloud run services describe "$SERVICE" --project "$PROJECT_ID" --region "$REGION" --format=json > "$SERVICE_JSON"

IFS=$'\t' read -r SERVICE_URL LATEST_REVISION IMAGE RUNTIME_SA BUCKET CONNECTION DB_USER DB_NAME DB_SECRET FENCE APP_COMMIT < <(
  python3 - "$SERVICE_JSON" <<'PY'
import json
import sys

payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
status = payload.get("status") or {}
template = (payload.get("spec") or {}).get("template") or {}
spec = template.get("spec") or {}
containers = spec.get("containers") or []
if not containers:
    raise SystemExit("live Console service has no container")
container = containers[0]
env = {str(item.get("name")): item for item in container.get("env") or []}
traffic = status.get("traffic") or []
latest = str(status.get("latestReadyRevisionName") or "")
latest_percent = sum(int(item.get("percent") or 0) for item in traffic if item.get("revisionName") == latest)
all_percent = sum(int(item.get("percent") or 0) for item in traffic)
if not latest or latest_percent != 100 or all_percent != 100:
    raise SystemExit(f"fenced revision is not serving 100%: latest={latest!r} latest_percent={latest_percent} all_percent={all_percent}")

def value(name):
    return str((env.get(name) or {}).get("value") or "")

fence = value("HISTORICAL_DUPLICATE_CLOSURE_WRITE_FENCE")
if fence.strip().lower() != "true":
    raise SystemExit(f"HISTORICAL_DUPLICATE_CLOSURE_WRITE_FENCE is not true: {fence!r}")
app_commit = value("APP_GIT_COMMIT")
if not app_commit:
    raise SystemExit("live Console service has no APP_GIT_COMMIT")

secret_item = env.get("DB_PASSWORD") or {}
secret_ref = (secret_item.get("valueSource") or {}).get("secretKeyRef") or {}
secret_ref = secret_ref or (secret_item.get("valueFrom") or {}).get("secretKeyRef") or {}
db_secret = str(secret_ref.get("secret") or secret_ref.get("name") or "")
annotations = (template.get("metadata") or {}).get("annotations") or {}
connection = value("CLOUD_SQL_CONNECTION_NAME") or str(annotations.get("run.googleapis.com/cloudsql-instances") or "").split(",")[0]
values = [
    str(status.get("url") or ""), latest, str(container.get("image") or ""),
    str(spec.get("serviceAccountName") or ""), value("GCS_BUCKET"), connection,
    value("DB_USER"), value("DB_NAME"), db_secret, fence, app_commit,
]
if not all(values):
    raise SystemExit(f"incomplete live Console configuration: {values}")
print("\t".join(values))
PY
)

if [[ ",${EXPECTED_APP_GIT_COMMITS}," != *",${APP_COMMIT},"* ]]; then
  echo "deployed fenced APP_GIT_COMMIT is not an expected implementation: ${APP_COMMIT}" >&2
  exit 1
fi

gcloud run jobs deploy "$JOB_NAME" \
  --project "$PROJECT_ID" --region "$REGION" --image "$IMAGE" \
  --command python \
  --args=scripts/historical_duplicate_closure_fingerprint.py,--output,/tmp/historical-duplicate-closure-fingerprint.json \
  --service-account "$RUNTIME_SA" --set-cloudsql-instances "$CONNECTION" \
  --set-env-vars="GCS_BUCKET=${BUCKET},CLOUD_SQL_CONNECTION_NAME=${CONNECTION},DB_USER=${DB_USER},DB_NAME=${DB_NAME},APP_GIT_COMMIT=${APP_COMMIT},HISTORICAL_DUPLICATE_CLOSURE_WRITE_FENCE=true,CLOSURE_FINGERPRINT_GCS_URI=${GCS_OUTPUT_ROOT}/initial.json" \
  --set-secrets="DB_PASSWORD=${DB_SECRET}:latest" \
  --cpu=1 --memory=1Gi --tasks=1 --parallelism=1 --max-retries=0 --task-timeout=600s --quiet

capture() {
  local label="$1"
  local object_uri="${GCS_OUTPUT_ROOT}/${label}.json"
  gcloud run jobs execute "$JOB_NAME" --project "$PROJECT_ID" --region "$REGION" \
    --update-env-vars="CLOSURE_FINGERPRINT_GCS_URI=${object_uri}" --wait --quiet
  gcloud storage cp "$object_uri" "${OUTPUT_DIR}/${label}.json"
}

capture t0
sleep 60
capture t60
sleep 60
capture t120

python3 - "$OUTPUT_DIR" <<'PY'
import datetime as dt
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
payloads = {name: json.loads((root / f"{name}.json").read_text(encoding="utf-8")) for name in ("t0", "t60", "t120")}
keys = ("image_asset_count", "latest_updated_at", "truth_fingerprint")
baseline = tuple(payloads["t0"].get(key) for key in keys)
errors = []
t0_captured = dt.datetime.fromisoformat(str(payloads["t0"]["captured_at"]).replace("Z", "+00:00"))
for label, payload in payloads.items():
    current = tuple(payload.get(key) for key in keys)
    if current != baseline:
        errors.append({"sample": label, "expected": baseline, "actual": current})
    latest = payload.get("latest_updated_at")
    if latest and dt.datetime.fromisoformat(str(latest).replace("Z", "+00:00")) > t0_captured:
        errors.append({"sample": label, "latest_updated_at": latest, "t0_captured_at": payloads["t0"].get("captured_at")})
if errors:
    print("POST_FENCE_QUIESCENCE=FAIL")
    print(json.dumps({"errors": errors, "samples": payloads}, ensure_ascii=False, sort_keys=True))
    raise SystemExit(1)
print("POST_FENCE_QUIESCENCE=PASS")
print(json.dumps({"samples": payloads}, ensure_ascii=False, sort_keys=True))
PY

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  {
    echo "service_url=${SERVICE_URL}"
    echo "fenced_revision=${LATEST_REVISION}"
    echo "app_git_commit=${APP_COMMIT}"
    echo "write_fence_active=true"
  } >> "$GITHUB_OUTPUT"
fi

