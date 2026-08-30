#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-gemini-api-project-503706}"
PROJECT_NUMBER="${PROJECT_NUMBER:-571785698442}"
REGION="${REGION:-asia-east1}"
SERVICE="${SERVICE:-yujian-model-factory-console}"
BUILD_SA="${BUILD_SA:-${PROJECT_NUMBER}-compute@developer.gserviceaccount.com}"
BUILD_SA_RESOURCE="${BUILD_SA_RESOURCE:-projects/${PROJECT_ID}/serviceAccounts/${BUILD_SA}}"
DEPLOY_SA="${DEPLOY_SERVICE_ACCOUNT:-yujian-github-deployer@${PROJECT_ID}.iam.gserviceaccount.com}"
GIT_SHA="${GIT_SHA:-${GITHUB_SHA:-$(git rev-parse HEAD 2>/dev/null || true)}}"
HEALTH_ATTEMPTS="${HEALTH_ATTEMPTS:-12}"
HEALTH_DELAY_SECONDS="${HEALTH_DELAY_SECONDS:-5}"

log() { printf '\n==> %s\n' "$*"; }

command -v gcloud >/dev/null 2>&1 || { echo "gcloud is required" >&2; exit 2; }
command -v curl >/dev/null 2>&1 || { echo "curl is required" >&2; exit 2; }

if [[ ! "$GIT_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "GIT_SHA must be a full 40-character Git commit SHA; got: ${GIT_SHA:-<empty>}" >&2
  exit 2
fi

log "Authenticated principal"
gcloud auth list --filter=status:ACTIVE --format='value(account)'
printf 'Cloud Run build service account: %s\n' "$BUILD_SA_RESOURCE"

# Preserve an existing ingest key. UAT bootstraps one random key only when the
# service has never had one; later deploys never rotate it. The value is masked
# immediately and is never committed or printed.
PREVIOUS_SERVICE_JSON="$(gcloud run services describe "$SERVICE" \
  --project "$PROJECT_ID" --region "$REGION" --format=json 2>/dev/null || true)"
FEEDBACK_ENV_PRESENT=0
FEEDBACK_INGEST_KEY=""
if [[ -n "$PREVIOUS_SERVICE_JSON" ]]; then
  FEEDBACK_ENV_PRESENT="$(printf '%s' "$PREVIOUS_SERVICE_JSON" | python -c '
import json,sys
d=json.load(sys.stdin)
containers=((d.get("spec") or {}).get("template") or {}).get("spec",{}).get("containers") or []
env=(containers[0].get("env") if containers else []) or []
print(1 if any(x.get("name")=="FEEDBACK_INGEST_KEY" for x in env) else 0)
')"
  FEEDBACK_INGEST_KEY="$(printf '%s' "$PREVIOUS_SERVICE_JSON" | python -c '
import json,sys
d=json.load(sys.stdin)
containers=((d.get("spec") or {}).get("template") or {}).get("spec",{}).get("containers") or []
env=(containers[0].get("env") if containers else []) or []
print(next((x.get("value","") for x in env if x.get("name")=="FEEDBACK_INGEST_KEY"), ""))
')"
fi

DEPLOY_ENV_VARS="APP_GIT_COMMIT=${GIT_SHA}"
if [[ "$FEEDBACK_ENV_PRESENT" == "0" ]]; then
  FEEDBACK_INGEST_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
  DEPLOY_ENV_VARS="${DEPLOY_ENV_VARS},FEEDBACK_INGEST_KEY=${FEEDBACK_INGEST_KEY}"
  printf 'Bootstrapping UAT feedback ingest key: yes\n'
else
  printf 'Bootstrapping UAT feedback ingest key: no (preserving existing configuration)\n'
fi
if [[ -n "$FEEDBACK_INGEST_KEY" && -n "${GITHUB_ACTIONS:-}" ]]; then
  printf '::add-mask::%s\n' "$FEEDBACK_INGEST_KEY"
fi

log "Runtime-only deploy ${SERVICE} @ ${GIT_SHA}"
# Do not mutate IAM, Cloud SQL, trainer resources, or existing secret bindings.
# Source deploy updates provenance and bootstraps the UAT feedback key only when absent.
set +e
gcloud run deploy "$SERVICE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --source . \
  --build-service-account "$BUILD_SA_RESOURCE" \
  --update-env-vars="$DEPLOY_ENV_VARS" \
  --quiet
DEPLOY_RC=$?
set -e

if [[ "$DEPLOY_RC" -ne 0 ]]; then
  cat >&2 <<EOF

Cloud Run source deploy failed.
If the error mentions iam.serviceAccounts.actAs or missing build-service-account
permissions, run this ONE-TIME bootstrap from a project Owner / IAM Admin identity:

  PROJECT_ID=${PROJECT_ID} PROJECT_NUMBER=${PROJECT_NUMBER} \\
    BUILD_SA=${BUILD_SA} bash scripts/bootstrap_github_wif.sh

Equivalent minimum IAM bindings are:

  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \\
    --member="serviceAccount:${BUILD_SA}" \\
    --role="roles/run.builder" --condition=None

  gcloud iam service-accounts add-iam-policy-binding "${BUILD_SA}" \\
    --project "${PROJECT_ID}" \\
    --member="serviceAccount:${DEPLOY_SA}" \\
    --role="roles/iam.serviceAccountUser"

EOF
  exit "$DEPLOY_RC"
fi

SERVICE_JSON="$(gcloud run services describe "$SERVICE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --format=json)"

SERVICE_URL="$(printf '%s' "$SERVICE_JSON" | python -c 'import json,sys; print(json.load(sys.stdin)["status"]["url"])')"
REVISION="$(printf '%s' "$SERVICE_JSON" | python -c 'import json,sys; print(json.load(sys.stdin)["status"]["latestReadyRevisionName"])')"
DEPLOYED_SHA="$(printf '%s' "$SERVICE_JSON" | python -c '
import json,sys
d=json.load(sys.stdin)
containers=((d.get("spec") or {}).get("template") or {}).get("spec",{}).get("containers") or []
env=(containers[0].get("env") if containers else []) or []
print(next((x.get("value","") for x in env if x.get("name")=="APP_GIT_COMMIT"), ""))
')"

if [[ "$DEPLOYED_SHA" != "$GIT_SHA" ]]; then
  echo "Cloud Run service template SHA mismatch: expected ${GIT_SHA}, got ${DEPLOYED_SHA:-<empty>}" >&2
  exit 1
fi

log "Online basic health smoke"
BASIC_HEALTH="$(curl --retry 5 --retry-all-errors --retry-delay 2 --connect-timeout 10 --max-time 30 -fsS "${SERVICE_URL}/health")"
BASIC_STATUS="$(printf '%s' "$BASIC_HEALTH" | python -c 'import json,sys; print(json.load(sys.stdin).get("status", ""))')"
if [[ "$BASIC_STATUS" != "ok" ]]; then
  echo "Basic /health smoke failed: ${BASIC_HEALTH}" >&2
  exit 1
fi

log "Online deployment provenance verification"
DEPLOY_HEALTH=""
for attempt in $(seq 1 "$HEALTH_ATTEMPTS"); do
  if DEPLOY_HEALTH="$(curl --connect-timeout 10 --max-time 30 -fsS "${SERVICE_URL}/health/deploy" 2>/dev/null)"; then
    HEALTH_STATUS="$(printf '%s' "$DEPLOY_HEALTH" | python -c 'import json,sys; print(json.load(sys.stdin).get("status", ""))' 2>/dev/null || true)"
    HEALTH_SHA="$(printf '%s' "$DEPLOY_HEALTH" | python -c 'import json,sys; print(json.load(sys.stdin).get("git_commit", ""))' 2>/dev/null || true)"
    HEALTH_REVISION="$(printf '%s' "$DEPLOY_HEALTH" | python -c 'import json,sys; print(json.load(sys.stdin).get("revision", ""))' 2>/dev/null || true)"
    if [[ "$HEALTH_STATUS" == "ok" && "$HEALTH_SHA" == "$GIT_SHA" && "$HEALTH_REVISION" == "$REVISION" ]]; then
      break
    fi
  fi
  DEPLOY_HEALTH=""
  if [[ "$attempt" -lt "$HEALTH_ATTEMPTS" ]]; then
    sleep "$HEALTH_DELAY_SECONDS"
  fi
done

if [[ -z "$DEPLOY_HEALTH" ]]; then
  echo "Online provenance smoke failed: /health/deploy did not report expected SHA/revision" >&2
  echo "Expected SHA: ${GIT_SHA}" >&2
  echo "Expected revision: ${REVISION}" >&2
  exit 1
fi

FEEDBACK_READY="$(printf '%s' "$DEPLOY_HEALTH" | python -c 'import json,sys; print("true" if json.load(sys.stdin).get("feedback_ingest_key_configured") else "false")')"
if [[ "$FEEDBACK_READY" != "true" ]]; then
  echo "Feedback ingest readiness failed: FEEDBACK_INGEST_KEY is not configured" >&2
  exit 1
fi

FEEDBACK_SMOKE='{"status":"skipped","reason":"key value managed by external secret binding"}'
if [[ -n "$FEEDBACK_INGEST_KEY" ]]; then
  log "Online feedback multipart + GCS + DB smoke"
  SMOKE_IMAGE="$(mktemp --suffix=.png)"
  python - "$SMOKE_IMAGE" <<'PY'
import binascii
import struct
import sys
import zlib

path = sys.argv[1]
def chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)
png = b"\x89PNG\r\n\x1a\n"
png += chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
png += chunk(b"IDAT", zlib.compress(b"\x00\x2e\x8b\x83\xff"))
png += chunk(b"IEND", b"")
open(path, "wb").write(png)
PY
  SMOKE_EVENT_ID="uat_${GIT_SHA:0:12}_${REVISION//[^A-Za-z0-9_.-]/_}"
  FEEDBACK_SMOKE="$(curl --retry 2 --retry-all-errors --retry-delay 2 --connect-timeout 10 --max-time 45 -fsS \
    -H "X-YuJian-Ingest-Key: ${FEEDBACK_INGEST_KEY}" \
    -F "source_event_id=${SMOKE_EVENT_ID}" \
    -F "feedback_type=confirmed" \
    -F "source=uat_deploy_smoke" \
    -F "model_version=deploy-smoke" \
    -F "predicted_species=草鱼" \
    -F "confidence=0.99" \
    -F "smoke=true" \
    -F "file=@${SMOKE_IMAGE};type=image/png" \
    "${SERVICE_URL}/api/feedback/ingest")"
  rm -f "$SMOKE_IMAGE"
  printf '%s' "$FEEDBACK_SMOKE" | python -c '
import json,sys
d=json.load(sys.stdin)
assert d.get("status")=="ok", d
assert d.get("smoke") is True, d
assert d.get("gcs_write_delete") is True, d
assert d.get("db_reachable") is True, d
'
fi

printf 'SERVICE_URL=%s\n' "$SERVICE_URL"
printf 'REVISION=%s\n' "$REVISION"
printf 'APP_GIT_COMMIT=%s\n' "$GIT_SHA"
printf 'BUILD_SA=%s\n' "$BUILD_SA"
printf 'BUILD_SA_RESOURCE=%s\n' "$BUILD_SA_RESOURCE"
printf 'HEALTH=%s\n' "$BASIC_HEALTH"
printf 'DEPLOY_HEALTH=%s\n' "$DEPLOY_HEALTH"
printf 'FEEDBACK_SMOKE=%s\n' "$FEEDBACK_SMOKE"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  {
    echo "service_url=${SERVICE_URL}"
    echo "revision=${REVISION}"
    echo "git_commit=${GIT_SHA}"
    echo "build_service_account=${BUILD_SA}"
    echo "feedback_ingest_ready=${FEEDBACK_READY}"
  } >> "$GITHUB_OUTPUT"
fi
