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

log "Runtime-only deploy ${SERVICE} @ ${GIT_SHA}"
# Deliberately do not create infrastructure, rotate secrets, mutate IAM, deploy the
# trainer, or replace the service's existing Cloud SQL / secret configuration here.
# Unspecified Cloud Run settings are preserved; only a new source revision and the
# provenance variable are updated. Pin the build identity so Cloud Run cannot silently
# switch to a different project-default service account.
set +e
gcloud run deploy "$SERVICE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --source . \
  --build-service-account "$BUILD_SA_RESOURCE" \
  --update-env-vars="APP_GIT_COMMIT=${GIT_SHA}" \
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

printf 'SERVICE_URL=%s\n' "$SERVICE_URL"
printf 'REVISION=%s\n' "$REVISION"
printf 'APP_GIT_COMMIT=%s\n' "$GIT_SHA"
printf 'BUILD_SA=%s\n' "$BUILD_SA"
printf 'BUILD_SA_RESOURCE=%s\n' "$BUILD_SA_RESOURCE"
printf 'HEALTH=%s\n' "$BASIC_HEALTH"
printf 'DEPLOY_HEALTH=%s\n' "$DEPLOY_HEALTH"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  {
    echo "service_url=${SERVICE_URL}"
    echo "revision=${REVISION}"
    echo "git_commit=${GIT_SHA}"
    echo "build_service_account=${BUILD_SA}"
  } >> "$GITHUB_OUTPUT"
fi
