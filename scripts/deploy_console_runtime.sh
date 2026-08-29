#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-gemini-api-project-503706}"
REGION="${REGION:-asia-east1}"
SERVICE="${SERVICE:-yujian-model-factory-console}"
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

log "Runtime-only deploy ${SERVICE} @ ${GIT_SHA}"
# Deliberately do not create infrastructure, rotate secrets, mutate IAM, deploy the
# trainer, or replace the service's existing Cloud SQL / secret configuration here.
# Unspecified Cloud Run settings are preserved; only a new source revision and the
# provenance variable are updated.
gcloud run deploy "$SERVICE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --source . \
  --update-env-vars="APP_GIT_COMMIT=${GIT_SHA}" \
  --quiet

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

log "Online smoke and SHA verification"
HEALTH_JSON=""
for attempt in $(seq 1 "$HEALTH_ATTEMPTS"); do
  if HEALTH_JSON="$(curl --connect-timeout 10 --max-time 30 -fsS "${SERVICE_URL}/health" 2>/dev/null)"; then
    HEALTH_STATUS="$(printf '%s' "$HEALTH_JSON" | python -c 'import json,sys; print(json.load(sys.stdin).get("status", ""))' 2>/dev/null || true)"
    HEALTH_SHA="$(printf '%s' "$HEALTH_JSON" | python -c 'import json,sys; print(json.load(sys.stdin).get("git_commit", ""))' 2>/dev/null || true)"
    HEALTH_REVISION="$(printf '%s' "$HEALTH_JSON" | python -c 'import json,sys; print(json.load(sys.stdin).get("revision", ""))' 2>/dev/null || true)"
    if [[ "$HEALTH_STATUS" == "ok" && "$HEALTH_SHA" == "$GIT_SHA" && "$HEALTH_REVISION" == "$REVISION" ]]; then
      break
    fi
  fi
  HEALTH_JSON=""
  if [[ "$attempt" -lt "$HEALTH_ATTEMPTS" ]]; then
    sleep "$HEALTH_DELAY_SECONDS"
  fi
done

if [[ -z "$HEALTH_JSON" ]]; then
  echo "Online smoke failed: /health did not report expected SHA/revision" >&2
  echo "Expected SHA: ${GIT_SHA}" >&2
  echo "Expected revision: ${REVISION}" >&2
  exit 1
fi

printf 'SERVICE_URL=%s\n' "$SERVICE_URL"
printf 'REVISION=%s\n' "$REVISION"
printf 'APP_GIT_COMMIT=%s\n' "$GIT_SHA"
printf 'HEALTH=%s\n' "$HEALTH_JSON"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  {
    echo "service_url=${SERVICE_URL}"
    echo "revision=${REVISION}"
    echo "git_commit=${GIT_SHA}"
  } >> "$GITHUB_OUTPUT"
fi
