#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-gemini-api-project-503706}"
PROJECT_NUMBER="${PROJECT_NUMBER:-571785698442}"
REGION="${REGION:-asia-east1}"
SERVICE="${SERVICE:-yujian-model-factory-console}"
DEPLOY_SHA="${DEPLOY_SHA:-}"
BUILD_SA="${BUILD_SA:-${PROJECT_NUMBER}-compute@developer.gserviceaccount.com}"

command -v gcloud >/dev/null 2>&1 || { echo "gcloud is required" >&2; exit 2; }
command -v curl >/dev/null 2>&1 || { echo "curl is required" >&2; exit 2; }
[[ "$DEPLOY_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "DEPLOY_SHA must be a full commit SHA" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "$DEPLOY_SHA" ]] || { echo "checkout SHA does not match DEPLOY_SHA" >&2; exit 2; }

SERVICE_JSON="$(gcloud run services describe "$SERVICE" --project "$PROJECT_ID" --region "$REGION" --format=json)"
readarray -t SERVICE_STATE < <(printf '%s' "$SERVICE_JSON" | python -c '
import json,sys
d=json.load(sys.stdin)
status=d.get("status") or {}
spec=d.get("spec") or {}
template=spec.get("template") or {}
template_annotations=((template.get("metadata") or {}).get("annotations") or {})
service_annotations=((d.get("metadata") or {}).get("annotations") or {})
connection_value=(template_annotations.get("run.googleapis.com/cloudsql-instances")
                  or service_annotations.get("run.googleapis.com/cloudsql-instances") or "")
connections=[x.strip() for x in connection_value.split(",") if x.strip()]
print(status.get("latestReadyRevisionName", ""))
print(status.get("url", ""))
print(connections[0] if len(connections)==1 else "")
print(len(connections))
')
PREVIOUS_REVISION="${SERVICE_STATE[0]:-}"
SERVICE_URL="${SERVICE_STATE[1]:-}"
SQL_CONNECTION="${SERVICE_STATE[2]:-}"
SQL_CONNECTION_COUNT="${SERVICE_STATE[3]:-0}"
if [[ -z "$PREVIOUS_REVISION" || -z "$SERVICE_URL" ]]; then
  echo "Existing Cloud Run service has no ready revision or URL; refusing deployment" >&2
  exit 2
fi
if [[ "$SQL_CONNECTION_COUNT" != "1" || -z "$SQL_CONNECTION" ]]; then
  echo "Expected exactly one Cloud SQL connection on the existing service; refusing deployment" >&2
  exit 2
fi

SQL_INSTANCE="${SQL_CONNECTION##*:}"
INSTANCE_STATE="$(gcloud sql instances describe "$SQL_INSTANCE" --project "$PROJECT_ID" --format='value(state)')"
[[ "$INSTANCE_STATE" == "RUNNABLE" ]] || { echo "Cloud SQL instance is not RUNNABLE: $INSTANCE_STATE" >&2; exit 2; }

# The synchronous backup is a hard gate. The script never restores over the
# live instance; on a failed health check it only returns Cloud Run traffic to
# the pre-deploy revision. The backup ID is recorded for an operator-directed
# restore to a separate recovery instance if needed.
BACKUP_DESCRIPTION="fishcms-catch-${DEPLOY_SHA:0:12}-$(date -u +%Y%m%dT%H%M%SZ)"
gcloud sql backups create \
  --instance "$SQL_INSTANCE" \
  --project "$PROJECT_ID" \
  --description "$BACKUP_DESCRIPTION" \
  --quiet
BACKUP_LIST="$(gcloud sql backups list --instance "$SQL_INSTANCE" --project "$PROJECT_ID" --limit=10 --format=json)"
readarray -t BACKUP_STATE < <(printf '%s' "$BACKUP_LIST" | python -c '
import json,sys
description=sys.argv[1]
rows=json.load(sys.stdin)
row=next((x for x in rows if x.get("description")==description), None)
print(row.get("id", "") if row else "")
print(row.get("status", "") if row else "")
' "$BACKUP_DESCRIPTION")
BACKUP_ID="${BACKUP_STATE[0]:-}"
BACKUP_STATUS="${BACKUP_STATE[1]:-}"
if [[ -z "$BACKUP_ID" || "$BACKUP_STATUS" != "SUCCESSFUL" ]]; then
  echo "Cloud SQL backup did not reach SUCCESSFUL; refusing deployment (id=${BACKUP_ID:-missing}, status=${BACKUP_STATUS:-missing})" >&2
  exit 2
fi

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  {
    echo "### Backend-only deploy preflight"
    echo "- Target SHA: \`$DEPLOY_SHA\`"
    echo "- Existing revision: \`$PREVIOUS_REVISION\`"
    echo "- Cloud SQL instance: \`$SQL_INSTANCE\`"
    echo "- On-demand backup: \`$BACKUP_ID\` (\`$BACKUP_STATUS\`)"
    echo "- Restore policy: no in-place automatic restore; recovery is operator-directed to a separate instance."
  } >> "$GITHUB_STEP_SUMMARY"
fi

# This deploy updates only the existing Backend/Console service. It does not
# boot GPU workers, start training jobs, alter their configuration, or touch
# fish assets. App startup runs the additive, idempotent database migrations.
gcloud run deploy "$SERVICE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --source . \
  --build-service-account "projects/${PROJECT_ID}/serviceAccounts/${BUILD_SA}" \
  --add-cloudsql-instances "$SQL_CONNECTION" \
  --memory 4Gi \
  --timeout=1200s \
  --min 1 \
  --no-cpu-throttling \
  --update-env-vars="APP_GIT_COMMIT=${DEPLOY_SHA}" \
  --quiet

DEPLOYED=false
for attempt in $(seq 1 24); do
  SERVICE_JSON="$(gcloud run services describe "$SERVICE" --project "$PROJECT_ID" --region "$REGION" --format=json)"
  SERVICE_URL="$(printf '%s' "$SERVICE_JSON" | python -c 'import json,sys; print(json.load(sys.stdin).get("status",{}).get("url", ""))')"
  if [[ -n "$SERVICE_URL" ]] && HEALTH_JSON="$(curl --connect-timeout 10 --max-time 30 -fsS "$SERVICE_URL/health/deploy" 2>/dev/null)"; then
    if printf '%s' "$HEALTH_JSON" | python -c 'import json,sys; assert json.load(sys.stdin).get("git_commit")==sys.argv[1]' "$DEPLOY_SHA"; then
      DEPLOYED=true
      break
    fi
  fi
  sleep 5
done

if [[ "$DEPLOYED" != "true" ]]; then
  echo "New revision did not report the requested SHA; restoring traffic to $PREVIOUS_REVISION" >&2
  gcloud run services update-traffic "$SERVICE" \
    --project "$PROJECT_ID" \
    --region "$REGION" \
    --to-revisions="${PREVIOUS_REVISION}=100" \
    --quiet
  exit 1
fi

LATEST_REVISION="$(printf '%s' "$SERVICE_JSON" | python -c 'import json,sys; print(json.load(sys.stdin).get("status",{}).get("latestReadyRevisionName", ""))')"
echo "Backend deployment passed: service=$SERVICE revision=$LATEST_REVISION sha=$DEPLOY_SHA backup_id=$BACKUP_ID"
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  {
    echo "- New revision: \`$LATEST_REVISION\`"
    echo "- Health SHA: \`$DEPLOY_SHA\`"
    echo "- Cloud SQL backup retained: \`$BACKUP_ID\`"
  } >> "$GITHUB_STEP_SUMMARY"
fi
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  echo "revision=$LATEST_REVISION" >> "$GITHUB_OUTPUT"
  echo "service_url=$SERVICE_URL" >> "$GITHUB_OUTPUT"
  echo "backup_id=$BACKUP_ID" >> "$GITHUB_OUTPUT"
fi
