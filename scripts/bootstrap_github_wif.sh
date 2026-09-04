#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-gemini-api-project-503706}"
PROJECT_NUMBER="${PROJECT_NUMBER:-571785698442}"
GITHUB_REPOSITORY="${GITHUB_REPOSITORY:-pan277942135/Yujian}"
POOL_ID="${POOL_ID:-github-actions}"
PROVIDER_ID="${PROVIDER_ID:-yujian-main}"
DEPLOY_SA_NAME="${DEPLOY_SA_NAME:-yujian-github-deployer}"
DEPLOY_SA="${DEPLOY_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
RUNTIME_SA="${RUNTIME_SA:-yujian-model-factory@${PROJECT_ID}.iam.gserviceaccount.com}"
BUILD_SA="${BUILD_SA:-${PROJECT_NUMBER}-compute@developer.gserviceaccount.com}"

log() { printf '\n==> %s\n' "$*"; }

command -v gcloud >/dev/null 2>&1 || { echo "gcloud is required" >&2; exit 2; }

gcloud config set project "$PROJECT_ID" >/dev/null

log "Enable APIs required by GitHub OIDC/WIF and Cloud Run source deploy"
gcloud services enable \
  iamcredentials.googleapis.com \
  sts.googleapis.com \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  serviceusage.googleapis.com \
  --project "$PROJECT_ID" >/dev/null

log "Ensure GitHub deployer service account"
if ! gcloud iam service-accounts describe "$DEPLOY_SA" --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$DEPLOY_SA_NAME" \
    --display-name="YuJian GitHub UAT Deployer" \
    --project "$PROJECT_ID"
fi

log "Grant least-privilege source deploy roles"
for role in roles/run.sourceDeveloper roles/serviceusage.serviceUsageConsumer; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${DEPLOY_SA}" \
    --role="$role" \
    --condition=None >/dev/null
done

# The deployer must be able to attach the existing Cloud Run runtime identity.
gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_SA" \
  --project "$PROJECT_ID" \
  --member="serviceAccount:${DEPLOY_SA}" \
  --role="roles/iam.serviceAccountUser" >/dev/null

# `gcloud run deploy --source` submits a Cloud Build using BUILD_SA. Google Cloud
# requires two independent permissions: BUILD_SA needs roles/run.builder on the
# project, and the deployer needs iam.serviceAccounts.actAs on that build identity.
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${BUILD_SA}" \
  --role="roles/run.builder" \
  --condition=None >/dev/null

gcloud iam service-accounts add-iam-policy-binding "$BUILD_SA" \
  --project "$PROJECT_ID" \
  --member="serviceAccount:${DEPLOY_SA}" \
  --role="roles/iam.serviceAccountUser" >/dev/null

log "Ensure Workload Identity Pool"
if ! gcloud iam workload-identity-pools describe "$POOL_ID" \
  --location=global --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud iam workload-identity-pools create "$POOL_ID" \
    --location=global \
    --display-name="GitHub Actions" \
    --description="OIDC identities for GitHub Actions" \
    --project "$PROJECT_ID"
fi

ATTRIBUTE_MAPPING="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref"
ATTRIBUTE_CONDITION="assertion.repository=='${GITHUB_REPOSITORY}' && assertion.ref=='refs/heads/main'"

log "Ensure GitHub OIDC provider restricted to ${GITHUB_REPOSITORY}@main"
if gcloud iam workload-identity-pools providers describe "$PROVIDER_ID" \
  --location=global \
  --workload-identity-pool="$POOL_ID" \
  --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud iam workload-identity-pools providers update-oidc "$PROVIDER_ID" \
    --location=global \
    --workload-identity-pool="$POOL_ID" \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="$ATTRIBUTE_MAPPING" \
    --attribute-condition="$ATTRIBUTE_CONDITION" \
    --project "$PROJECT_ID" >/dev/null
else
  gcloud iam workload-identity-pools providers create-oidc "$PROVIDER_ID" \
    --location=global \
    --workload-identity-pool="$POOL_ID" \
    --display-name="YuJian main deploy" \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="$ATTRIBUTE_MAPPING" \
    --attribute-condition="$ATTRIBUTE_CONDITION" \
    --project "$PROJECT_ID" >/dev/null
fi

WIF_MEMBER="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/attribute.repository/${GITHUB_REPOSITORY}"
log "Allow the repository identity to impersonate the deployer service account"
gcloud iam service-accounts add-iam-policy-binding "$DEPLOY_SA" \
  --project "$PROJECT_ID" \
  --member="$WIF_MEMBER" \
  --role="roles/iam.workloadIdentityUser" >/dev/null

PROVIDER_RESOURCE="$(gcloud iam workload-identity-pools providers describe "$PROVIDER_ID" \
  --location=global \
  --workload-identity-pool="$POOL_ID" \
  --project "$PROJECT_ID" \
  --format='value(name)')"

cat <<EOF

============================================================
YuJian GitHub OIDC/WIF bootstrap complete
============================================================
Project: ${PROJECT_ID} (${PROJECT_NUMBER})
Repository: ${GITHUB_REPOSITORY}
Provider: ${PROVIDER_RESOURCE}
Deploy service account: ${DEPLOY_SA}
Runtime service account: ${RUNTIME_SA}
Build service account: ${BUILD_SA}

No long-lived Google Cloud JSON key is required.

One GitHub repository variable enables automatic UAT deploys:
  YUJIAN_UAT_DEPLOY_ENABLED=true
EOF

if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  log "Enable automatic UAT deploy in GitHub repository variables"
  gh variable set YUJIAN_UAT_DEPLOY_ENABLED \
    --repo "$GITHUB_REPOSITORY" \
    --body "true"
  echo "GitHub variable set successfully."
  echo "To deploy current main immediately:"
  echo "  gh workflow run 'UAT Deploy' --repo ${GITHUB_REPOSITORY}"
else
  cat <<EOF

GitHub CLI is not authenticated in this shell. Set the repository variable once:
  GitHub -> Settings -> Secrets and variables -> Actions -> Variables
  YUJIAN_UAT_DEPLOY_ENABLED = true

Or, from an authenticated gh CLI:
  gh variable set YUJIAN_UAT_DEPLOY_ENABLED --repo ${GITHUB_REPOSITORY} --body true
  gh workflow run 'UAT Deploy' --repo ${GITHUB_REPOSITORY}
EOF
fi
