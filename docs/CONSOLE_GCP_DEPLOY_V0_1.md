# YuJian Model Factory Console V0.1 — GCP Deployment

This deployment is intentionally simple and low-cost for the first working data flywheel.

## Runtime

- Cloud Run service: `yujian-model-factory-console`
- Region: `asia-east1`
- Cloud SQL PostgreSQL 15 instance: `yujian-registry`
- Initial machine tier: `db-f1-micro`
- Database: `yujian_registry`
- Runtime service account: `yujian-model-factory@<project>.iam.gserviceaccount.com`
- GCS: existing Model Factory bucket
- Secrets: Secret Manager

The Console remains publicly reachable at the Cloud Run URL, but operator pages/API/media are protected by the application access key stored in Secret Manager. `/health` stays public for health checks.

## One-time deployment

From Cloud Shell in the repository root:

```bash
cd ~/Yujian
git pull origin main
chmod +x scripts/deploy_console_gcp.sh
./scripts/deploy_console_gcp.sh
```

The script is idempotent enough for normal redeployments. It:

1. enables required APIs;
2. creates Cloud SQL if absent;
3. creates the application database and database user;
4. creates/uses Secret Manager secrets;
5. grants the runtime service account Cloud SQL / Secret Manager / GCS access;
6. builds the Dockerfile from source;
7. deploys Cloud Run with the Cloud SQL Unix socket;
8. performs `/health` verification;
9. prints the Console URL.

## Console access key

Retrieve it only when needed:

```bash
gcloud secrets versions access latest \
  --secret=yujian-console-access-key \
  --project=gemini-api-project-503706
```

Do not place the access key or DB password in GitHub.

## Daily operation after deployment

Normal data work should happen only in the browser:

```text
Overview
  -> Batches
  -> Review
  -> Species
  -> Feedback
  -> Dataset
```

Cloud Shell becomes deployment / troubleshooting only.

## Current scaling boundary

`db-f1-micro` and `max-instances=2` are intentionally conservative for V0.1. Upgrade Cloud SQL and Cloud Run sizing when concurrency, review traffic, or the online-feedback ingestion rate increases. Dataset images and artifacts remain in GCS; the database stores registry/state, not image bytes.

## Dataset durability model

- Approved images accumulate in the Master Pool.
- Dataset versions are immutable cumulative snapshots.
- Each Dataset freezes its own `class_map.json`.
- New Species Catalog entries can be activated without rewriting old Dataset/Model versions.
- User inference feedback can be materialized into a normal feedback Batch and return to Review.
