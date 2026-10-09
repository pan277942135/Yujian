# 01 — Environment Audit

**Audit window:** 2026-10-09 15:19–15:23 Asia/Shanghai  
**CMS candidate:** `pan277942135/Yujian` / `fix/fish-knowledge-cms-v1-4-publication-closure` @ `ea7f50d5b4d5beadf51564bc2f0d75854ee49eed`; PR #159 is Open / Draft / Unmerged.  
**Android candidate:** `pan277942135/Yujian_App` / `fix/fish-guide-published-asset-render-parity-v1` @ `505690f0a681334b6c3411de50f55e026076b464`; PR #125 is Open / Draft / Unmerged.

## Results

| Check | Result | Evidence |
|---|---|---|
| Fresh CMS checkout | PASS | Clean clone at the exact authoritative HEAD above. |
| CMS GitHub CI | PASS | Run `37897272676` (`ci`, run 1193), validation job success. It is a CI validation, not a deployment. |
| Android GitHub CI | PASS for build checks | Run `37886891748` (`Android CI`, run 1535): unit tests, Android runtime harness contract tests, assembleDebug, AndroidTest APK compilation, lint, and artifact upload succeeded. The separate `android-runtime` job was **skipped**. |
| GCP CLI / direct cloud identity | BLOCKED | `gcloud`, `gsutil`, `psql`, Docker and Terraform are not installed. No Google Cloud credential environment variables or standard ADC credential files were present. No GCP/Cloud Run/Cloud SQL/GCS connector is available in this session. |
| Separate Staging identifiers | BLOCKED | No verified `STAGING_PROJECT`, `STAGING_CLOUD_RUN_SERVICE`, `STAGING_DATABASE`, `STAGING_GCS_BUCKET`, or `STAGING_SERVICE_ACCOUNT` was available. |
| Staging isolation | FAIL CLOSED | The repository's workflow called `UAT Deploy` is configured for project `gemini-api-project-503706` (project number `571785698442`), Cloud Run service `yujian-model-factory-console`, region `asia-east1`; the deployment script defaults to that same project/service and bucket `yujian-model-factory-571785698442`. Those match the supplied production service context. Its script runs `gcloud run deploy`, reads the existing service configuration and Secret Manager, and updates runtime environment variables. It is not safe to invoke as Staging. |
| Secret values | NOT READ | No Secret Manager values, credentials, tokens or service-account keys were retrieved. |
| Production writes/deployment | NOT RUN | No production deployment, SQL, GCS mutation, migration, upload, activation, or publication was attempted. |

The only production interaction in this audit was unauthenticated public API and image HTTP **GET** requests to the user-supplied production API. The JSON and image response metadata are retained in the sibling snapshots. This does not expose the live Cloud Run revision, IAM policy, database configuration, or GCS metadata.

## Gate 1

`BLOCKED_STAGING_ENVIRONMENT`. Isolation cannot be proven, and the available repository “UAT” deployment target is the production project/service. No deployment or database write was made. To unblock, the environment owner must provide verified isolated Staging resource IDs and a deployment path that cannot update production service traffic or write production data.
