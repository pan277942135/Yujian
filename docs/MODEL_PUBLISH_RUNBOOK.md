# Model Promotion / Release Runbook

The Console endpoint `POST /api/models/{model_id}/publish` creates one durable
publish job and dispatches `.github/workflows/mobile-model-analysis.yml`.
GitHub Actions downloads the exact registered GCS model contract, exports and
validates TFLite, archives the versioned artifact, replaces the stable Release
asset, verifies the remote bytes, writes `publish_manifest.json`, and calls the
Console with a one-time opaque token.

## Required secret

`YUJIAN_GITHUB_RELEASE_TOKEN` must be injected into Cloud Run from Google Secret
Manager. Do not configure it as a plain repository variable or commit it to an
environment file. The fine-grained token only needs access to
`pan277942135/Yujian` with Actions read/write and Metadata read permissions;
the dispatched workflow uses its own short-lived `GITHUB_TOKEN` to update the
Release.

Example one-time operator setup (supply the token through stdin):

```bash
gcloud secrets create yujian-github-release-token \
  --project gemini-api-project-503706 \
  --replication-policy automatic

gcloud secrets versions add yujian-github-release-token \
  --project gemini-api-project-503706 \
  --data-file=-

gcloud secrets add-iam-policy-binding yujian-github-release-token \
  --project gemini-api-project-503706 \
  --member serviceAccount:571785698442-compute@developer.gserviceaccount.com \
  --role roles/secretmanager.secretAccessor

gcloud run services update yujian-model-factory-console \
  --project gemini-api-project-503706 \
  --region asia-east1 \
  --update-secrets YUJIAN_GITHUB_RELEASE_TOKEN=yujian-github-release-token:latest
```

The normal UAT source deployment preserves existing Cloud Run secret bindings.

## Stable production contract

- Release tag: `mobile-model-v0.2`
- Android asset: `fish_classifier_v0_2.tflite`
- Metadata: `fish_classifier_v0_2.metadata.json`
- Version archive: `gs://.../models/{model_id}/export/{model_id}.tflite`
- Publish evidence: `gs://.../models/{model_id}/publish/publish_manifest.json`

The production pointer changes only after conversion, parity, Release upload,
and remote byte comparison have all passed. A failed or timed-out job clears
the publish lock without promoting the candidate model.
