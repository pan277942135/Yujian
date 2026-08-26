# Large Batch Upload Flow

Large datasets must not pass through Cloud Shell browser upload. Use Google Cloud Storage as the direct ingress path.

## Canonical flow

```text
Local workstation
  -> GCS incoming/<BATCH_ID>/
  -> promote_incoming_batch.py (server-side GCS copy)
  -> raw/batches/<BATCH_ID>/
  -> Review / Species Truth / Annotation / Dataset Freeze
```

## Upload requirements

Upload an **uncompressed batch folder**, not a ZIP, so promotion can happen entirely inside GCS without consuming Cloud Shell disk.

The uploaded folder must contain exactly one `fish_manifest.csv` and at least one supported image (`jpg`, `jpeg`, `png`, `webp`). Existing nested structure is preserved.

Example destination:

```text
gs://yujian-model-factory-571785698442/incoming/BATCH_20260826_WB_001/
```

## Option A: Google Cloud Console

Open Cloud Storage -> Buckets -> `yujian-model-factory-571785698442` -> create/open `incoming/BATCH_.../` -> Upload folder.

## Option B: local Google Cloud CLI

```bash
gcloud auth login
gcloud config set project gemini-api-project-503706

gcloud storage cp --recursive \
  "<LOCAL_BATCH_FOLDER>/*" \
  "gs://yujian-model-factory-571785698442/incoming/BATCH_20260826_WB_001/"
```

## Validate without modifying data

From a machine with repository code and ADC credentials:

```bash
python scripts/promote_incoming_batch.py \
  --bucket yujian-model-factory-571785698442 \
  --batch-id BATCH_20260826_WB_001 \
  --source workbuddy \
  --incoming-prefix incoming/BATCH_20260826_WB_001/ \
  --dry-run
```

Check `image_count`, `manifest_rows`, and object listing.

## Promote into immutable raw storage

```bash
python scripts/promote_incoming_batch.py \
  --bucket yujian-model-factory-571785698442 \
  --batch-id BATCH_20260826_WB_001 \
  --source workbuddy \
  --incoming-prefix incoming/BATCH_20260826_WB_001/
```

Promotion uses GCS server-side copy: the dataset does not download to Cloud Shell/local disk.

The script records object generation, MD5 (when present), CRC32C and size into `batch.json`. It refuses to overwrite an existing `raw/batches/<BATCH_ID>/batch.json`.

After verification, the incoming copy can be removed with `--delete-source`, but keeping it temporarily is safer during rollout.
