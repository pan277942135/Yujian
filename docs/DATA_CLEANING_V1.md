# YuJian AI Data Cleaning V1

## Goal

Prevent low-quality or structurally invalid scraped data from entering reviewed datasets or training datasets.

Pipeline:

```text
incoming/<batch>/
  -> audit_incoming_batch.py
  -> cleaning/<canonical_batch>/auto_v1/
  -> human visual review
  -> reviewed/batches/<canonical_batch>/
  -> annotation
  -> dataset freeze
```

## Stage A - deterministic automated audit

No image is deleted or modified. The script only reads GCS metadata and the small fish_manifest.csv, then creates audit artifacts.

Checks:

- exactly one fish_manifest.csv
- required manifest columns
- manifest row/image object linkage
- orphan images not referenced by manifest
- manifest rows whose image object is missing
- duplicate image_id
- duplicate file_name
- duplicate source_url
- exact duplicate image bytes using GCS MD5
- empty or non-target claimed_species

Statuses:

- `CANDIDATE`: structurally valid; still requires visual review
- `NEEDS_REVIEW`: ambiguous metadata/species issue
- `AUTO_REJECT`: deterministic hard failure; do not send to training

Artifacts:

- `audit_report.json`
- `review_queue.csv`
- `orphan_images.csv`

## Stage B - visual review

Every `CANDIDATE` must still be checked for the actual fish-recognition use case. Approve only when the image is suitable as training evidence.

Primary checks:

- fish is visible and recognizable
- single main fish preferred for M1
- complete or mostly complete body; at least two of head/body/tail visible
- not severe blur/overexposure
- not cooked/processed/specimen/AI-generated
- real angling/catch domain preferred
- claimed species is plausible; hard pairs get explicit review

Recommended manual states:

- `approved`
- `needs_review`
- `rejected`
- `hard_case`

## Current incoming mapping

The historical incoming prefixes were uploaded before canonical naming was finalized:

- `incoming/BATCH_20260826_PILOT_001/` -> canonical `BATCH_20260826_PILOT_001`, source `pilot`
- `incoming/BATCH_20260826_WB_001/` -> canonical `BATCH_20260826_DB_001`, source `doubao`
- `incoming/BATCH_20260826_WB_002/` -> canonical `BATCH_20260826_WB_001`, source `workbuddy`

Do not rename the incoming folders. Canonical identity is assigned by `--batch-id` and `--source`.

## Commands

### Doubao

```bash
python scripts/audit_incoming_batch.py \
  --bucket yujian-model-factory-571785698442 \
  --batch-id BATCH_20260826_DB_001 \
  --source doubao \
  --incoming-prefix incoming/BATCH_20260826_WB_001/ \
  --write-report
```

### WorkBuddy

```bash
python scripts/audit_incoming_batch.py \
  --bucket yujian-model-factory-571785698442 \
  --batch-id BATCH_20260826_WB_001 \
  --source workbuddy \
  --incoming-prefix incoming/BATCH_20260826_WB_002/ \
  --write-report
```

### Pilot

```bash
python scripts/audit_incoming_batch.py \
  --bucket yujian-model-factory-571785698442 \
  --batch-id BATCH_20260826_PILOT_001 \
  --source pilot \
  --incoming-prefix incoming/BATCH_20260826_PILOT_001/ \
  --write-report
```

## Important

`AUTO_REJECT` is only for deterministic structural failures. Visual-quality rejection remains a review decision. Raw/incoming data is preserved for traceability.
