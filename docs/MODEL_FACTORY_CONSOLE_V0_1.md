# YuJian Model Factory Console V0.1

## Scope

V0.1 deliberately covers only the shortest usable Model Factory loop:

```text
GCS incoming
  -> automatic audit
  -> immutable raw batch
  -> Registry
  -> manual visual review / Species Truth
  -> approved pool
  -> immutable Dataset Version
```

Training, CVAT, evaluation and Error Pool remain V0.2+.

## Pages

### Overview `/`

Shows:

- registered batches
- registered image count
- approved / pending / needs_review / rejected
- species distribution
- frozen dataset count
- links to the next workflow step

### Batches `/batches`

Discovers first-level folders under `gs://$GCS_BUCKET/incoming/`.

For each incoming folder the operator can set a canonical Batch ID and source. The incoming folder name does not need to equal the canonical ID.

Primary action: **准备审核**.

It runs, in order:

1. Audit incoming objects and manifest.
2. Promote the immutable batch into `raw/batches/<batch_id>/` using server-side GCS copy.
3. Sync the raw batch into Registry.

Audit artifacts are written to:

```text
cleaning/<batch_id>/auto_v1/
  audit_report.json
  review_queue.csv
  orphan_images.csv
```

The promote operation is retry-safe: existing destination objects are reused only when size/hash are consistent. `batch.json` is the immutable completion marker.

Registry sync consumes `review_queue.csv` when available:

- `CANDIDATE` -> `pending`
- `NEEDS_REVIEW` -> `needs_review`
- `AUTO_REJECT` -> `rejected`

An explicit valid review status already present in the source manifest takes precedence, so approved Pilot seed data stays approved.

### Review `/review`

One-image-at-a-time review optimized for fast manual QA.

Actions:

- Approve
- Reject
- Needs Review
- Hard Case
- edit truth species
- edit truth status
- add notes

Keyboard:

```text
A  Approve
R  Reject
H  Hard Case
N  Needs Review
-> Skip
```

Every mutation writes a `ReviewEvent` containing before/after JSON.

### Dataset `/datasets`

Displays Approved Pool by batch and species.

The operator selects source batches and freezes a Dataset Version. Only `review_status=approved` rows are eligible.

The freeze is deterministic at group level and writes:

```text
datasets/<dataset_version>/
  dataset_manifest.csv
  dataset.json
```

The GCS `dataset.json` marker and Registry DatasetVersion make the version immutable.

## Data safety rules

- `incoming/` is never modified by Console V0.1.
- audit output is derived and may be regenerated.
- `raw/batches/<batch_id>/batch.json` makes the raw batch immutable.
- manual review never overwrites source manifests; it lives in Registry + ReviewEvent.
- Dataset freeze accepts approved data only.
- Dataset versions cannot overwrite an existing GCS/Registry version.

## Runtime configuration

```text
GCP_PROJECT_ID=gemini-api-project-503706
GCS_BUCKET=yujian-model-factory-571785698442
REGISTRY_DB_URL=sqlite:///./var/yujian_registry.db   # local only
APP_GIT_COMMIT=<deployed commit sha>
```

For Cloud Run, configure `REGISTRY_DB_URL` to persistent PostgreSQL / Cloud SQL. Cloud Run instance-local SQLite is not a production registry because the filesystem is ephemeral.

## Deployment boundary

V0.1 code is container-ready through the repository Dockerfile. Production deployment should provide:

- Cloud Run service identity with GCS access
- persistent PostgreSQL / Cloud SQL registry
- `GCS_BUCKET`
- `REGISTRY_DB_URL`
- `APP_GIT_COMMIT`
- authenticated/private operator access

No service-account JSON key is required or recommended.
