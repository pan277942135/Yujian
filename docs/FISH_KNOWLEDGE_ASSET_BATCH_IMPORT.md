# Fish Knowledge Asset Batch Import V1

## Purpose

The importer moves a GCS staging folder into versioned Fish Knowledge assets. It never accepts a large ZIP in a Cloud Run request and it never changes imported Cover/Card records to ACTIVE automatically.

## Cloud workflow

1. Create a folder under gs://<GCS_BUCKET>/fish-assets/imports/<batch_id>/.
2. Unpack the asset package in Cloud Shell/Work and copy the files into that folder. A canonical folder uses an existing species_id; numbered Chinese folders are accepted as a compatibility format.
3. Open /fish-knowledge/assets/import, enter the GCS folder, and click Scan.
4. Resolve every INVALID row. WARNING rows require the explicit allow-warnings confirmation.
5. Review the matrix and click Execute. Source objects are read from GCS, decoded and converted through the existing Pillow/WebP pipeline, then copied to fish-assets/fish-knowledge/<species_id>/<asset-directory>/vN.webp.
6. Imported versions are stored as DRAFT. Use the per-item set ACTIVE action only after Admin QA.

## File contract

The preferred names are:

- 00_cover.(png|jpg|jpeg|webp)
- 01_hero.(png|jpg|jpeg|webp)
- 02_identification.(png|jpg|jpeg|webp)
- 03_ecology.(png|jpg|jpeg|webp)
- 04_gear.(png|jpg|jpeg|webp)
- 05_skill.(png|jpg|jpeg|webp)

README and manifest files at the batch root are ignored as auxiliary metadata. GCS objects remain the source of truth.

Species folders resolve against the existing fish_species, aliases, SpeciesCatalog, and the compatibility alias baitiao -> sharpbelly. No new species identity is created by the importer.

## Validation and idempotency

The scan records per-item source object, MIME, dimensions, aspect ratio, file size, SHA-256, errors, warnings and target object. It rejects unknown species, unknown asset type, duplicate slots, corrupt images, unsupported formats, obvious non-square images, files over 10 MB and severely low resolution. Mild ratio/resolution deviations are WARNING.

A matching species_id + asset_type + sha256 already present in a previous imported version is marked ASSET_ALREADY_EXISTS and skipped during Execute. Imported items are also skipped on retry. Failed items can be retried without repeating successful items.

## API

All endpoints use the existing console authentication middleware:

- POST /api/v1/admin/fish/assets/import-batches
- POST /api/v1/admin/fish/assets/import-batches/{batch_id}/scan
- GET /api/v1/admin/fish/assets/import-batches/{batch_id}
- GET /api/v1/admin/fish/assets/import-batches
- POST /api/v1/admin/fish/assets/import-batches/{batch_id}/execute
- POST /api/v1/admin/fish/assets/import-batches/{batch_id}/retry
- GET /api/v1/admin/fish/assets/import-batches/{batch_id}/items/{item_id}/source
- POST /api/v1/admin/fish/assets/import-batches/{batch_id}/versions/{version_id}/activate

Execute requires READY, always blocks INVALID, and requires allow_warnings=true when warnings exist.

## DRAFT isolation

The public /api/v1/fish/species/{species_id}/detail endpoint continues to read only ACTIVE species, Cover and Cards. Versioned imported objects are served by the managed media endpoint only after the corresponding version has been explicitly activated. Structured card content is not overwritten by Execute.

## UAT / production checklist

Use FK_ASSET_UAT_001 with White Stripe/Sharpbelly and Crucian Carp first, then FK_ASSET_20260913_001 for the complete 14 x 6 package. Record scan totals, imported/skipped/failed counts, target paths, completion before/after, public DRAFT isolation and one explicit ACTIVE smoke test.
