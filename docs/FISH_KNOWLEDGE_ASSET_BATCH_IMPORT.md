# Fish Knowledge Asset Batch Import V1

## Purpose

The importer moves a GCS staging folder into versioned Fish Knowledge assets and binds the resulting image URL to the existing Cover/Card draft slots. It never accepts a large ZIP in a Cloud Run request and it never changes imported Cover/Card records to ACTIVE automatically.

## Local browser workflow

1. Open /fish-knowledge/assets/import and click **选择文件夹**. Select the package root folder; Chrome sends each file with its relative path.
2. Enter a unique batch ID, click **上传文件夹**, and wait for every file to finish. Files are uploaded one by one to gs://<GCS_BUCKET>/fish-assets/imports/<batch_id>/; a large ZIP is never sent to Cloud Run.
3. Click **扫描并预检** only after upload completes. The preview records per-file errors and warnings before any import is executed.
4. Resolve every INVALID row. WARNING rows require the explicit allow-warnings confirmation.

## Cloud compatibility workflow

Existing staging folders remain supported: create or unpack a folder under gs://<GCS_BUCKET>/fish-assets/imports/<batch_id>/, enter that GCS URI under the compatibility section, and click **使用此 GCS 目录并扫描**.
5. Review the matrix and click Execute. Source objects are read from GCS, decoded and converted through the existing Pillow/WebP pipeline, then copied to fish-assets/fish-knowledge/<species_id>/<asset-directory>/vN.webp and bound to the existing Cover/Card image field only; structured text is preserved.
6. Imported versions and bound Cover/Card records are stored as DRAFT. Use the per-item set ACTIVE action only after Admin QA. An already completed batch can use `同步到鱼鉴内容` to repair a prior import that only created version rows.

## File contract

The preferred names are:

- Root folder: any local package folder name is allowed; the importer preserves the relative path and resolves the first nested directory matching a known species.
- Recommended canonical layout:
  ```
  <package>/
    sharpbelly/
      00_cover.png
      01_hero.png
      02_identification.png
      03_ecology.png
      04_gear.png
      05_skill.png
  ```
- Numbered/Chinese species folders such as `01_白条/` and `02_鲫鱼/` remain supported.
- 00_cover.(png|jpg|jpeg|webp)
- 01_hero.(png|jpg|jpeg|webp)
- 02_identification.(png|jpg|jpeg|webp)
- 03_ecology.(png|jpg|jpeg|webp)
- 04_gear.(png|jpg|jpeg|webp)
- 05_skill.(png|jpg|jpeg|webp)

README.txt, manifest.csv and asset_manifest.csv are ignored as auxiliary metadata at any folder depth. Supported image extensions are PNG, JPG, JPEG and WEBP; one image must not exceed 10 MB. Recommended minimum size is 1024px and near-square images are preferred. Partial species packages are allowed and preview completion is shown as x/6. GCS objects remain the source of truth.

Species folders resolve against the existing fish_species, aliases, SpeciesCatalog, and the compatibility alias baitiao -> sharpbelly. No new species identity is created by the importer.

## Validation and idempotency

The scan records per-item source object, MIME, dimensions, aspect ratio, file size, SHA-256, errors, warnings and target object. It rejects unknown species, unknown asset type, duplicate slots, corrupt images, unsupported formats, obvious non-square images, files over 10 MB and severely low resolution. Mild ratio/resolution deviations are WARNING.

A matching species_id + asset_type + sha256 already present in a previous imported version is marked ASSET_ALREADY_EXISTS and skipped during Execute. Imported items are also skipped on retry. Failed items can be retried without repeating successful items.

## API

All endpoints use the existing console authentication middleware:

- POST /api/v1/admin/fish/assets/import-batches
- POST /api/v1/admin/fish/assets/import-batches/local
- POST /api/v1/admin/fish/assets/import-batches/{batch_id}/upload (multipart: relative_path + file)
- POST /api/v1/admin/fish/assets/import-batches/{batch_id}/scan
- GET /api/v1/admin/fish/assets/import-batches/{batch_id}
- GET /api/v1/admin/fish/assets/import-batches
- POST /api/v1/admin/fish/assets/import-batches/{batch_id}/execute
- POST /api/v1/admin/fish/assets/import-batches/{batch_id}/retry
- POST /api/v1/admin/fish/assets/import-batches/{batch_id}/sync-content
- GET /api/v1/admin/fish/assets/import-batches/{batch_id}/items/{item_id}/source
- GET /api/v1/admin/fish/assets/import-batches/{batch_id}/versions/{version_id}/preview
- POST /api/v1/admin/fish/assets/import-batches/{batch_id}/versions/{version_id}/activate

Execute requires READY, always blocks INVALID, and requires allow_warnings=true when warnings exist.

## DRAFT isolation

The public /api/v1/fish/species/{species_id}/detail endpoint continues to read only ACTIVE species, Cover and Cards. DRAFT imported images are previewed through the authenticated Admin preview endpoint; the public managed media endpoint serves a version only after it has been explicitly activated. Structured card content is not overwritten by Execute.

## UAT / production checklist

Use FK_ASSET_UAT_001 with White Stripe/Sharpbelly and Crucian Carp first, then FK_ASSET_20260913_001 for the complete 14 x 6 package. Record scan totals, imported/skipped/failed counts, target paths, completion before/after, public DRAFT isolation and one explicit ACTIVE smoke test.
