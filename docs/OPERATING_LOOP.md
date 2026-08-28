# YuJian AI Model Factory — Operating Loop V1

## Frozen architecture

- **GitHub**: source code, configs, schemas, infrastructure-as-code, PR/Issue history.
- **Google Cloud**: GCS datasets/artifacts, GPU compute, future managed database/runtime.
- **ChatGPT/Codex**: planning, code changes, execution orchestration, QA, evaluation analysis.

## Lifecycle

1. **Ingest Batch**
   - Assign immutable `BATCH_*` ID.
   - Validate `fish_manifest.csv` and image files.
   - Calculate SHA256.
   - Upload to `gs://<bucket>/raw/batches/<batch_id>/`.
   - Never overwrite an existing Batch.

2. **Review / Species Truth**
   - Move metadata state through `INGESTED -> REVIEWING -> APPROVED/REJECTED/NEEDS_REVIEW`.
   - Images marked `pending`, `hard_case`, or `needs_review` cannot enter the main training dataset.

3. **Annotation**
   - CVAT is the annotation workspace, not the source of truth.
   - Export reviewed labels to `annotations/cvat/<batch_id>/`.

4. **Freeze Dataset**
   - Open `/datasets`, generate a preview, verify eligibility, species distribution, exclusions, parent class map and warnings, then confirm the freeze.
   - Only active Species Catalog entries are eligible. Explicit no-fish, multi-fish and non-representative near duplicates are excluded; single-fish, uncertain and not-scanned images remain eligible.
   - Split by the canonical group key; never split the same catch/event/group across train/val/test.
   - Save only `dataset_manifest.csv`, `class_map.json` and `dataset.json` to `datasets/<dataset_version>/`.
   - Never edit an existing dataset version. Create the next version instead.

5. **Register Training Run**
   - Create immutable `RUN_*` descriptor before GPU execution.
   - Bind: dataset version + Git commit + model family + parameters + seed.

6. **Train on Google Cloud GPU**
   - Worker downloads the exact frozen dataset.
   - Produces weights, metrics, logs and evaluation artifacts.
   - Uploads artifacts to `models/<model_version>/` and `evaluations/<evaluation_id>/`.

7. **Evaluate**
   - Track overall metrics, per-species metrics, confusion matrix and hard-pair metrics.
   - Failed cases enter `error_pool/` for the next acquisition cycle.

8. **Promote or Reject**
   - Model states: `EXPERIMENT -> CANDIDATE -> PRODUCTION`, or `REJECTED/ARCHIVED`.
   - Production promotion must retain lineage to run, dataset and Git commit.

## Naming convention

- Batch: `BATCH_YYYYMMDD_<SOURCE>_<NNN>`
- Dataset: `DS_M1_v0.1`, `DS_M1_v0.2`, ...
- Run: `RUN_M1_0001`, `RUN_M1_0002`, ...
- Model: `MODEL_M1_0001`, `MODEL_M1_0002`, ...

## Current migration targets

- `YujianAI_P2_Pilot200.zip` -> `BATCH_20260826_PILOT_001`
- `YujianAI_P2_Pilot200_wb.zip` -> `BATCH_20260826_WB_001`

The first dataset version should only be frozen after image/manifest verification and Species Truth review. WorkBuddy rows currently marked `pending` must not be silently promoted into training truth.
