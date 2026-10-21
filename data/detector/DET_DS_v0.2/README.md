# DET_DS_v0.2 reviewed detector dataset

**Current repository status: `INTAKE_OPEN / NOT TRAINABLE`.** No Accepted Pool export, database snapshot, source scenes, frozen Manifest, or COCO package is checked into this directory. The previous v0.1 Google Vision pseudo-label dataset is not a v0.2 substitute.

## Source and label contract

The v0.2 builder reads the actual `datasets/accepted_pool/manifest.csv` and `metadata.json`, then reconciles every `source_review_id`, source URI, and `accepted_bbox` against current human review rows. It downloads and audits each manifest source URI even when the review row or label later fails reconciliation. It never uses `crop_path`, the materialized crop, `detector_bbox`, or Google Vision candidate boxes as detector labels.

The `accepted_bbox` and its `bbox` alias are normalized `xywh` on the EXIF-transposed full scene. `pixel_bbox` is source-scene pixel `xyxy` calculated against `source_size`; `crop_left/top/right/bottom` describe the source-scene crop rectangle and are not detector labels. The builder checks these relationships against downloaded bytes. It records the raw source SHA, EXIF orientation, and normalized dimensions for each source record. PNG sources are preserved losslessly as normalized PNG; other supported source formats are normalized to RGB high-quality JPEG (quality 95), with both raw source SHA and materialized-image SHA recorded.

The manifest records `training_eligibility_basis`: the database selection conditions (`training_eligible=true` plus the accepted human bbox review state). This field is not a claim of a separate legal consent record. A missing accepted review, reviewer, timestamp, or database training eligibility blocks the dataset.

## Split, held-out cases, and negatives

Train/validation/test assignment is deterministic and groups exact source SHA hashes with `ImageAsset.group_id`; when no group ID exists, it conservatively uses the batch ID. The former Accepted Pool `stable_hash_per_pool_key` split is ignored. Case A, Case B, `04-181200.jpg`, and the seven reviewed hard-case fixtures are excluded by source SHA.

Negative images are read only from active `det_fish_negative_pool_entries`. Each member must have a latest `ADD`/`RE_ADD` event with an operator and timestamp, a DET_FISH-scoped explicit no-fish confirmation snapshot, a complete-scene source URI, and a stored source SHA that matches the bytes downloaded during the build. The console pool is independent of `FishPresenceResult` and the frozen bulk-review workflow: machine-only `no_fish` records remain pending until an operator checks the original scene and confirms the selection in the DET_FISH add dialog. That action adds an append-only pool event; it does not rewrite the original AI result or approval record. The COCO package retains the negative image in `images` and creates no fish annotation. If the pool is empty or a member has missing evidence, a source mismatch, a positive-label conflict, or a duplicate source SHA, the build report records the issue and the gate stays blocked.

Historical v0.1 Google Vision negatives are exposed as candidates for manual review only. They are not imported into the DET_FISH pool automatically and their presence in the old pseudo-label dataset is not human confirmation.

## Build from the project's existing Cloud Shell

Check out the current PR #166 head, then run:

```bash
bash scripts/build_det_ds_v02_cloud_shell.sh
```

The script reads the live console service configuration, derives its Cloud SQL connection, DB secret reference and runtime service account, and stops if the required existing Artifact Registry or resources cannot be read. It builds and executes a dedicated CPU-only Cloud Run dataset-builder Job. It does not run the GPU training Job or grant IAM roles.

The only frozen artifacts are published to:

```text
gs://yujian-model-factory-571785698442/detector-datasets/DET_DS_v0.2/
```

The runtime service account needs existing read access to Accepted Pool and source-scene objects, write access to the versioned detector-dataset prefix, Cloud SQL connectivity, and access to the DB secret. The caller needs existing Cloud Build and Cloud Run Job permissions. If any prerequisite or data gate fails, the job reports `BLOCKED` and does not publish a ready report. Uploads use GCS generation preconditions, and the report is uploaded last as the commit marker; different existing frozen objects are never overwritten.
