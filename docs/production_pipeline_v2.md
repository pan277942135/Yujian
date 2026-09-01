# YuJian Production Pipeline v2

This document describes the additive Detector → Crop → Classifier data
contract.  Existing Android, detector, classifier, Dataset Freeze and legacy
whole-image artifacts remain supported.

## Runtime and training contracts

```text
Android:  original image → DET_FISH_v0.1 → candidate_bbox → quality gate
          → expanded crop (0.15) → classifier → user feedback

Training: original image → reviewed accepted_bbox → YOLO detector dataset
          → regenerated crop → DS_CROP_M1_v0.1 → CROP_CLASSIFIER_V1
```

`candidate_bbox` is an immutable model proposal.  It is never written as a
training label.  A reviewer must call the inference review endpoint with an
explicit normalized `accepted_bbox`; only `ACCEPTED` or `TRAINING_READY`
records are eligible for dataset generation.

## Android inference asset

Each recognition receives one UUID-style `image_id` (`yj_img_<UUID>`).  The
same identifier links the original JPEG, optional crop JPEG, JSON
`INFERENCE_RECORD_V2`, feedback and backend asset.  App-private storage is
retryable and uses atomic writes:

```text
filesDir/yujian/inference/YYYY/MM/DD/
  <image_id>.jpg
  <image_id>_crop.jpg
  <image_id>.json
```

The multipart upload is authenticated with `X-YuJian-Ingest-Key`; the
pre-existing `/api/feedback/ingest` endpoint is unchanged.

## Backend state and storage

`POST /api/v1/inference/upload` stores immutable objects under
`app_feedback/inference/YYYY/MM/DD/` and records an `InferenceAsset`.  Writes
are hash-idempotent: the same path and hash is skipped; a different hash for
the same `image_id` returns a conflict and never overwrites the object.

The review endpoint advances the additive state machine:

```text
RECEIVED → CANDIDATE → REVIEW_REQUIRED → ACCEPTED → TRAINING_READY
```

The upload may route a record with a detector candidate directly into the
review queue (`REVIEW_REQUIRED`); this is an operational shortcut, not an
acceptance.  User feedback is also materialized into the existing
`FeedbackEvent` review pool, while the immutable inference JSON remains the
source artifact.

## Dataset builders

`trainer.build_reviewed_datasets` contains two review-gated builders:

- `build_reviewed_detector_dataset` writes one-class YOLO `fish` images and
  labels as `DS_DET_FISH_v0.1`.
- `build_crop_dataset` regenerates crops from the original image and accepted
  box, writes species directories and `metadata/crop_manifest.csv` as
  `DS_CROP_M1_v0.1`, and emits a class map.  The manifest now records
  `source_image_id`, `bbox_source=accepted_review`, the expanded crop bounds,
  and the crop pixel dimensions.

`trainer.crop_dataset_validator` enforces the crop training gate before a
dataset is used: the crop file must exist, a reviewed species and normalized
`accepted_bbox` must be present, and `image_id` values must be unique.  A
`candidate_bbox` is never a valid substitute.  The training worker runs the
same validator again after materialising remote files, so a missing GCS crop
cannot enter `CROP_CLASSIFIER_V1` by accident.

Both builders report excluded candidates and safety flags.  Neither changes
labels, freezes a dataset, creates a batch or starts training.

The read-only `/crop-qa` page and `/api/crop-qa` endpoint show the original
image with the human accepted box alongside its crop preview.  Media access
is limited to `ACCEPTED`/`TRAINING_READY` assets with a valid accepted box;
the page has no label, review, or Freeze mutation controls.

## Classifier pipelines and preprocessing

`CROP_CLASSIFIER_V1` is the default new training pipeline and refuses rows
without `input_type=crop` and the matching pipeline marker.  It uses the same
224px RGB ImageNet normalization and centered letterbox/padding contract as
the Android classifier.  `WHOLE_IMAGE_V1` remains available for legacy
baselines and is explicitly marked in model/run metadata.

Model registry records carry `pipeline_type`, `detector_version`,
`crop_version`, `classifier_version` and `dataset_version`.  Training still
requires a frozen dataset and is never triggered by an inference upload or a
review action.

## Intelligence hand-off

Model Intelligence can consume the evaluation artifact contract and uploaded
inference assets.  Its Detector Error Analyzer reports misses, multiple fish,
misaligned/over-sized/under-sized boxes and occlusion as advisory
`DETECTOR_IMPROVEMENT` tasks.  These tasks do not create batches, modify
labels, freeze datasets or launch training automatically.
