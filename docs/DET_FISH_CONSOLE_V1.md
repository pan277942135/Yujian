# DET_FISH Console V1

## Menus

- /platform/model/det-fish-training: reports the live accepted-bbox count, frozen DET_DS_v0.2 artifact and training-gate blockers; only starts yujian-detector-trainer after a strict dataset gate passes. Training defaults to YOLOX-Nano, 416×416, 30 epochs, batch size 16 and an NVIDIA L4 job in asia-southeast1.
- /platform/model/det-fish-test: accepts one JPG/PNG/WEBP upload and returns a server-rendered BBox overlay, pixel XYXY, normalized XYWH, confidence, area ratio, model SHA, latency and orientation-retry trace. Uploads are not persisted or added to training data.

## V0.2 training gate

A detector-datasets/DET_DS_v0.2/bootstrap_report.json is trainable only when it states:

- dataset_version=DET_DS_v0.2 and status=FROZEN_READY_FOR_TRAINING;
- source=accepted_bbox_human_review or human_reviewed_accepted_bbox;
- human_review_verified=true;
- held_out_exclusions_verified=true;
- split_integrity=PASS;
- negative_coverage_verified=true;
- a 64-character lowercase manifest_sha256;
- an accompanying DET_DS_v0.2.tar.gz archive.

The report must contain at least 40 train images, 30 train boxes, 3 validation boxes, 1 independent test image and 1 human-reviewed no-fish negative. The gate intentionally rejects auto-generated presence boxes and v0.1 report artifacts.

## Model testing

The runtime accepts explicit model selection only for DET_FISH_v0.1 and DET_FISH_v0.2. Each model is loaded from its own GCS version prefix and validated against metadata, SHA-256, input tensor shape and the frozen recognition contract. The default production detector path remains unchanged for existing callers.

## Safety boundaries

- No production model switch, model publish, Dataset Freeze mutation or upload-to-training side effect.
- No train task is launched while the dataset gate is blocked or if a prior execution is queued/running.
- Menu code is additive and keeps the existing classifier training and inference pages intact.
