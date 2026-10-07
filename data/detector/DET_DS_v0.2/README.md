# DET_DS_v0.2 intake gate

Status: **INTAKE_OPEN / NOT TRAINABLE**. This directory is only a collection protocol. It contains no image samples and no ground-truth labels.

## Frozen boundaries

- Do not edit or rebuild `DET_DS_v0.1`.
- Keep both incident images held out from train, validation, and test.
  - Case A SHA-256: `2dd703a46b378093604e10f0add454606393138f7501774bd3bbcca75352c9fe`.
  - Case B SHA-256: `a7ed3bb0410c191364f6d853b4078aebdc2f3b9baa4da34b7b3f6e8e0e93a56b`.
- Do not accept detector-produced boxes as annotations.

## Required manually reviewed positives

Collect distinct source images for each applicable category and assign human-reviewed fish boxes:

- vertical fish orientation;
- person holding fish;
- hand near or partly occluding fish;
- night / flash capture;
- black background;
- elongated fish;
- fish occupying 8–20% of frame area;
- hook or line interference;
- cluttered vegetation or rock backgrounds.

## Matched hard negatives

Collect images with no fish for: person only, hand only, rod or hook only, rocks, vegetation, night scenes, and water reflections. A negative row must explicitly record `fish_present=false` after human review; an empty box list alone is not proof.

## Per-image intake record

Keep a version-controlled, privacy-safe manifest with one row per source image:

- stable image ID and SHA-256;
- source URI and license / permission basis;
- image dimensions and orientation policy;
- train / validation / test split and source-group ID;
- positive/negative label and hard-case tags;
- normalized xyxy fish boxes, or an explicit no-fish decision;
- reviewer ID, review date, and review status.

Keep identifiable or otherwise sensitive source images in access-controlled storage; do not commit them to this repository. Use group-wise splits so near-duplicate frames and the same source session cannot leak across train, validation, or test.

## Training gate

No `DET_FISH_v0.2` training starts until the image inventory, human review, licensing/permission, held-out exclusion, split integrity, and negative coverage have been checked and the frozen dataset package is independently validated. The v0.2 YOLOX experiment must select exact 90°, 180°, and 270° augmentations. No AP-only acceptance is allowed.
