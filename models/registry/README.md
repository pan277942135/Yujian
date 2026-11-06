# YuJian frozen model artifacts

Model binaries are published as versioned GitHub Releases in this repository.

## Frozen versions

- Detector: `DET_FISH_v0.3` — release tag `detector-model-v0.3`
- Classifier: `MODEL_CROP_M1_v0.2` — source release tag `classifier-model-v0.2`
- Existing Android classifier runtime asset: release tag `mobile-model-v0.2`

## Retrieval

Use versioned release assets, not a floating branch URL. Download the bundle or individual runtime asset and verify SHA-256 against `MODEL_LOCK.json` attached to each release.

## Freeze policy

- Release tags are created only if they do not already exist.
- Publishing refuses to overwrite an existing tag.
- GitHub Actions compares GCS bytes with the exact SHA-256 values of the uploaded source packages.
- The workflow downloads published assets again and verifies their SHA-256 before reporting success.
- Public metadata excludes source-image audit rows, review IDs, per-image source hashes, and internal GCS URIs.
- Build/test status is not a substitute for runtime or visual acceptance.
