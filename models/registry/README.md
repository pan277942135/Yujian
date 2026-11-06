# YuJian frozen model artifacts

Model binaries are published as versioned GitHub Releases in this repository.

## Frozen versions

- Detector: `DET_FISH_v0.3` — release tag `detector-model-v0.3`
- Classifier: `MODEL_CROP_M1_v0.2` — source release tag `classifier-model-v0.2`
- Existing Android classifier runtime asset: release tag `mobile-model-v0.2`

## Publish procedure

1. Create a draft GitHub Release on `main` using the exact version tag.
2. Upload only the matching prebuilt bundle:
   - `DET_FISH_v0.3_bundle.zip`
   - `MODEL_CROP_M1_v0.2_bundle.zip`
3. Publish the release. The `Verify and expand frozen model releases` workflow downloads the bundle, verifies its archive SHA-256, validates the internal `SHA256SUMS`, checks public metadata redaction, expands the individual model files into the release, and downloads them again for remote SHA-256 verification.
4. Do not upload the original source ZIP or raw detector metadata. The release bundles contain redacted public metadata.

## Retrieval

Use versioned release assets, not a floating branch URL. Download the bundle or the individual runtime asset and verify SHA-256 against `MODEL_LOCK.json` and `SHA256SUMS`.

## Freeze policy

- The release tag identifies a model version; do not reuse or replace an existing tag.
- The verification workflow does not overwrite existing release assets.
- It verifies exact model hashes and re-downloads every expanded asset before reporting success.
- Public metadata excludes source-image audit rows, review IDs, per-image source hashes, and internal GCS URIs.
- The existing `mobile-model-v0.2` Android classifier release remains unchanged.
- GitHub's platform-level immutable-release setting is not assumed; the freeze evidence is the versioned tag, no-overwrite workflow, and SHA-256 manifests.
