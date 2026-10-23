# SWOS Web Resource and Quality Track V1

**Task ID:** YUJIAN_SWOS_WEB_RESOURCE_QUALITY_LINE_V1

**Live pin checkpoint — 2026-10-23:** product PR #177 is Draft/Open/Unmerged at `70fd30b17839f5874d87b78a2f6e44eadd3cdeec`. No pull-request CI run is visible for this exact SHA yet. Its predecessor `ae42f3edd7deb7da60d76649507adff04fa182bc`, run [39978079329](https://github.com/pan277942135/Yujian/actions/runs/39978079329), failed `validate` because a deterministic DOM test double did not strip the `data-` prefix when mapping attributes to `dataset`; that run's Chromium job is still in progress and is predecessor-only. The latest test fix now models native DOM mapping and retains the overlay opt-in, reset, and stale-decode assertions. No deployment, live inference, or SAM result is claimed.

## Four workstreams

1. **Fish Mask:** use the product `/api/product/catch-recognition/infer` response so detector BBox, classifier result, and opt-in SAM candidate Mask remain bound to one upload SHA and one request. Collect returned Mask, transparent cutout, overlay, heuristic metrics, and detector/classifier/SAM/end-to-end latency. A candidate remains `REVIEW_PROXY`; heuristic GOOD is not production certification.
2. **15-frame baseline:** run the canonical design repository's `verify_swos_static_visual_v1_1.py` against the main-adopted 15-frame contract on each preview run. This proves contract/timing/geometry consistency only. Pixel-level visual calibration requires browser captures from the playable SWOS renderer; static proxy frames are not browser acceptance.
3. **Current-source preview:** `TARGET_SHA` is pinned to product PR #177 HEAD `70fd30b17839f5874d87b78a2f6e44eadd3cdeec`. The workflow requires the same-repository PR to remain open/unmerged on its expected branch, latest exact-SHA CI to succeed, and the canonical cross-repository source audit to pass. It enables the opt-in SAM candidate only on the `catch-pr177` Cloud Run tag and deploys with `--no-traffic`. No exact-source CI PASS exists for this pin, so this configuration is not deployment authorization. After all gates pass, verify default traffic unchanged and `/health/deploy` exact SHA and Revision before real-photo inference.
4. **Multi-scene and performance probe:** use four public, license-attributed Wikimedia Commons sources with pinned source digests. Derived dark/near-square/low-resolution variants are labeled as derivatives. Record model provenance, detection/classification, latencies, geometry, mask heuristics, and mask/cutout overlays as run artifacts.

## Public fixture policy

The probe downloads fixtures at runtime; source photos are not committed to the repository. Source photos and license attribution are recorded in `tests/fixtures/swos_resource_quality/photo_manifest_v1.json`. Private user uploads and `208758.jpg` are excluded.

## Status vocabulary

- `PASS`: exact SHA run genuinely completed and passed the named check.
- `BLOCKED_*`: a required precondition or output is absent, mismatched, or failed.
- `NOT_RUN`: a check has not executed; it is not a pass.

## Explicit boundaries

- This workflow does not change Cloud Run default traffic; production default traffic must be byte-for-byte unchanged.
- The 3650ms nominal visual timeline is not an inference countdown.
- Static frames, UI simulation, HTTP success, and heuristic GOOD masks do not prove visual acceptance.
- Browser optical frame-time benchmarking must run against the actual playable SWOS renderer, separately from SAM/API latency.
- Exact-source CI's public-fixture Chromium playback is fixture-only browser evidence, not deployed authenticated browser acceptance.
- The unrestricted test suite remains blocked by the metadata-access security review. It must not probe metadata endpoints or read runtime credentials; safely isolated targeted checks may still run.
- Current deployment Revision, health SHA, live model versions, live SAM checkpoint and real inference response: `NOT_VERIFIED`. Production Mask qualification: `NOT_QUALIFIED`.
