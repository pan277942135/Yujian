# SWOS Web Resource and Quality Track V1

**Task ID:** YUJIAN_SWOS_WEB_RESOURCE_QUALITY_LINE_V1

**State:** ACTIVE; product PR #177 source is 7083ae691d424b4f6d5680fb77f019c9b437505b. Exact-source CI is running. The CI harness now records a complete fixture-only 3650ms browser playback and checks repeated-frame determinism. Same-request production inference, SAM output, and candidate deployment have not run because the no-traffic deploy configuration remains open and unmerged.

## Four workstreams

1. **Fish Mask:** use the product /api/product/catch-recognition/infer response so detector bbox, classifier result and the SAM candidate Mask are bound to one upload SHA and one inference request. Collect the returned Mask, transparent cutout, overlay, heuristic metrics and per-case detector/classifier/SAM/end-to-end latency. The candidate remains REVIEW_PROXY; heuristic GOOD is not production certification. Production qualification remains blocked until reviewed ground-truth masks, boundary metrics, difficult-case review and explicit visual sign-off exist.
2. **15-frame baseline:** run the canonical design repository's verify_swos_static_visual_v1_1.py against the main-adopted 15-frame contract on every preview run. This proves contract/timing/geometry consistency only. Pixel-level visual calibration is pending real browser frame captures from the playable SWOS renderer; static proxy frames must not be frozen or reported as browser acceptance.
3. **Current-source preview:** the pending configuration update pins PR #177 at 7083ae691d424b4f6d5680fb77f019c9b437505b, requires the same-repository PR to remain open/unmerged on the expected branch, and requires exact-SHA CI success. It enables the opt-in SAM candidate only on the existing catch-pr177 Cloud Run tag and deploys with --no-traffic. This candidate has not been deployed; after the configuration is authorized, verify default traffic is unchanged and /health/deploy reports the exact source SHA and Revision before any real-photo inference.
4. **Multi-scene and performance probe:** use four public, license-attributed Wikimedia Commons sources with pinned source digests. Derived dark/near-square/low-resolution variants are labeled as derivatives, not independent scenes. Record detector latency, SAM latency, end-to-end request latency, geometry, mask heuristics, and mask/cutout overlays as run artifacts.

## Public fixture policy

The probe downloads fixtures at runtime; source photos are not committed to the repository. The four original sources and license attribution are recorded in tests/fixtures/swos_resource_quality/photo_manifest_v1.json. Private user-uploaded photos and the 208758.jpg handoff asset are prohibited from this public workflow.

## Status vocabulary

- CONTRACT_PASS: formal timing/geometry contract validates.
- CANDIDATE_OUTPUTS_COLLECTED_NOT_PRODUCTION_QUALIFIED: actual SAM outputs and metrics were collected; visual and ground-truth qualification remains outstanding.
- BLOCKED_*: runtime SHA/revision mismatch, unavailable segmentation, insufficient candidate masks, or one or more failed cases.

## Explicit boundaries

- This workflow does not change Cloud Run default traffic.
- The 3650ms nominal visual timeline is not an inference countdown.
- Static frames, UI simulation, HTTP success, and heuristic GOOD masks do not prove high-fidelity visual acceptance.
- Browser optical frame-time benchmarking must be performed against the actual playable renderer and reported separately from SAM/API latency.
- Exact-source CI includes a public-fixture 3650ms browser preview smoke. It is fixture-only browser evidence, not the deployed authenticated browser run and not visual acceptance.
- The unrestricted test suite remains BLOCKED by the existing metadata-access review. PR #182 CI therefore runs SWOS offline unit tests, compilation, shell checks and route/import checks; the broad suite is not used as acceptance evidence.
- Current deployed source SHA, Revision, live SAM checkpoint, and real inference response: NOT_VERIFIED. Production Mask qualification: NOT_QUALIFIED.

