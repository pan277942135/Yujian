# SWOS Web Resource and Quality Track V1

**Task ID:** YUJIAN_SWOS_WEB_RESOURCE_QUALITY_LINE_V1

**State:** ACTIVE; evidence is collected on a no-traffic candidate revision only.

## Four workstreams

1. **Fish Mask:** reuse the existing SAM ViT-B segmentation endpoint. Collect actual detector bbox, SAM mask, transparent cutout, overlay, mask-quality heuristic and per-case latency. The endpoint is experimental; GOOD is not production certification. Production qualification remains blocked until reviewed ground-truth masks, boundary metrics, difficult-case review and explicit visual sign-off exist.
2. **15-frame baseline:** run the canonical design repository's verify_swos_static_visual_v1_1.py against the main-adopted 15-frame contract on every preview run. This proves contract/timing/geometry consistency only. Pixel-level visual calibration is pending real browser frame captures from the playable SWOS renderer; static proxy frames must not be frozen or reported as browser acceptance.
3. **Current-source preview:** resolve PR #177's live HEAD at run time, require the same-repository PR to remain open/unmerged on the expected branch, and require exact-SHA CI success. Deploy only that SHA to the existing catch-pr177 Cloud Run tag with --no-traffic. Verify default traffic is unchanged and /health/deploy reports the exact source SHA and Revision.
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

