# SWOS Web Resource and Quality Track V1

**Task ID:** YUJIAN_SWOS_WEB_RESOURCE_QUALITY_LINE_V1

**Live pin checkpoint — 2026-10-23:** product PR #177 is Draft/Open/Unmerged at `73b5070abea65186b1a31adf616f417064c2cedd`. Exact-source CI run [39979556132](https://github.com/pan277942135/Yujian/actions/runs/39979556132) is in progress; no current-source PASS is claimed. Its previous exact-source run [39978306782](https://github.com/pan277942135/Yujian/actions/runs/39978306782) passed validate and Chromium on `70fd30b`, including 9 viewport UI simulation, fixture 15-frame playback, and BBox focus pixel regression. It did not report actual product water-renderer execution. This source adds runtime WebGL2/Canvas2D path and draw-count evidence, bound to the PR head SHA. No deployment, live inference, or SAM result is claimed.

## Four workstreams

1. **Fish Mask:** use product `/api/product/catch-recognition/infer` response so detector BBox, classifier result, and opt-in SAM candidate Mask remain bound to one upload SHA and one request. Collect actual returned Mask, transparent cutout, overlay, heuristic metrics, and detector/classifier/SAM/end-to-end latency. Candidate Mask remains `REVIEW_PROXY`; heuristic GOOD is not production certification.
2. **15-frame baseline:** run the canonical design repository's `verify_swos_static_visual_v1_1.py` against the main-adopted 15-frame contract on every preview run. Static timing/geometry and fixture frame playback are not pixel-level real-photo visual acceptance.
3. **Current-source preview:** `TARGET_SHA` is pinned to PR #177 HEAD `73b5070abea65186b1a31adf616f417064c2cedd`. The workflow requires the same-repository PR to remain open/unmerged on its expected branch, latest exact-SHA CI to succeed, and canonical cross-repository source audit to pass. It uses only the `catch-pr177` Cloud Run tag, `--no-traffic`, and enables opt-in SAM only on the candidate Revision. Current exact-source CI is still running, so this pin is not deployment authorization.
4. **Multi-scene and performance probe:** use four public, license-attributed Wikimedia Commons photos with pinned source digests. Derived dark/near-square/low-resolution variants are labeled as derivatives. Record live model provenance, detection/classification, timings, geometry, Mask heuristics, and Mask/cutout overlays as run artifacts.

## Public fixture policy

The probe downloads fixtures at runtime; source photos are not committed to the repository. Source photos and license attribution are recorded in `tests/fixtures/swos_resource_quality/photo_manifest_v1.json`. Private user uploads and `208758.jpg` are excluded from public CI and public artifacts.

## Status vocabulary

- `PASS`: exact SHA run genuinely completed and passed the named check.
- `BLOCKED_*`: a required precondition or output is absent, mismatched, or failed.
- `NOT_RUN`: a check has not executed; it is not a pass.

## Explicit boundaries

- This workflow does not change Cloud Run default traffic; production default traffic must be byte-for-byte unchanged.
- The nominal visual timeline is 2800ms processing + 850ms reveal = 3650ms. It is not an inference countdown.
- Static frames, UI simulation, HTTP success, and heuristic GOOD masks do not prove visual acceptance.
- The product Chromium runtime probe records the selected WebGL2/Canvas2D backend and actual water draw count; a fallback records its reason. It remains fixture/UI-simulation evidence.
- Browser optical frame-time benchmarking must run against the actual deployed product renderer, separately from SAM/API latency.
- The unrestricted test suite remains blocked by the metadata-access security review. It must not probe metadata endpoints or read runtime credentials; safely isolated targeted checks may still run.
- Current deployment Revision, health SHA, live model versions, live SAM checkpoint and real inference response: `NOT_VERIFIED`. Production Mask qualification: `NOT_QUALIFIED`.
