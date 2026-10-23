# PR177 B2 no-traffic preview source promotion

Base main at the last live audit: 6a2064c2a02a76437b6e09a5241bc199cfa05048.

Current candidate source: a0596aaf609914e0de8a325da06b2f286a139150 (PR #177, same repository and expected branch). Exact-source CI run: https://github.com/pan277942135/Yujian/actions/runs/39963664724 (PASS: validate and Chromium). The fixture BBox focus report is BBOX_ONLY with no outline; ROI mean pixel delta 4.22 and max 17. This is fixture evidence only, not live inference, production Mask, or visual acceptance.

The preview workflow is pinned to this exact candidate SHA. Before any no-traffic deployment, it verifies that PR #177 is still open/unmerged on the expected branch, that its live HEAD equals the pin, and that the latest exact-source ci.yml pull-request run has succeeded. A changed PR HEAD makes preflight fail closed; update the pin only after confirming the new exact-SHA CI.

The proposed workflow update preserves the existing main-only Workload Identity Federation provider/service-account boundaries, catch-pr177 tag, --no-traffic, default-traffic snapshot comparison, and deployment SHA/Revision health checks. It enables the opt-in SAM candidate on the tagged no-traffic candidate revision only. The existing default traffic must remain byte-for-byte equivalent before and after deployment. No deployment has run from this update yet.

The resource-quality extension runs the canonical SWOS static 15-frame contract verifier and calls the product same-request inference endpoint with four public, license-attributed, digest-pinned photo sources plus clearly labeled derived low-light/geometry variants. It checks that the inference input SHA, segmentation SHA and Mask SHA refer to the same upload, then records detector/classifier/SAM/end-to-end latency and writes review-only Mask/overlay/cutout artifacts. No private user photo is uploaded or committed.

The quality probe deliberately reports Production Mask as NOT_QUALIFIED until reviewed ground-truth masks, boundary metrics, difficult-case analysis and explicit visual sign-off exist. The static contract verifier proves timing/geometry consistency only; it is not pixel-level browser acceptance. Browser optical frame-time benchmarking remains separate and must run against the actual playable SWOS renderer.

Exact-source CI covers safe targeted validation, UI simulation and the public-fixture 3650ms browser preview smoke. It does not verify the deployed authenticated browser, live production inference, real-photo accuracy or user visual acceptance. Full-suite metadata access review remains a separate unresolved gate. The preview must not change production default traffic, and product PR #177 remains Draft/unmerged. The deployment configuration update is still pending review/merge; the current deployed revision remains unverified for this source.
