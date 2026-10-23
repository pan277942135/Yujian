# PR177 B2 no-traffic preview source promotion

Base main: 3c79c00db68ace94d8f6a7dcb4105d0026a09323.

Current candidate source: 4de911dc497c150545d938b60b2694f3c4723e37 (PR #177, same repository and expected branch). Exact-source CI run: https://github.com/pan277942135/Yujian/actions/runs/39922218177.

The preview workflow is pinned to this exact candidate SHA. Before any no-traffic deployment, it verifies that PR #177 is still open/unmerged on the expected branch, that its live HEAD equals the pin, and that the latest exact-source ci.yml pull-request run has succeeded. A changed PR HEAD makes preflight fail closed; update the pin only after confirming the new exact-SHA CI.

The workflow preserves the existing main-only Workload Identity Federation provider/service-account boundaries, catch-pr177 tag, --no-traffic, default-traffic snapshot comparison, and deployment SHA/Revision health checks. The source is deployed only to a tagged no-traffic candidate revision; the existing default traffic must remain byte-for-byte equivalent before and after deployment.

The resource-quality extension also runs the canonical SWOS static 15-frame contract verifier and probes the candidate SAM endpoint with four public, license-attributed, digest-pinned photo sources plus clearly labeled derived low-light/geometry variants. It records detector/SAM/end-to-end latency and uploads mask-overlay/cutout evidence. No private user photo is uploaded or committed.

The quality probe deliberately reports Production Mask as NOT_QUALIFIED until reviewed ground-truth masks, boundary metrics, difficult-case analysis and explicit visual sign-off exist. The static contract verifier proves timing/geometry consistency only; it is not pixel-level browser acceptance. Browser optical frame-time benchmarking remains separate and must run against the actual playable SWOS renderer.

Source CI currently covers safe targeted validation and Chromium UI simulation. It does not verify live production inference, real-photo accuracy or user visual acceptance. Full-suite metadata access review remains a separate unresolved gate. The preview must not change production default traffic, and product PR #177 remains Draft/unmerged.
