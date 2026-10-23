# PR177 B2 no-traffic preview source promotion

## Live pin checkpoint — 2026-10-23

- Product PR #177: Draft/Open/Unmerged, HEAD `70fd30b17839f5874d87b78a2f6e44eadd3cdeec` (branch `feat/ai-catch-recognition-web-swos-rr09-v1`).
- No pull-request CI run is visible yet for this exact SHA. The previous run [39978079329](https://github.com/pan277942135/Yujian/actions/runs/39978079329) was on `ae42f3edd7deb7da60d76649507adff04fa182bc`; its `validate` failed only because the deterministic DOM test double did not strip `data-` before mapping to `dataset`. Its Chromium job is still in progress and is not current-source evidence.
- The latest test fixture now follows native DOM name mapping. The exact-source validation still must pass before any merge/deploy.
- Workflow `TARGET_SHA` is re-pinned to `70fd30b17839f5874d87b78a2f6e44eadd3cdeec`. This PR remains preparation only; it requires exact-source CI and canonical cross-repository admission before the controlled main-only deployment.
- PR #182 remains Open/Unmerged. No candidate deployment has run from this configuration; no current-source Revision is verified and default production traffic is unchanged.

## Deployment controls

The main-only workflow verifies the same-repository PR #177 is open/unmerged on its expected branch and that its live HEAD equals the pin. It also requires the latest exact-source `ci.yml` pull-request run to succeed. After the CI and design-contract admission gates pass, the controlled main workflow uses the `catch-pr177` tag and `--no-traffic`, snapshots default traffic before and after, enables `CATCH_SWOS_SAM_CANDIDATE_ENABLED=true` only on the candidate Revision, and checks `APP_GIT_COMMIT`, Revision, `/login`, and the product route. No manual deployment path is authorized.

## Model and visual evidence boundaries

The resource-quality extension runs the canonical 15-frame contract verifier and calls the real product same-request inference endpoint with four public, license-attributed, digest-pinned photo sources plus clearly labeled derived variants. It checks the inference input SHA, segmentation SHA, BBox and Mask against the same upload, then records detector/classifier/SAM provenance, latency and review-only Mask/overlay/cutout artifacts.

No deployed authenticated browser session, live inference, candidate Mask, or 15-frame real-photo visual acceptance has been verified. The SWOS renderer's `REVIEW_PROXY` fill is explicitly opted in and remains outside the trusted gold-mask path.

Production Mask remains `NOT_QUALIFIED` pending reviewed ground-truth masks, boundary metrics, difficult-case review and explicit visual sign-off. Browser optical frame-time benchmarking remains separate from SAM/API latency. The broad test suite remains `BLOCKED` pending the existing metadata-access review. No default traffic has moved.
