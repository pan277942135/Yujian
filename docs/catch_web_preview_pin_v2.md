# PR177 B2 no-traffic preview source promotion

## Live pin checkpoint — 2026-10-23

- Product PR #177: Draft/Open/Unmerged, HEAD `ae42f3edd7deb7da60d76649507adff04fa182bc` (branch `feat/ai-catch-recognition-web-swos-rr09-v1`).
- Latest exact-source CI run: [39978079329](https://github.com/pan277942135/Yujian/actions/runs/39978079329), currently queued; no PASS is claimed.
- Predecessor run [39977837307](https://github.com/pan277942135/Yujian/actions/runs/39977837307) tested `75313f93`; `validate` failed only because its DOM test double did not mirror `dataset` attribute removal. The Chromium job is still running; both results are predecessor-only.
- The workflow `TARGET_SHA` is updated to `ae42f3edd7deb7da60d76649507adff04fa182bc`. The current source must pass its own exact-SHA CI and the canonical cross-repository source audit before PR #182 can merge or deployment can run.
- PR #182 remains Open/Unmerged. No candidate deployment has run from this configuration, no current source Revision is verified, and default production traffic is unchanged.

## Deployment controls

The main-only workflow verifies the same-repository PR #177 is open/unmerged on its expected branch and that its live HEAD equals the pin. It also requires the latest exact-source `ci.yml` pull-request run to succeed. After the CI and design-contract admission gates pass, the controlled main workflow uses the `catch-pr177` tag and `--no-traffic`, snapshots default traffic before and after, enables `CATCH_SWOS_SAM_CANDIDATE_ENABLED=true` only on the candidate Revision, and checks `APP_GIT_COMMIT`, Revision, `/login`, and the product route. No manual deployment path is authorized.

## Model and visual evidence boundaries

The resource-quality extension runs the canonical 15-frame contract verifier and calls the real product same-request inference endpoint with four public, license-attributed, digest-pinned photo sources plus clearly labeled derived variants. It checks the inference input SHA, segmentation SHA, BBox and Mask against the same upload, then records detector/classifier/SAM provenance, latency and review-only Mask/overlay/cutout artifacts.

No deployed authenticated browser session, live inference, candidate Mask, or 15-frame real-photo visual acceptance has been verified. The SWOS renderer's new `REVIEW_PROXY` fill is an explicitly opted-in review overlay and remains outside the trusted gold-mask path.

Production Mask remains `NOT_QUALIFIED` pending reviewed ground-truth masks, boundary metrics, difficult-case review and explicit visual sign-off. Browser optical frame-time benchmarking remains separate from SAM/API latency. The broad test suite remains `BLOCKED` pending the existing metadata-access review. No default traffic has moved.
