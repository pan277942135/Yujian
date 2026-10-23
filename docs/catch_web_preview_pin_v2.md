# PR177 B2 no-traffic preview source promotion

## Live pin checkpoint — 2026-10-23

- Product PR #177: Draft/Open/Unmerged, HEAD `73b5070abea65186b1a31adf616f417064c2cedd` (branch `feat/ai-catch-recognition-web-swos-rr09-v1`).
- Exact-source CI run [39979556132](https://github.com/pan277942135/Yujian/actions/runs/39979556132) is in progress; deployment gates remain closed.
- Previous exact-source run [39978306782](https://github.com/pan277942135/Yujian/actions/runs/39978306782) passed both jobs on `70fd30b`, but only reported WebGL2 browser capability, not active product renderer draws. This pin adds actual backend/draw-count diagnostics and the Chromium assertion; it needs a new exact-source run.
- Workflow `TARGET_SHA` is re-pinned to `73b5070abea65186b1a31adf616f417064c2cedd`. PR #182 is preparation only; exact-source CI, canonical cross-repository design audit, and PR #182 CI must pass before the controlled main-only deploy.
- No candidate deployment has run from this configuration. PR #182 remains Open/Unmerged; no current-source Revision or health SHA is verified, and default production traffic is unchanged.

## Deployment controls

The main-only workflow verifies that PR #177 is same-repository, open/unmerged, on the expected branch, and exactly equal to the pin. It also requires the latest exact-source `ci.yml` pull-request run to succeed. After CI and design-contract admission gates pass, the controlled main workflow uses tag `catch-pr177` with `--no-traffic`, compares default traffic before and after, enables `CATCH_SWOS_SAM_CANDIDATE_ENABLED=true` only on the candidate Revision, and checks `APP_GIT_COMMIT`, Revision, `/login`, and the product route. No manual deployment path is authorized.

## Model and visual evidence boundaries

The quality extension uses the real product same-request inference endpoint with four public, license-attributed, digest-pinned photos. It verifies source SHA/BBox/Mask consistency and records detector/classifier/SAM provenance, latency and review-only Mask/overlay/cutout artifacts.

No deployed authenticated browser session, live inference, candidate Mask, or real-photo 15-frame visual acceptance is verified. A `REVIEW_PROXY` fill is explicitly opted in and remains outside the trusted gold-mask path. Production Mask stays `NOT_QUALIFIED` pending reviewed ground truth, boundary metrics, difficult-case review and visual sign-off. The broad test suite remains `BLOCKED` pending metadata-access security review.
