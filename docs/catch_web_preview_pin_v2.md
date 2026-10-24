# PR177 B2 no-traffic preview source promotion

## Current execution contract

The candidate workflow is manual-only. It does not deploy automatically when the workflow file changes. From the Actions page, select AI Catch Recognition PR177 Tagged Preview, choose the main branch, and supply the exact current PR #177 HEAD in the required target_sha input.

The preflight verifies all of the following before cloud authentication:
- PR #177 is same-repository, open/unmerged, on feat/ai-catch-recognition-web-swos-rr09-v1, and its live HEAD exactly equals target_sha.
- The latest ci.yml pull-request run for that exact SHA completed successfully.
- The deployment uses the exact approved source SHA and the fixed catch-pr177 tag with --no-traffic.
- Default Cloud Run traffic is captured before deployment and verified unchanged afterward.
- Candidate health APP_GIT_COMMIT, Revision, /login, and product route must match the deployed source.

## Authenticated public-fixture probe

The post-deploy probe uses only the four license-attributed, digest-pinned public fixtures in tests/fixtures/swos_resource_quality/photo_manifest_v1.json. Private user photos are never used in CI.

The probe requires the repository Actions secret CONSOLE_ACCESS_KEY, passed only as SWOS_PROBE_CONSOLE_ACCESS_KEY to the probe process. The value is never printed or written to artifacts. The probe derives the same console cookie hash used by app/secure.py; if the secret is absent, it reports AUTH_NOT_CONFIGURED and makes no protected inference requests. If the service rejects it, the run reports AUTH_REJECTED; neither case is counted as a zero-mask model result.

Fixture downloads use bounded retries for HTTP 429 and transient 5xx/URL failures, respect Retry-After where available, verify the pinned source digest after download, and record fixture-rate-limit errors separately from inference failures.

## Evidence boundaries

A successful deployment or health check is not visual acceptance. The probe records the exact source SHA, candidate Revision, detector/classifier/SAM provenance, same-request source/BBox/Mask binding, per-stage latency, and review-only Mask/overlay/cutout artifacts.

REVIEW_PROXY remains candidate evidence, not production qualification. Production Mask stays NOT_QUALIFIED until reviewed ground-truth masks, boundary metrics, hard-case review and explicit visual sign-off exist. Browser optical frame-time and the 15-frame real-photo reveal are separate acceptance gates.

No product PR is merged by this workflow. No default production traffic is changed.
