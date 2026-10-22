# PR177 B2 no-traffic preview source promotion

Base main: 1d01231477645f5b888b8815c5602576dac90c61.
B2 candidate source: cb9e3780afb5e2904e3dfe5148210a4ccc3747b4 (PR177, same repository and expected branch; its workflow must pass the latest exact-source CI before cloud authentication).

This infrastructure-only change updates the existing fixed source SHA, verifies the expected branch, and requires the latest exact-source ci.yml PR run to succeed before the existing main-only cloud authentication. It preserves the service, WIF/provider/service-account boundaries, catch-pr177 tag, --no-traffic, default-traffic comparison and deployment SHA/Revision health checks.

Source CI is limited to the safe targeted checks and Chromium UI simulation. It does not verify live model inference, real-photo accuracy, user visual acceptance, or the blocked full-suite metadata path. Full regression remains BLOCKED_AUTO_REVIEW_METADATA_ACCESS.

Infrastructure CI is intentionally skipped, not PASS, to avoid rerunning the unresolved full-suite metadata path in another environment. No cloud deployment command was executed by this update.

The task sections 2.3 and 11.2 require an independent infrastructure PR and approval before main merge. Do not merge automatically. Once independently approved, the existing main-path push can deploy only this pinned candidate, only after its latest necessary source CI succeeds, and only with --no-traffic. Product PR177 remains Draft/unmerged; production default traffic must remain unchanged.


## P0 provenance refresh V1

Current candidate pin: `32b9463887c6620b32a3dfdd986eccc6ff4a569b` (PR177 remains unmerged). This source adds exact provenance to /health. Both health endpoints must match the candidate SHA and K_REVISION at the actual tagged URL. Infrastructure-only main CI now has a strict, negatively tested path allowlist and runs pure targeted validation; FULL_SUITE remains explicitly BLOCKED and no product PASS is inferred. Source PR177 exact-SHA CI, main-only WIF, same-repo/branch/HEAD verification, zero traffic and traffic-equality checks remain mandatory.
