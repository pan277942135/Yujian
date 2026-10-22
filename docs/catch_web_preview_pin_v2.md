# PR177 B1 preview source promotion

Base main: 1d01231477645f5b888b8815c5602576dac90c61.
Approved candidate source: 655c4af07a0244fcf29441b5652c48ca74f2e6a2 (PR177, same repository, expected branch).

This infrastructure-only change updates the existing fixed SHA, verifies the expected branch, and requires the latest exact-source ci.yml PR run to succeed before the existing main-only cloud authentication. It preserves the service, WIF/provider/service-account boundaries, catch-pr177 tag, --no-traffic, default traffic equality check and health SHA/Revision verification.

Source CI means B1 targeted checks and UI simulation only. Full regression remains BLOCKED_AUTO_REVIEW_METADATA_ACCESS. This is not production release approval, high-fidelity signoff, live-model accuracy evidence or permission to merge product PR177.

Local YAML parsing, every shell block's bash -n and no-traffic/source invariants passed. No cloud command was executed. This branch intentionally uses [skip ci] because main's unmodified general CI runs the full suite whose metadata review remains unresolved; it must not be rerun remotely to bypass that review. Infrastructure checks here are static only and are not reported as CI PASS.

The task sections 2.3 and 11.2 require an independent infrastructure PR and approval before main merge. Do not merge automatically. Once approved, the existing main-path push can deploy only the fixed candidate and only if its necessary CI has succeeded. For each future source change, review a new explicit pinned SHA; never accept arbitrary workflow-dispatch PR input or widen WIF trust.
