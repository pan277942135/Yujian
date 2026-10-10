# 05 — QA and Publication Audit

**Staging execution:** NOT RUN. No actual reviewer, QA event, publication transaction or recovery event was recorded in Staging.

## Code/test evidence

The candidate CMS code supports distinct Visual and Content QA actions and rejects a request that submits both stages together. Each review creates its own immutable audit row, bound to the exact version/species/role and captures reviewer, timestamp, note, source/derived SHA, GCS generation and content revision/card where applicable. Content saves reset only the DRAFT Content QA summary to PENDING. Publication rechecks the role, binding, QA, content and stored GCS SHA/generation before its atomic old/new ACTIVE switch.

The test for post-commit serializer failure confirms the version stays ACTIVE, the response identifies `PUBLICATION_COMMITTED_READBACK_FAILED` / `publication_committed=true`, and the publication audit gets `FAILED_AFTER_COMMIT`. The persisted public API/image check requires an exact ACTIVE version and matching byte hashes. Separate client acceptance is gated on `PUBLIC_API_OK` for the same version.

The targeted local suite passed **31 tests** with four existing FastAPI lifecycle deprecation warnings. This proves only the tested local/fake-storage code paths. It does not prove a live PostgreSQL commit, GCS generation, retry/timeout/concurrency behavior or client display.

## Real review fields

| Evidence field | Staging value |
|---|---|
| `VISUAL_QA=PASS` | NOT RUN |
| `CONTENT_QA=PASS` | NOT RUN |
| reviewer / time / evidence | No Staging record |
| exact published `version_id` | None |
| publication audit | No Staging record |
| `PUBLIC_API_OK` / image byte match | NOT RUN in Staging |
| Android status | `CLIENT_PENDING` |
