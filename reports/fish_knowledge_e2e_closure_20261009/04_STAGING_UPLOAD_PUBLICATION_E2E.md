# 04 — Staging Upload → Publication E2E

**Status:** `BLOCKED_REAL_GCS_EVIDENCE` (upstream blocker: `BLOCKED_STAGING_ENVIRONMENT`). No real Staging upload or publication was executed.

| Step | Result |
|---|---|
| Pick permitted real test photo; verify its source/license | NOT RUN — no isolated Staging; production grass carp media license/source proof is also not in API response |
| Select grass carp / COVER_HERO in deployed CMS | NOT RUN |
| Preflight, preview and warning confirmation | NOT RUN in deployed CMS |
| INVALID / duplicate / interrupted upload / retry / GCS failure | NOT RUN against real GCS |
| Create and read back a DRAFT | NOT RUN |
| Verify PostgreSQL version and exact `species_id + asset_role` | NOT RUN |
| Verify GCS object, source/derived SHA and generation | NOT RUN |
| Edit/save, review, publish and public API readback | NOT RUN |
| Verify old ACTIVE and FishCard atomic switch/recovery | NOT RUN |

The local CMS test suite uses in-memory/fake storage for its upload and publication cases; it is useful code-contract coverage, **not** a real-GCS E2E result. No mock or unit test is presented as `STAGING_PUBLICATION_E2E_PASS`.

The real production public API exposed a grass carp COVER_HERO URL that returned HTTP 200, but the API exposes no source license, target version ID or database/GCS binding. That asset was not copied or reused in Staging.
