# Fish Knowledge CMS V1.4 Operator UI & Publication Closure — Handoff

**Audit date:** 2026-10-09 (UTC)  
**Repository:** `pan277942135/Yujian`  
**Branch:** `fix/fish-knowledge-cms-v1-4-publication-closure`  
**BEFORE source HEAD:** `3a9126e18eeb87c3228dea0760539a47cacfea20`  
**Draft PR:** [#159](https://github.com/pan277942135/Yujian/pull/159)  
**Android reference:** `pan277942135/Yujian_App` @ `505690f0a681334b6c3411de50f55e026076b464` (not changed)

## BEFORE audit snapshot

- v1.3 upload, batch import, warning acknowledgement, DRAFT versioning, QA/freeze records already existed. Existing import execution was reused.
- v1.4 had an independent workspace, publication service, and migration 0031. The 0031 migration performs only exact one-to-one species/role/URL binding; ambiguous old FishCards remain unbound. It preserves historical rows.
- The v1.4 UI exposed batch import and extensions in primary navigation, used JSON as the ordinary card editor, and combined visual/content QA into one PASS action. Upload did not give the operator a preflight-preview-confirm sequence before DRAFT creation.
- The publication service switched the old ACTIVE version/card and the new pair in one database transaction, but exceptions during post-commit public serializer readback were not distinguished from transaction failures.
- Read-only production API evidence was obtained from `https://yujian-model-factory-console-571785698442.asia-east1.run.app`:
  - `GET /api/v1/fish/species`: HTTP 200, 20 species; the deployed response omitted all three `cover_hero_*` list fields.
  - `GET /api/v1/fish/species/redfin_culter/detail`: HTTP 200 for all 20 species.
  - Current public API returned 100/100 `cards[]` rows with `status=ACTIVE`, but 0/100 exposed `asset_version_id` values. Grass carp returned three `knowledge_assets` IDs (136–138), without status/publication source and without those IDs on matching `cards[]` rows. Other knowledge asset role entries were absent in the observed response.
  - One detail response contained a COVER_HERO URL, but no version ID/status; list/detail consistency could not be established.
- Production PostgreSQL and GCS credentials/connectors were unavailable. Their records, object SHA/generation, DRAFT/QA/ACTIVE counts, and exact FishCard bindings are **unknown**, not zero. The 120-row matrix reports this explicitly.

## A. Page audit and changes

Primary navigation is now: 鱼种概览、基本信息、图片资产、五张知识卡、发布审核、历史与审计. Batch import remains an independent operation link inside 图片资产; 扩展资料 remains reachable from the overview/assets pages.

Hidden from ordinary pages: the legacy black-gold card editor; direct legacy Cover/Card create, overwrite, and delete controls; direct ACTIVE-image changes; low-level database identifier inputs; default-expanded source SHA/GCS path/generation/history payloads; duplicate legacy import shortcuts and invalid direct API entry points. Legacy active FishCards remain read-only previews. Version history, provenance, immutable QA audit, publication audit, and 0031 compatibility columns are retained. Legacy mutation APIs still close with HTTP 409; front-end hiding is not the authorization control.

## B. Production asset audit — 120 slots

The attached CSV has one row for each of 20 public species × (`COVER_HERO`, `HERO`, `IDENTIFICATION`, `ECO`, `GEAR`, `SKILL`). `UPLOADED`, `DRAFT`, `QA_PASS`, and target-version `ACTIVE` are marked `UNKNOWN_PRODUCTION_DB_UNAVAILABLE`; no database/GCS absence is inferred. `PUBLIC_API_OK` is blocked wherever the deployed response cannot prove the target version. `CLIENT_PENDING=YES` for all 120; the separate Android validation work has not been run here.

Observed API diagnostic counts: 20 species; 100/100 legacy/current card objects report `ACTIVE`; 0/100 `cards[]` objects expose a version ID; 3 roles expose a `knowledge_assets.version_id`; 1/20 details return a COVER_HERO image URL. Those values do not establish the target v1.3 asset state.

[Download the 120-slot asset matrix](sandbox:/workspace/scratch/a62cc5093716/yujian-cms-v14-work/reports/fish_knowledge_cms_v14_asset_matrix_2026-10-09.csv)

## C. Upload, edit, review, publish, and API readback

- Single-image upload now stages through the existing v1.3 batch core with `preflight_only`; valid/warning files show source preview before explicit execute. WARNING needs confirmation. INVALID and duplicate-blocked items do not execute. DRAFT creation is reported only after the workspace re-reads the exact persisted version ID/status and preview URL.
- The old active image and the editing DRAFT are previewed side by side. Upload does not activate or overwrite either one.
- Five-card editing now uses role-specific structured forms. Raw JSON is behind an advanced disclosure. Save binds `species_id + asset_role + version_id`, increments content revision, returns readback fields, and resets only DRAFT content QA to PENDING. Save failures do not redraw the form.
- Visual QA and content QA are separate actions in both the v1.4 publication center and batch import review; the backend rejects a combined-stage submission. Each action writes an immutable audit row with result, reviewer, time, note, source/derived SHA, GCS generation, exact version and content revision. Publication presents a prepublish checklist and requires confirmation of the explicit version ID. Backend revalidates GCS generation/SHA, role, binding, QA and content; old and new ACTIVE rows switch atomically.
- After database commit, a public-detail readback failure returns `PUBLICATION_COMMITTED_READBACK_FAILED` with `publication_committed=true`; it does not claim rollback. The publication page separately checks real public list/detail values and compares the public image bytes with the selected version preview, then saves reviewer/time/version/hash evidence to the publication audit. Client acceptance has a separate exact-version endpoint and is gated on the same version having `PUBLIC_API_OK`; it records `CLIENT_PASSED` or `CLIENT_FAILED` without conflating API and client state.
- Production upload, content write, QA, publication, GCS read, migration, and activation were not performed. Thus there is no real GCS upload-to-publication evidence from staging in this handoff.

## D. Data consistency

Code enforces PostgreSQL `version_id` → FishCard binding and same species/role/image URL, with revision history. Migration 0031 remains additive. New 0032 adds separate QA-stage evidence and an immutable audit table; startup migration is idempotent. Production row counts, hashes, generation values, GCS readability, and target version/API alignment remain unavailable because the database and GCS evidence were not accessible.

## E. Basic tests

**PASS**

- Python `compileall` for `app` and `tests`.
- CMS upload/version binding, WARNING confirmation, INVALID handling, separate QA audit, publish atomicity/post-commit error semantics, 0031 migration, public API contract, and legacy 409/UI tests (31 passed in the final focused run; 4 existing FastAPI lifecycle deprecation warnings).
- Node syntax checks for all v1.4 ES modules and embedded v1.3 batch-import script.
- `git diff --check`.

**NOT RUN**

- Authorized non-production 0031/0032 migration against a staging database.
- Real GCS upload, preview, publication, public HTTP image check, and failure recovery on staging.
- Android device/client acceptance.

## F. Git and PR

- Starting remote HEAD: `3a9126e18eeb87c3228dea0760539a47cacfea20`.
- Target branch and Draft PR #159 are retained; `main` is not merged.
- Final implementation commit and branch HEAD are recorded in the PR after push.
- No production ACTIVE asset was changed.

## G. Unresolved items and next steps

| Owner | Blocker | Next action |
|---|---|---|
| Environment owner | No authorized staging PostgreSQL/GCS environment or credentials were present | Provide a non-production DB snapshot, service identity, and test bucket; verify migrations 0031 and 0032 before deployment |
| CMS operator + environment owner | No real GCS asset was staged/published in this work | Execute one species/role end-to-end in staging; verify DB binding, SHA/generation, image HTTP bytes, API values, retries and recovery |
| Android Validation Work owner | Client display not tested in this work | Validate the published target version on an authorized Android device and record the result independently |
| Data owner | Production DB/GCS state for all 120 target slots unavailable | Run the read-only SQL/GCS audit in an authorized environment and refresh the matrix without treating missing evidence as missing objects |

## Final status

`READY_FOR_STAGING_VALIDATION` — code and local/fake-storage tests passed; staging migration, real GCS/API closure, and Android acceptance remain outstanding. This is not a claim that production is safe to publish.
