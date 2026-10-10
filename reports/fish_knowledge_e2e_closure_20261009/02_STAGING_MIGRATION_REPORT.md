# 02 — Staging Migration Report

**Status:** `BLOCKED_STAGING_ENVIRONMENT` — migration stopped before any database connection or write.

## Targets reviewed statically

- `schemas/0031_fish_knowledge_publication_v14.sql`
- `schemas/0032_fish_knowledge_separate_qa_v14.sql`

The 0031 script adds `fish_cards.asset_version_id` with `ON DELETE RESTRICT` and `content_revision`; migrates only unique species + canonical role + image URL pairs; leaves ambiguous pairs unbound; adds a version lookup index and conditionally installs unique indexes only if existing rows permit; creates content revision/publication audit tables with restrictive foreign keys; and seeds revision snapshots for already-bound cards. Static review does not prove the target database's constraints, counts or migration state.

The 0032 script adds independent visual/content QA notes, reviewers and timestamps, plus `fish_knowledge_asset_qa_audits` with stage check, exact asset/card foreign keys and version-stage indexes. The down script intentionally retains QA evidence. Local targeted tests and the exact-CI validation passed; neither is evidence of migration against Staging PostgreSQL.

## Execution evidence

| Required operation | Result |
|---|---|
| Identify Staging PostgreSQL/version/database | NOT RUN — no isolated DB endpoint/identity available |
| Recoverable backup/snapshot | NOT RUN |
| Pre-migration schema/table/index/constraint and row-count snapshot | NOT RUN |
| Determine whether 0031/0032 already ran | NOT RUN |
| Execute 0031 then 0032 | NOT RUN |
| Verify idempotence, row preservation, ACTIVE/DRAFT counts and FishCard binding | NOT RUN |
| Restore rehearsal | NOT RUN |

No migration SQL, ORM startup DDL, backup, restore or down migration was executed against any database. Production remains untouched.

## Gate

`SCHEMA_0031_PASS` and `SCHEMA_0032_PASS` are **NOT ESTABLISHED**. Do not proceed until a separate Staging project/database and recoverable snapshot are verified.
