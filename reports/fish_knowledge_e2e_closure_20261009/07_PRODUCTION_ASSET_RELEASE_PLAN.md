# 07 — Production Asset Release Plan (No Actions Executed)

This is a future controlled release sequence. It does not authorize production changes.

## Preconditions and approvals

1. Environment/platform owner provides an isolated Staging project, Cloud Run service, PostgreSQL database, GCS bucket/prefix and deploy service account. Prove there is no production DB, write IAM, bucket path or service-traffic coupling.
2. Database/GCS owner provides audited **read-only** production DB and object metadata access to complete all 120 slots. Missing rows/objects remain unknown until queried.
3. Content owner confirms each exact existing version's source/license, species, role, SHA, generation and intended publication. Manual review is required for unbound/ambiguous FishCards and any prior black/gold artwork.
4. CMS owner completes staging migration/deploy and a real one-version GCS-to-public-API closure. Android Validation Work validates that same published version. Product/data owner approves a production change window separately.

## Release order after approvals

1. Resolve public API compatibility in Staging first: list/detail must expose consistent `COVER_HERO` URL/version/status; each of the five `knowledge_assets[role]` records must match its `cards[]` row by species, role, URL and exact version ID. Never infer DB binding from URL equality alone.
2. For each species, review the existing COVER_HERO record/object first. Keep the current public image while creating any replacement as DRAFT. Run separate visual and content QA appropriate to the role; review source provenance and GCS hash/generation.
3. Publish one explicitly approved species/role version at a time. Record old/new IDs and card binding before each atomic switch. Verify public list/detail/API bytes immediately after each switch.
4. Process the five knowledge roles per species in a separately approved sequence. Never bulk-promote DRAFTs or substitute an unreviewed/black-gold image for a missing version.
5. Pause on any 404, hash/generation drift, duplicate ACTIVE, orphan, species/role mismatch, failed readback or client discrepancy. Preserve all prior objects and records.

## Recovery plan

Before an individually approved release, capture database backup/snapshot, current ACTIVE version/card IDs, content revision, publication audit and GCS object generation/hash. Recovery is a reviewed atomic repoint to the recorded prior version/card if that object remains valid; it is followed by API and image-byte checks. No history deletion, object overwrite or automated rollback is allowed. If publication committed but readback failed, inspect the database/audit first; do not assume rollback.

## Current recommendation

Do not publish production assets yet. Current public API exposes 100 ACTIVE card rows but omits all card `asset_version_id` values; 85 of 100 distinct card-slot URLs returned HTTP 404 in the read-only audit. This does not establish whether matching GCS objects exist. The single exposed COVER_HERO URL is readable, while the other 19 hero URLs/versions are absent from the API response. Resolve with DB/GCS evidence and manual review before preparing any production change.
