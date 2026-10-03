"""Phase A audit for historical exact-image duplicates.

This operator entry point deliberately has no cleanup/apply mode.  ``--bootstrap``
delegates the only allowed production write to ``bootstrap_global_registry``;
the remaining work is read-only reporting over the existing business tables,
Accepted Pool source references, and immutable DatasetItem lineage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from google.cloud import storage
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.accepted_pool import ACCEPTED_STATUSES, _pool_manifest_rows
from app.db import SessionLocal
from app.dataset_models import DatasetItem
from app.exact_dedupe import bootstrap_global_registry
from app.models import (
    BatchCropReview,
    DatasetVersion,
    GlobalImageContent,
    GlobalImageDuplicateMember,
    ImageAsset,
)


AUTO_REVIEWERS = {"鱼体检测", "近重复检测"}
CANONICAL_TRUTH_STATUSES = {"LIKELY_CORRECT", "CANONICAL", "CONFIRMED"}
REVIEW_STATUSES = ("pending", "needs_review", "hard_case", "approved", "rejected")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _git_sha() -> str | None:
    configured = os.getenv("AUDIT_GIT_SHA", "").strip() or os.getenv("APP_GIT_COMMIT", "").strip()
    if configured:
        return configured
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        return None


def _count(db: Session, model: Any, *criteria: Any) -> int:
    statement = select(func.count()).select_from(model)
    if criteria:
        statement = statement.where(*criteria)
    return int(db.scalar(statement) or 0)


def business_counts(db: Session, accepted_pool_source_count: int | None = None) -> dict[str, int]:
    """Capture only counts that Phase A is forbidden to change."""

    result = {
        "image_assets": _count(db, ImageAsset),
        "dataset_versions": _count(db, DatasetVersion),
        "dataset_items": _count(db, DatasetItem),
        "approved_image_assets": _count(db, ImageAsset, ImageAsset.review_status == "approved"),
        "pending_image_assets": _count(db, ImageAsset, ImageAsset.review_status == "pending"),
        "rejected_image_assets": _count(db, ImageAsset, ImageAsset.review_status == "rejected"),
    }
    if accepted_pool_source_count is not None:
        result["accepted_pool_source_count"] = int(accepted_pool_source_count)
    return result


def _parse_bbox(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
        if not isinstance(parsed, (list, tuple)) or len(parsed) != 4:
            return None
        result = [float(item) for item in parsed]
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not all(0.0 <= item <= 1.0 for item in result):
        return None
    return result


def _truth(value: Any) -> str:
    return str(value or "").strip()


def _truth_key(value: Any) -> str:
    return _truth(value).casefold()


def _canonical_truth(image: ImageAsset) -> bool:
    return bool(_truth(image.truth_species)) and str(image.truth_status or "").strip().upper() in CANONICAL_TRUTH_STATUSES


def _human_reviewed(image: ImageAsset) -> bool:
    return bool(image.reviewed_at and str(image.reviewed_by or "").strip() not in AUTO_REVIEWERS)


def _created_key(image: ImageAsset) -> tuple[str, int]:
    created = image.created_at.isoformat() if image.created_at else "9999-12-31T23:59:59+00:00"
    return created, int(image.id or 0)


def _member_sort_key(member: dict[str, Any]) -> tuple[int, str, int]:
    return int(member["image_asset_id"]), str(member.get("batch_id") or ""), int(member["image_asset_id"])


def _canonical_rank(member: dict[str, Any]) -> int:
    if member["frozen_refs"]:
        return 1
    if member["review_status"] == "approved" and member["canonical_truth"] and member["accepted_bbox"] is not None:
        return 2
    if member["review_status"] == "approved" and member["canonical_truth"]:
        return 3
    if member["human_reviewed"]:
        return 4
    return 5


def _select_canonical(members: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the documented Phase A comparator, with id as final tie-break."""

    return min(
        members,
        key=lambda member: (
            _canonical_rank(member),
            member["created_at"] or "9999-12-31T23:59:59+00:00",
            int(member["image_asset_id"]),
        ),
    )


def _member_payload(
    image: ImageAsset,
    *,
    frozen_refs: list[dict[str, Any]],
    non_frozen_refs: list[dict[str, Any]],
    accepted_pool_source: bool,
    accepted_bbox: list[float] | None,
) -> dict[str, Any]:
    return {
        "image_asset_id": int(image.id),
        "batch_id": image.batch_id,
        "image_id": image.image_id,
        "object_name": image.object_name,
        "gcs_uri": image.gcs_uri,
        "created_at": image.created_at.isoformat() if image.created_at else None,
        "review_status": str(image.review_status or "").strip().lower(),
        "reviewed_by": image.reviewed_by,
        "reviewed_at": image.reviewed_at,
        "truth_species": _truth(image.truth_species),
        "truth_status": str(image.truth_status or "").strip(),
        "accepted_bbox": accepted_bbox,
        "accepted_pool_source": accepted_pool_source,
        "frozen_refs": frozen_refs,
        "non_frozen_refs": non_frozen_refs,
        "canonical_truth": _canonical_truth(image),
        "human_reviewed": _human_reviewed(image),
    }


def _accepted_pool_snapshot(db: Session) -> dict[str, Any]:
    """Read Accepted Pool source/manifest exposure without rebuilding it."""

    source_statement = (
        select(BatchCropReview, ImageAsset)
        .join(ImageAsset, ImageAsset.id == BatchCropReview.image_asset_id)
        .where(
            ImageAsset.review_status == "approved",
            BatchCropReview.status.in_(ACCEPTED_STATUSES),
            BatchCropReview.accepted_bbox_json.is_not(None),
        )
        .order_by(BatchCropReview.id)
    )
    source_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for review, image in db.execute(source_statement).all():
        bbox = _parse_bbox(review.accepted_bbox_json)
        if bbox is None:
            continue
        source_by_key[(image.batch_id, image.image_id)] = {
            "image_asset_id": int(image.id),
            "batch_id": image.batch_id,
            "image_id": image.image_id,
            "bbox": bbox,
            "review_id": int(review.id),
            "status": review.status,
        }

    manifest_rows: list[dict[str, Any]] = []
    manifest_sha256 = ""
    manifest_error: str | None = None
    try:
        manifest_rows, manifest_sha256 = _pool_manifest_rows()
    except Exception as exc:  # pragma: no cover - exercised by production config failures
        manifest_error = f"{type(exc).__name__}: {exc}"

    manifest_keys: set[tuple[str, str]] = set()
    for row in manifest_rows:
        batch_id = str(row.get("source_batch") or row.get("batch_id") or "").strip()
        image_id = str(row.get("source_image_id") or row.get("image_id") or "").strip()
        if batch_id and image_id:
            manifest_keys.add((batch_id, image_id))

    return {
        "source_by_key": source_by_key,
        "source_count": len(source_by_key),
        "manifest_rows": manifest_rows,
        "manifest_active_count": len(manifest_rows),
        "manifest_keys": manifest_keys,
        "manifest_sha256": manifest_sha256,
        "manifest_error": manifest_error,
    }


def _frozen_manifest_snapshot(db: Session) -> dict[str, Any]:
    """Hash Frozen Dataset manifest objects without changing them."""

    rows = db.scalars(
        select(DatasetVersion)
        .where(func.upper(DatasetVersion.status) == "FROZEN")
        .order_by(DatasetVersion.dataset_version)
    ).all()
    records: list[dict[str, Any]] = []
    client = None
    read_error: str | None = None
    for row in rows:
        uri = str(row.manifest_uri or "").strip()
        record: dict[str, Any] = {"dataset_version": row.dataset_version, "manifest_uri": uri}
        try:
            if uri.startswith("gs://"):
                body = uri[5:]
                bucket_name, object_name = body.split("/", 1)
                if client is None:
                    client = storage.Client()
                record["content_sha256"] = hashlib.sha256(
                    client.bucket(bucket_name).blob(object_name).download_as_bytes()
                ).hexdigest()
            else:
                record["content_sha256"] = None
                record["error"] = "manifest URI is not gs://"
        except Exception as exc:  # pragma: no cover - depends on production GCS
            record["content_sha256"] = None
            record["error"] = f"{type(exc).__name__}: {exc}"
            read_error = record["error"]
        records.append(record)
    canonical = json.dumps(records, ensure_ascii=False, sort_keys=True, default=_json_default).encode("utf-8")
    return {
        "records": records,
        "snapshot_sha256": hashlib.sha256(canonical).hexdigest(),
        "read_error": read_error,
    }


def _frozen_refs(db: Session, image_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
    if not image_ids:
        return {}
    rows = db.scalars(
        select(DatasetItem)
        .join(DatasetVersion, DatasetVersion.dataset_version == DatasetItem.dataset_version)
        .where(
            DatasetItem.image_asset_id.in_(image_ids),
            func.upper(DatasetVersion.status) == "FROZEN",
        )
    ).all()
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[int(row.image_asset_id)].append(
            {
                "dataset_version": row.dataset_version,
                "dataset_item_id": int(row.id),
                "split": row.split,
                "species": row.species_name,
                "gcs_uri": row.gcs_uri,
            }
        )
    return result


def _non_frozen_dataset_snapshot(db: Session) -> dict[str, Any]:
    """Capture non-Frozen lineage as diagnostics, never as Frozen authority."""

    versions = db.scalars(
        select(DatasetVersion)
        .where(func.upper(DatasetVersion.status) != "FROZEN")
        .order_by(DatasetVersion.dataset_version)
    ).all()
    rows = db.scalars(
        select(DatasetItem)
        .join(DatasetVersion, DatasetVersion.dataset_version == DatasetItem.dataset_version)
        .where(func.upper(DatasetVersion.status) != "FROZEN")
        .order_by(DatasetItem.id)
    ).all()
    refs_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        refs_by_image[int(row.image_asset_id)].append(
            {
                "dataset_version": row.dataset_version,
                "dataset_item_id": int(row.id),
                "split": row.split,
                "species": row.species_name,
                "gcs_uri": row.gcs_uri,
            }
        )
    version_details = {
        str(row.dataset_version): {
            "dataset_version": row.dataset_version,
            "status": str(row.status or "").strip(),
            "manifest_uri": str(row.manifest_uri or "").strip(),
            "dataset_item_count": 0,
            "duplicate_group_count": 0,
            "duplicate_member_count": 0,
            "_duplicate_groups": set(),
            "_duplicate_members": set(),
        }
        for row in versions
    }
    for row in rows:
        detail = version_details[str(row.dataset_version)]
        detail["dataset_item_count"] += 1
    return {
        "refs_by_image": refs_by_image,
        "rows": rows,
        "version_details": version_details,
    }


def _non_frozen_exposure(snapshot: dict[str, Any], groups: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize non-Frozen lineage separately from Frozen Dataset exposure."""

    version_details = snapshot["version_details"]
    duplicate_group_sha256: set[str] = set()
    duplicate_member_ids: set[int] = set()
    for group in groups:
        for member in group["members"]:
            refs = member.get("non_frozen_refs") or []
            if not refs:
                continue
            duplicate_group_sha256.add(str(group["sha256"]))
            duplicate_member_ids.add(int(member["image_asset_id"]))
            for ref in refs:
                detail = version_details[str(ref["dataset_version"])]
                detail["_duplicate_groups"].add(str(group["sha256"]))
                detail["_duplicate_members"].add(int(member["image_asset_id"]))

    statuses: dict[str, dict[str, int]] = {}
    for detail in version_details.values():
        detail["duplicate_group_count"] = len(detail.pop("_duplicate_groups"))
        detail["duplicate_member_count"] = len(detail.pop("_duplicate_members"))
        status = detail["status"] or "UNKNOWN"
        bucket = statuses.setdefault(
            status,
            {"dataset_versions": 0, "dataset_items": 0, "duplicate_groups": 0, "duplicate_members": 0},
        )
        bucket["dataset_versions"] += 1
        bucket["dataset_items"] += detail["dataset_item_count"]
        bucket["duplicate_groups"] += detail["duplicate_group_count"]
        bucket["duplicate_members"] += detail["duplicate_member_count"]

    details = list(version_details.values())
    return {
        "non_frozen_dataset_versions": len(details),
        "non_frozen_dataset_items": len(snapshot["rows"]),
        "non_frozen_duplicate_groups": len(duplicate_group_sha256),
        "non_frozen_duplicate_members": len(duplicate_member_ids),
        "statuses": statuses,
        "dataset_versions": details,
    }


def _frozen_dataset_item_count(db: Session) -> int:
    return int(
        db.scalar(
            select(func.count())
            .select_from(DatasetItem)
            .join(DatasetVersion, DatasetVersion.dataset_version == DatasetItem.dataset_version)
            .where(func.upper(DatasetVersion.status) == "FROZEN")
        )
        or 0
    )


def _accepted_bboxes(db: Session, image_ids: list[int]) -> dict[int, list[float]]:
    if not image_ids:
        return {}
    rows = db.scalars(
        select(BatchCropReview).where(
            BatchCropReview.image_asset_id.in_(image_ids),
            BatchCropReview.status.in_(ACCEPTED_STATUSES),
        )
    ).all()
    return {
        int(row.image_asset_id): bbox
        for row in rows
        if (bbox := _parse_bbox(row.accepted_bbox_json)) is not None
    }


def _serialize_group(group: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(group, ensure_ascii=False, default=_json_default))


def audit_registry(db: Session, *, environment: dict[str, Any]) -> dict[str, Any]:
    """Build all Phase A metrics and the proposed, non-mutating cleanup plan."""

    member_rows = db.scalars(select(GlobalImageDuplicateMember).order_by(GlobalImageDuplicateMember.sha256, GlobalImageDuplicateMember.image_asset_id)).all()
    by_sha: dict[str, list[int]] = defaultdict(list)
    for row in member_rows:
        by_sha[str(row.sha256)].append(int(row.image_asset_id))

    image_ids = sorted({image_id for ids in by_sha.values() for image_id in ids})
    images = {
        int(row.id): row
        for row in db.scalars(select(ImageAsset).where(ImageAsset.id.in_(image_ids))).all()
    } if image_ids else {}
    frozen = _frozen_refs(db, image_ids)
    non_frozen_snapshot = _non_frozen_dataset_snapshot(db)
    non_frozen = non_frozen_snapshot["refs_by_image"]
    bboxes = _accepted_bboxes(db, image_ids)
    accepted_pool = _accepted_pool_snapshot(db)
    source_keys = set(accepted_pool["source_by_key"])

    groups: list[dict[str, Any]] = []
    all_sha256 = set(by_sha)
    review_exposure = Counter()
    groups_with_multiple_approved = 0
    noncanonical_approved_candidates = 0
    same_batch_groups = 0
    cross_batch_groups = 0
    frozen_group_count = 0
    cross_split_groups = 0
    accepted_pool_groups = 0
    accepted_pool_duplicate_members = 0
    accepted_pool_redundant = 0
    redundant_images = 0
    conflict_groups: list[dict[str, Any]] = []

    for sha256, ids in sorted(by_sha.items()):
        if len(ids) < 2:
            continue
        members: list[dict[str, Any]] = []
        for image_id in ids:
            image = images.get(image_id)
            if image is None:
                continue
            members.append(
                _member_payload(
                    image,
                    frozen_refs=frozen.get(image_id, []),
                    non_frozen_refs=non_frozen.get(image_id, []),
                    accepted_pool_source=(image.batch_id, image.image_id) in source_keys,
                    accepted_bbox=bboxes.get(image_id),
                )
            )
        if len(members) < 2:
            continue

        batches = {member["batch_id"] for member in members}
        same_batch = len(batches) == 1
        same_batch_groups += int(same_batch)
        cross_batch_groups += int(not same_batch)
        frozen_group = any(member["frozen_refs"] for member in members)
        frozen_group_count += int(frozen_group)
        split_values = {
            ref["split"]
            for member in members
            for ref in member["frozen_refs"]
            if ref.get("split")
        }
        cross_split_groups += int(len(split_values) > 1)

        truth_values = {_truth_key(member["truth_species"]) for member in members if _truth(member["truth_species"])}
        truth_conflict = len(truth_values) > 1
        truth_classification = "conflict" if truth_conflict else ("consistent" if truth_values else "unlabeled")
        classification = "truth_conflict" if truth_conflict else ("same_batch" if same_batch else "cross_batch")

        approved_members = [member for member in members if member["review_status"] == "approved"]
        groups_with_multiple_approved += int(len(approved_members) > 1)
        for member in members:
            review_exposure[member["review_status"] or "unknown"] += 1

        canonical = None if truth_conflict else _select_canonical(members)
        redundant_ids = None if truth_conflict else [
            int(member["image_asset_id"])
            for member in members
            if member is not canonical
        ]
        if canonical is not None:
            noncanonical_approved_candidates += sum(
                1 for member in approved_members if member is not canonical
            )
            accepted_pool_redundant += sum(
                1 for member in members
                if member is not canonical and member["accepted_pool_source"]
            )
            redundant_images += len(redundant_ids or [])

        accepted_in_group = any(member["accepted_pool_source"] for member in members)
        accepted_pool_groups += int(accepted_in_group)
        accepted_pool_duplicate_members += sum(1 for member in members if member["accepted_pool_source"])
        group = {
            "sha256": sha256,
            "size": len(members),
            "classification": classification,
            "truth_classification": truth_classification,
            "proposed_canonical_id": int(canonical["image_asset_id"]) if canonical else None,
            "proposed_canonical": canonical,
            # A conflict intentionally has no proposed redundant list.
            "proposed_redundant_image_asset_ids": redundant_ids,
            "manual_conflict": truth_conflict,
            "members": members,
        }
        groups.append(group)
        if truth_conflict:
            conflict_groups.append(group)

    duplicate_member_images = sum(group["size"] for group in groups)
    unique_sha256 = len(all_sha256)
    review = {f"duplicate_{status}": int(review_exposure.get(status, 0)) for status in REVIEW_STATUSES}
    frozen_image_ids = {image_id for image_id in image_ids if frozen.get(image_id)}
    frozen_duplicate_member_ids = {
        int(member["image_asset_id"])
        for group in groups
        for member in group["members"]
        if member["frozen_refs"]
    }
    frozen_versions = {
        ref["dataset_version"]
        for group in groups
        for member in group["members"]
        for ref in member["frozen_refs"]
        if ref.get("dataset_version")
    }
    accepted_pool_source_ids = {
        int(row["image_asset_id"])
        for row in accepted_pool["source_by_key"].values()
    }
    cross_split_group_sha256 = [group["sha256"] for group in groups if len({
        ref["split"]
        for member in group["members"]
        for ref in member["frozen_refs"]
        if ref.get("split")
    }) > 1]
    non_frozen_exposure = _non_frozen_exposure(non_frozen_snapshot, groups)

    return {
        "environment": environment,
        "comparator": {
            "description": [
                "1. member in immutable Frozen Dataset",
                "2. approved + canonical truth + accepted bbox",
                "3. approved + canonical truth",
                "4. human-reviewed member",
                "5. earliest created_at",
                "final tie-break: lowest ImageAsset.id",
                "canonical truth means non-empty truth_species and truth_status in LIKELY_CORRECT/CANONICAL/CONFIRMED",
                "human-reviewed means reviewed_at is present and reviewed_by is not an automatic reviewer",
                "no filename or batch ordering is used",
            ]
        },
        "exact_duplicate_summary": {
            "total_image_assets": _count(db, ImageAsset),
            "unique_sha256": unique_sha256,
            "exact_duplicate_groups": len(groups),
            "affected_images": duplicate_member_images,
            "redundant_images": redundant_images,
            "same_batch_groups": same_batch_groups,
            "same_batch_duplicate_groups": same_batch_groups,
            "cross_batch_groups": cross_batch_groups,
            "cross_batch_duplicate_groups": cross_batch_groups,
            "manual_conflict_groups": len(conflict_groups),
            "manual_conflict_images": sum(group["size"] for group in conflict_groups),
            "global_image_content_rows": _count(db, GlobalImageContent),
            "global_image_content_distinct_sha256": int(db.scalar(select(func.count(func.distinct(GlobalImageContent.sha256)))) or 0),
            "global_duplicate_member_rows": _count(db, GlobalImageDuplicateMember),
        },
        "review_status_exposure": {
            **review,
            "groups_with_multiple_approved_members": groups_with_multiple_approved,
            "noncanonical_approved_candidates": noncanonical_approved_candidates,
        },
        "truth_conflicts": {
            "consistent_truth_groups": sum(1 for group in groups if group["truth_classification"] == "consistent"),
            "no_truth_groups": sum(1 for group in groups if group["truth_classification"] == "unlabeled"),
            "conflict_groups": len(conflict_groups),
            "conflict_members": sum(group["size"] for group in conflict_groups),
            "groups": [_serialize_group(group) for group in conflict_groups],
        },
        "accepted_pool_exposure": {
            "source_images": len(accepted_pool_source_ids),
            "source_rows": accepted_pool["source_count"],
            "exact_duplicate_groups_with_source": accepted_pool_groups,
            "accepted_pool_source_duplicate_groups": accepted_pool_groups,
            "accepted_pool_source_duplicate_members": accepted_pool_duplicate_members,
            "source_images_proposed_redundant": accepted_pool_redundant,
            "accepted_pool_redundant_members": accepted_pool_redundant,
            "materialized_manifest_active_rows": accepted_pool["manifest_active_count"],
            "materialized_manifest_duplicate_source_rows": sum(
                1 for key in accepted_pool["manifest_keys"] if key in source_keys
            ),
            "manifest_sha256": accepted_pool["manifest_sha256"],
            "manifest_read_error": accepted_pool["manifest_error"],
            "resync_performed": False,
        },
        "frozen_dataset_exposure": {
            "frozen_dataset_item_count": _frozen_dataset_item_count(db),
            "frozen_image_asset_count": len(frozen_image_ids),
            "exact_duplicate_groups_with_frozen_member": frozen_group_count,
            "duplicate_groups_present_in_frozen_datasets": frozen_group_count,
            "duplicate_members_present_in_frozen_datasets": len(frozen_duplicate_member_ids),
            "frozen_dataset_versions_affected": len(frozen_versions),
            "cross_split_exact_duplicate_groups": cross_split_groups,
            "cross_split_group_sha256": cross_split_group_sha256,
        },
        "non_frozen_dataset_exposure": non_frozen_exposure,
        "dry_run_preview": {
            "would_retain_canonical": len(groups) - len(conflict_groups),
            "would_quarantine_redundant": redundant_images,
            "would_remove_from_accepted_pool_source": accepted_pool_redundant,
            "would_leave_frozen_dataset_history_untouched": len(frozen_image_ids),
            "manual_conflict_groups_untouched": len(conflict_groups),
        },
        "groups": [_serialize_group(group) for group in groups],
    }


def _report(
    *,
    environment: dict[str, Any],
    coverage: dict[str, Any],
    audit: dict[str, Any],
    before_counts: dict[str, int],
    after_counts: dict[str, int],
) -> str:
    summary = audit["exact_duplicate_summary"]
    review = audit["review_status_exposure"]
    truth = audit["truth_conflicts"]
    accepted = audit["accepted_pool_exposure"]
    frozen = audit["frozen_dataset_exposure"]
    non_frozen = audit["non_frozen_dataset_exposure"]
    preview = audit["dry_run_preview"]
    safety = coverage.get("safety") or {}
    business_unchanged = before_counts == after_counts
    lines = [
        "# Coverage",
        "",
        f"- Environment: `{environment.get('name')}`; DB backend: `{environment.get('db_backend')}`; GCS bucket: `{environment.get('gcs_bucket')}`.",
        f"- Audit timestamp: `{environment.get('timestamp')}`; Git SHA: `{environment.get('git_sha') or 'unknown'}`; Cloud Run revision: `{environment.get('cloud_run_revision') or 'n/a'}`.",
        f"- total_image_assets: **{coverage.get('total_image_assets', 0)}**; processed: **{coverage.get('processed', 0)}**; missing: **{coverage.get('missing', 0)}**.",
        f"- created_global_content_rows: **{coverage.get('created_global_content_rows', 0)}**; created_global_duplicate_member_rows: **{coverage.get('created_global_duplicate_member_rows', 0)}**; historical_duplicate_members: **{coverage.get('historical_duplicate_members', 0)}**.",
        f"- coverage: **{coverage.get('coverage', 0):.6f}**; coverage_percent: **{coverage.get('coverage_percent', 0):.2f}%**; coverage_complete: **{coverage.get('coverage_complete', False)}**; status: **{coverage.get('status')}**.",
        "",
        "# Exact Duplicate Summary",
        "",
        f"- unique_sha256: **{summary['unique_sha256']}**; exact_duplicate_groups: **{summary['exact_duplicate_groups']}**; affected_images: **{summary['affected_images']}**; redundant_images: **{summary['redundant_images']}**.",
        f"- same_batch_duplicate_groups: **{summary['same_batch_duplicate_groups']}**; cross_batch_duplicate_groups: **{summary['cross_batch_duplicate_groups']}**; manual_conflict_groups: **{summary['manual_conflict_groups']}**.",
        "",
        "# Review Status Exposure",
        "",
        *[f"- {key}: **{review[key]}**" for key in ("duplicate_pending", "duplicate_needs_review", "duplicate_hard_case", "duplicate_approved", "duplicate_rejected")],
        f"- groups_with_multiple_approved_members: **{review['groups_with_multiple_approved_members']}**; noncanonical_approved_candidates: **{review['noncanonical_approved_candidates']}**.",
        "",
        "# Ground Truth Conflicts",
        "",
        f"- consistent_truth_groups: **{truth['consistent_truth_groups']}**; no_truth_groups: **{truth['no_truth_groups']}**; conflict_groups: **{truth['conflict_groups']}**; conflict_members: **{truth['conflict_members']}**.",
        "- Conflict groups have no proposed canonical winner and no proposed redundant list; they remain untouched for manual resolution.",
        "",
        "# Accepted Pool Exposure",
        "",
        f"- accepted_pool_source_duplicate_groups: **{accepted['accepted_pool_source_duplicate_groups']}**; accepted_pool_source_duplicate_members: **{accepted['accepted_pool_source_duplicate_members']}**; accepted_pool_redundant_members: **{accepted['accepted_pool_redundant_members']}**.",
        f"- materialized_manifest_active_rows: **{accepted['materialized_manifest_active_rows']}**; materialized_manifest_duplicate_source_rows: **{accepted['materialized_manifest_duplicate_source_rows']}**; resync_performed: **{accepted['resync_performed']}**.",
        "",
        "# Frozen Dataset Exposure",
        "",
        f"- duplicate_groups_present_in_frozen_datasets: **{frozen['duplicate_groups_present_in_frozen_datasets']}**; duplicate_members_present_in_frozen_datasets: **{frozen['duplicate_members_present_in_frozen_datasets']}**; frozen_dataset_versions_affected: **{frozen['frozen_dataset_versions_affected']}**.",
        f"- cross_split_exact_duplicate_groups: **{frozen['cross_split_exact_duplicate_groups']}**.",
        "- Frozen Dataset history is read-only in Phase A and was not changed.",
        "",
        "# Non-Frozen Dataset Diagnostic Exposure",
        "",
        f"- non_frozen_dataset_versions: **{non_frozen['non_frozen_dataset_versions']}**; non_frozen_dataset_items: **{non_frozen['non_frozen_dataset_items']}**; non_frozen_duplicate_groups: **{non_frozen['non_frozen_duplicate_groups']}**; non_frozen_duplicate_members: **{non_frozen['non_frozen_duplicate_members']}**.",
        f"- Status breakdown: `{json.dumps(non_frozen['statuses'], ensure_ascii=False, sort_keys=True)}`.",
        "- Non-Frozen lineage is diagnostic only and never participates in Frozen Dataset metrics or canonical rank 1.",
        *[
            f"- {detail['dataset_version']}: status **{detail['status']}**; manifest_uri `{detail['manifest_uri']}`; dataset_item_count **{detail['dataset_item_count']}**; duplicate_group_count **{detail['duplicate_group_count']}**; duplicate_member_count **{detail['duplicate_member_count']}**."
            for detail in non_frozen["dataset_versions"]
        ],
        "",
        "# Proposed Canonical Selection",
        "",
        "- Comparator: Frozen Dataset member; approved + canonical truth + accepted bbox; approved + canonical truth; human-reviewed; earliest `created_at`; lowest `ImageAsset.id` tie-break. Filename and batch ordering are not used.",
        f"- Proposed canonical selections: **{audit['dry_run_preview']['would_retain_canonical']}** non-conflict groups.",
        "",
        "# Cleanup Impact Preview",
        "",
        f"- would_retain_canonical: **{preview['would_retain_canonical']}**; would_quarantine_redundant: **{preview['would_quarantine_redundant']}**; would_remove_from_accepted_pool_source: **{preview['would_remove_from_accepted_pool_source']}**.",
        f"- would_leave_frozen_dataset_history_untouched: **{preview['would_leave_frozen_dataset_history_untouched']}**; manual_conflict_groups_untouched: **{preview['manual_conflict_groups_untouched']}**.",
        "- This is a dry-run preview only. No reject, delete, Accepted Pool resync, truth mutation, or Frozen Dataset mutation was executed.",
        "",
        "# Phase B Preconditions",
        "",
        "- Future Phase B must write only non-canonical, non-conflict `ImageAsset` review state and append a `ReviewEvent` with reason `GLOBAL_EXACT_DUPLICATE`, while preserving truth, notes, GCS, and history.",
        "- Future review write paths must return `GLOBAL_EXACT_DUPLICATE_NON_CANONICAL` (HTTP 409) for protected non-canonical members; Accepted Pool resync must be a separate explicit operation; Frozen Dataset rows remain immutable.",
        f"- Business counts before: `{json.dumps(before_counts, ensure_ascii=False, sort_keys=True)}`; after: `{json.dumps(after_counts, ensure_ascii=False, sort_keys=True)}`; business_data_mutated: **{not business_unchanged}**.",
        "",
        "## Safety",
        "",
        "- Phase A only; `--apply` is intentionally unavailable.",
        f"- ImageAsset count changed: **{'YES' if safety.get('image_asset_count_changed') else 'NO'}**; review state changed: **{'YES' if safety.get('review_state_changed') else 'NO'}**; Accepted Pool changed: **{'YES' if safety.get('accepted_pool_changed') else 'NO'}**; Frozen Dataset changed: **{'YES' if safety.get('frozen_dataset_changed') else 'NO'}**.",
        f"- GCS source images deleted: **{safety.get('gcs_source_images_deleted', 0)}**.",
        "- No DELETE ImageAsset/GCS, truth change, BatchCropReview change, Accepted Pool manifest resync, DatasetVersion/DatasetItem/Frozen Dataset mutation, or retraining/model publish was performed.",
    ]
    return "\n".join(lines) + "\n"


def _gcs_prefix_parts(value: str, default_bucket: str | None) -> tuple[str, str]:
    raw = str(value or "").strip()
    if raw.startswith("gs://"):
        body = raw[5:]
        if "/" not in body:
            return body, ""
        return body.split("/", 1)
    if not default_bucket:
        raise ValueError("GCS output prefix requires a bucket")
    return default_bucket, raw.strip("/")


def _persist_artifacts(
    output_dir: Path,
    *,
    gcs_output_prefix: str | None,
    environment: dict[str, Any],
    status: str,
    coverage: dict[str, Any],
    audit: dict[str, Any] | None,
    before_counts: dict[str, int],
    after_counts: dict[str, int],
) -> str | None:
    """Write execution metadata and optionally persist all evidence to GCS."""

    metadata = {
        "git_sha": environment.get("git_sha"),
        "execution_timestamp": environment.get("timestamp"),
        "project_id": environment.get("project_id"),
        "region": environment.get("region"),
        "cloud_run_job": environment.get("cloud_run_job"),
        "cloud_run_execution": environment.get("cloud_run_execution"),
        "db_backend": environment.get("db_backend"),
        "gcs_bucket": environment.get("gcs_bucket"),
        "production": bool(environment.get("production_mode")),
        "status": status,
        "gcs_evidence_prefix": gcs_output_prefix,
        "business_data_mutated": before_counts != after_counts,
        "safety": coverage.get("safety") or {},
        "gcs_source_image_deletions": 0,
        "artifact_files": [
            "coverage.json",
            "cleanup_plan.json",
            "conflicts.json",
            "report.md",
            "execution_metadata.json",
        ],
    }
    if audit:
        metadata["audit_summary"] = {
            "exact_duplicate_summary": audit.get("exact_duplicate_summary"),
            "review_status_exposure": audit.get("review_status_exposure"),
            "truth_conflicts": {
                key: value for key, value in (audit.get("truth_conflicts") or {}).items() if key != "groups"
            },
            "accepted_pool_exposure": audit.get("accepted_pool_exposure"),
            "frozen_dataset_exposure": audit.get("frozen_dataset_exposure"),
            "non_frozen_dataset_exposure": audit.get("non_frozen_dataset_exposure"),
            "dry_run_preview": audit.get("dry_run_preview"),
        }
    _write_json(output_dir / "execution_metadata.json", metadata)
    if not gcs_output_prefix:
        return None
    bucket_name, object_prefix = _gcs_prefix_parts(gcs_output_prefix, environment.get("gcs_bucket"))
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for filename in metadata["artifact_files"]:
        source = output_dir / filename
        if not source.exists():
            raise FileNotFoundError(f"required audit artifact is missing: {source}")
        object_name = f"{object_prefix.rstrip('/')}/{filename}" if object_prefix else filename
        content_type = "text/markdown" if filename.endswith(".md") else "application/json"
        bucket.blob(object_name).upload_from_filename(str(source), content_type=content_type)
    return f"gs://{bucket_name}/{object_prefix}".rstrip("/")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.bootstrap and not args.audit:
        raise SystemExit("at least one of --bootstrap or --audit is required")
    if args.production and args.limit is not None:
        raise SystemExit("--limit is forbidden with --production")
    if args.production and not os.getenv("GCS_BUCKET", "").strip():
        raise SystemExit("--production requires GCS_BUCKET")
    if args.production:
        cloud_sql = os.getenv("CLOUD_SQL_CONNECTION_NAME", "").strip()
        db_url = os.getenv("REGISTRY_DB_URL", "").strip()
        if not cloud_sql and (not db_url or db_url.startswith("sqlite:")):
            raise SystemExit("--production requires Cloud SQL or a non-SQLite REGISTRY_DB_URL")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gcs_output_prefix = args.gcs_output_prefix or os.getenv("AUDIT_GCS_OUTPUT_PREFIX", "").strip() or None
    db = SessionLocal()
    try:
        accepted_before = _accepted_pool_snapshot(db)
        frozen_before = _frozen_manifest_snapshot(db)
        before_counts = business_counts(db, accepted_before["source_count"])
        bind = db.get_bind()
        environment = {
            "name": os.getenv("AUDIT_ENVIRONMENT", "production" if args.production else "local"),
            "timestamp": _utc_iso(),
            "git_sha": _git_sha(),
            "db_backend": bind.dialect.name,
            "gcs_bucket": os.getenv("GCS_BUCKET", ""),
            "cloud_run_revision": os.getenv("K_REVISION"),
            "project_id": os.getenv("GCP_PROJECT_ID", "").strip() or os.getenv("GOOGLE_CLOUD_PROJECT", "").strip(),
            "region": os.getenv("GCP_REGION", "").strip() or os.getenv("REGION", "").strip(),
            "cloud_run_job": os.getenv("CLOUD_RUN_JOB", "").strip() or os.getenv("AUDIT_CLOUD_RUN_JOB", "").strip(),
            "cloud_run_execution": os.getenv("CLOUD_RUN_EXECUTION", "").strip() or os.getenv("AUDIT_CLOUD_RUN_EXECUTION", "").strip(),
            "gcs_output_prefix": gcs_output_prefix,
            "production_mode": bool(args.production),
        }

        total = before_counts["image_assets"]
        bootstrap_result: dict[str, Any] = {
            "total_image_assets": total,
            "processed": 0,
            "missing": 0,
            "missing_details": [],
            "created_global_content_rows": 0,
            "created_global_duplicate_member_rows": 0,
            "historical_duplicate_members": 0,
            "coverage": 1.0 if total == 0 else 0.0,
            "coverage_percent": 100.0 if total == 0 else 0.0,
            "coverage_complete": total == 0,
            "status": "NOT_RUN",
        }
        if args.bootstrap:
            raw = bootstrap_global_registry(db, bucket_name=os.getenv("GCS_BUCKET") or None, limit=args.limit)
            bootstrap_result = {
                "total_image_assets": total if args.limit is None else min(total, args.limit),
                "processed": int(raw.get("processed", 0)),
                "missing": int(raw.get("missing", 0)),
                "missing_details": list(raw.get("missing_details", [])),
                "created_global_content_rows": int(raw.get("created", 0)),
                "created_global_duplicate_member_rows": int(raw.get("created_global_duplicate_member_rows", 0)),
                "historical_duplicate_members": int(raw.get("historical_duplicate_members", 0)),
                "coverage": float(raw.get("coverage", 0.0)),
                "coverage_percent": float(raw.get("coverage", 0.0)) * 100.0,
                "coverage_complete": bool(raw.get("coverage_complete", False)),
                "status": str(raw.get("status", "BLOCKED_DEPENDENCY")),
            }
        accepted_after = _accepted_pool_snapshot(db)
        frozen_after = _frozen_manifest_snapshot(db)
        after_counts = business_counts(db, accepted_after["source_count"])
        business_data_mutated = before_counts != after_counts
        safety = {
            "image_asset_count_changed": before_counts["image_assets"] != after_counts["image_assets"],
            "approved_image_asset_count_changed": before_counts["approved_image_assets"] != after_counts["approved_image_assets"],
            "pending_image_asset_count_changed": before_counts["pending_image_assets"] != after_counts["pending_image_assets"],
            "rejected_image_asset_count_changed": before_counts["rejected_image_assets"] != after_counts["rejected_image_assets"],
            "review_state_changed": False,
            "accepted_pool_changed": accepted_before["manifest_sha256"] != accepted_after["manifest_sha256"],
            "accepted_pool_manifest_before": accepted_before["manifest_sha256"],
            "accepted_pool_manifest_after": accepted_after["manifest_sha256"],
            "frozen_dataset_changed": frozen_before["snapshot_sha256"] != frozen_after["snapshot_sha256"],
            "frozen_dataset_manifests_before": frozen_before,
            "frozen_dataset_manifests_after": frozen_after,
            "gcs_source_images_deleted": 0,
        }
        business_data_mutated = business_data_mutated or any(
            safety[key] for key in (
                "accepted_pool_changed",
                "frozen_dataset_changed",
            )
        )
        coverage = {**bootstrap_result, "business_data_mutated": business_data_mutated, "safety": safety}
        _write_json(output_dir / "coverage.json", {"environment": environment, **coverage, "before_counts": before_counts, "after_counts": after_counts})

        if args.bootstrap and not bootstrap_result["coverage_complete"]:
            report = "# Coverage\n\n- Production bootstrap did not reach 100% coverage. Audit and cleanup planning were not started.\n- Every missing ImageAsset detail is recorded in `coverage.json`; no cleanup mutation was attempted.\n"
            (output_dir / "report.md").write_text(report, encoding="utf-8")
            _write_json(output_dir / "conflicts.json", [])
            _write_json(output_dir / "cleanup_plan.json", {"status": "BLOCKED_DEPENDENCY", "groups": []})
            _persist_artifacts(
                output_dir,
                gcs_output_prefix=gcs_output_prefix,
                environment=environment,
                status="BLOCKED_DEPENDENCY",
                coverage=coverage,
                audit=None,
                before_counts=before_counts,
                after_counts=after_counts,
            )
            return {"coverage": coverage, "status": "BLOCKED_DEPENDENCY", "business_data_mutated": business_data_mutated}

        audit = audit_registry(db, environment=environment)
        _write_json(output_dir / "cleanup_plan.json", {
            "schema": "historical_exact_duplicate_cleanup_v1",
            "phase": "A",
            "mode": "dry_run",
            "apply_available": False,
            "comparator": audit["comparator"],
            "summary": audit["exact_duplicate_summary"],
            "groups": audit["groups"],
        })
        _write_json(output_dir / "conflicts.json", audit["truth_conflicts"]["groups"])
        report = _report(
            environment=environment,
            coverage=coverage,
            audit=audit,
            before_counts=before_counts,
            after_counts=after_counts,
        )
        (output_dir / "report.md").write_text(report, encoding="utf-8")
        result = {
            "environment": environment,
            "coverage": coverage,
            "audit": {key: value for key, value in audit.items() if key != "groups"},
            "before_counts": before_counts,
            "after_counts": after_counts,
            "business_data_mutated": business_data_mutated,
            "status": (
                "BLOCKED_INFRA"
                if args.production and any(
                    value
                    for value in (
                        accepted_before.get("manifest_error"),
                        accepted_after.get("manifest_error"),
                        frozen_before.get("read_error"),
                        frozen_after.get("read_error"),
                    )
                )
                else "COMPLETE" if coverage.get("coverage_complete") and not business_data_mutated else "BLOCKED_DEPENDENCY"
            ),
        }
        result["gcs_evidence_prefix"] = _persist_artifacts(
            output_dir,
            gcs_output_prefix=gcs_output_prefix,
            environment=environment,
            status=result["status"],
            coverage=coverage,
            audit=audit,
            before_counts=before_counts,
            after_counts=after_counts,
        )
        return result
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", action="store_true", help="run the additive historical registry bootstrap")
    parser.add_argument("--audit", action="store_true", help="generate the read-only exact duplicate audit")
    parser.add_argument("--production", action="store_true", help="require a non-SQLite production DB and GCS bucket")
    parser.add_argument("--output-dir", default="artifacts/historical_exact_duplicate_cleanup_v1")
    parser.add_argument("--gcs-output-prefix", default=None, help="upload evidence to this gs:// prefix after generation")
    parser.add_argument("--json", action="store_true", dest="as_json", help="print the machine-readable result")
    parser.add_argument("--limit", type=int, default=None, help="local/smoke limit; forbidden in production")
    args = parser.parse_args()
    result = run(args)
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(result["status"])
    return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
