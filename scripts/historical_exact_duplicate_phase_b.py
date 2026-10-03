"""Phase B: quarantine historical exact duplicates from future training sources.

The operator is deliberately separate from the Phase A audit.  It accepts one
byte-pinned Phase A cleanup plan, performs a complete live-state drift check,
and applies only training-eligibility and non-conflict registry-pointer
updates in one database transaction.  It never deletes an ImageAsset, source
object, review row, truth value, or frozen dataset row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from sqlalchemy import func, inspect, select

from app.accepted_pool import (
    _pool_manifest_rows,
    accepted_pool_summary,
    start_accepted_pool_sync,
    step_accepted_pool_job,
)
from app.db import SessionLocal, _ensure_training_eligibility_columns, engine
from app.dataset_models import DatasetItem
from app.models import DatasetVersion, GlobalImageContent, GlobalImageDuplicateMember, ImageAsset


PHASE_A_RUN_ID = "37101698837"
PHASE_A_AUDIT_SHA = "4de6de59704586fb52b9ae626bfe8f8b1c4d9abf"
PHASE_A_ARTIFACT_ID = "11265843128"
REASON_DUPLICATE = "GLOBAL_EXACT_DUPLICATE"
REASON_CONFLICT = "GLOBAL_EXACT_TRUTH_CONFLICT"
REASONS = {REASON_DUPLICATE, REASON_CONFLICT}
REQUIRED_ARTIFACTS = {"cleanup_plan.json", "conflicts.json", "coverage.json", "execution_metadata.json"}
SCHEMA_COLUMNS = (
    "training_eligible",
    "training_exclusion_reason",
    "duplicate_of_image_asset_id",
    "training_eligibility_source",
    "training_eligibility_updated_at",
)
SCHEMA_INDEXES = (
    "ix_image_assets_training_eligible",
    "ix_image_assets_training_exclusion_reason",
    "ix_image_assets_duplicate_of_image_asset_id",
)


class PhaseAPlanDrift(RuntimeError):
    code = "PHASE_A_PLAN_DRIFT"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Any) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else (str(value) if value is not None else None)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _schema_snapshot() -> dict[str, Any]:
    inspector = inspect(engine)
    if not inspector.has_table("image_assets"):
        return {"status": "PASS", "table_exists": False, "columns": [], "indexes": [], "foreign_keys": []}
    columns = {str(item["name"]) for item in inspector.get_columns("image_assets")}
    indexes = {str(item["name"]) for item in inspector.get_indexes("image_assets")}
    foreign_keys = [
        {
            "name": item.get("name"),
            "constrained_columns": sorted(str(value) for value in item.get("constrained_columns") or []),
            "referred_table": item.get("referred_table"),
            "referred_columns": sorted(str(value) for value in item.get("referred_columns") or []),
        }
        for item in inspector.get_foreign_keys("image_assets")
        if "duplicate_of_image_asset_id" in (item.get("constrained_columns") or [])
    ]
    return {
        "status": "PASS",
        "table_exists": True,
        "columns": sorted(column for column in columns if column in SCHEMA_COLUMNS),
        "indexes": sorted(index for index in indexes if index in SCHEMA_INDEXES),
        "foreign_keys": foreign_keys,
    }


def _schema_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {
        "columns_added": sorted(set(after.get("columns") or []) - set(before.get("columns") or [])),
        "indexes_added": sorted(set(after.get("indexes") or []) - set(before.get("indexes") or [])),
        "fk_added": [item for item in after.get("foreign_keys") or [] if item not in (before.get("foreign_keys") or [])],
    }


def _sanitize_text(value: str) -> str:
    text = str(value or "")
    text = re.sub(r"(?i)(password|secret|token|authorization|credential)[^\n=]*=[^\n]*", r"\1=[REDACTED]", text)
    text = re.sub(r"(?i)Bearer\s+[A-Za-z0-9._-]+", "Bearer [REDACTED]", text)
    return text


def _failure_metadata(*, mode: str, stage: str, exc: BaseException, schema_before: dict[str, Any] | None, schema_after: dict[str, Any] | None, authority: dict[str, Any] | None = None) -> dict[str, Any]:
    authority = authority or {}
    return {
        "status": "FAILED",
        "phase": "B",
        "mode": mode,
        "stage": stage,
        "exception_type": type(exc).__name__,
        "error_message": _sanitize_text(str(exc)),
        "phase_a_run_id": authority.get("run_id", PHASE_A_RUN_ID),
        "phase_a_audit_sha": authority.get("audit_sha", PHASE_A_AUDIT_SHA),
        "cleanup_plan_sha256": authority.get("cleanup_plan_sha256", os.getenv("CLEANUP_PLAN_SHA256", "")),
        "app_git_commit": os.getenv("APP_GIT_COMMIT", ""),
        "schema_before": schema_before,
        "schema_after": schema_after,
        "schema_diff": _schema_diff(schema_before or {}, schema_after or {}),
    }


def _upload_evidence(output_dir: Path, prefix: str, *, storage_client_factory=None) -> dict[str, Any]:
    value = str(prefix or "").strip()
    if not value:
        return {"status": "SKIPPED", "uploaded": []}
    if not value.startswith("gs://") or "/" not in value[5:]:
        raise ValueError("PHASE_B_OUTPUT_GCS_PREFIX must be a gs:// URI")
    from google.cloud import storage

    bucket_name, object_prefix = value[5:].split("/", 1)
    client = (storage_client_factory or storage.Client)()
    bucket = client.bucket(bucket_name)
    uploaded = []
    for path in sorted(output_dir.glob("*")):
        if not path.is_file():
            continue
        blob_name = f"{object_prefix.rstrip('/')}/{path.name}"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(path))
        if not blob.exists(client):
            raise RuntimeError(f"evidence upload verification failed: {blob_name}")
        uploaded.append(path.name)
    if not uploaded:
        raise RuntimeError("no Phase B evidence files were produced")
    return {"status": "PASS", "prefix": value, "uploaded": uploaded}


def _download_phase_a(prefix: str, destination: Path) -> Path:
    from google.cloud import storage

    value = str(prefix or "").strip()
    if not value.startswith("gs://") or "/" not in value[5:]:
        raise ValueError("PHASE_A_PLAN_GCS_PREFIX must be a gs:// URI")
    bucket_name, object_prefix = value[5:].split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    destination.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_ARTIFACTS | {"report.md"}:
        blob = bucket.blob(f"{object_prefix.rstrip('/')}/{name}")
        if not blob.exists(client):
            raise FileNotFoundError(f"Phase A artifact is missing: {blob.name}")
        target = destination / name
        target.write_bytes(blob.download_as_bytes())
    return destination


def load_phase_a_artifact(root: Path, *, audit_sha: str = PHASE_A_AUDIT_SHA) -> dict[str, Any]:
    present = {item.name for item in root.iterdir() if item.is_file()}
    missing = REQUIRED_ARTIFACTS - present
    if missing:
        raise ValueError(f"Phase A artifact is incomplete: missing {sorted(missing)}")
    plan_path = root / "cleanup_plan.json"
    plan = _read_json(plan_path)
    metadata = _read_json(root / "execution_metadata.json")
    coverage = _read_json(root / "coverage.json")
    summary = plan.get("summary") or {}
    audit_summary = metadata.get("audit_summary") or {}
    truth = audit_summary.get("truth_conflicts") or {}
    if metadata.get("status") != "COMPLETE":
        raise ValueError("Phase A execution_metadata.status is not COMPLETE")
    if float(coverage.get("coverage_percent", -1)) != 100.0 or coverage.get("missing") != 0:
        raise ValueError("Phase A coverage gate failed")
    if str(metadata.get("git_sha") or "") != audit_sha:
        raise ValueError("Phase A audit SHA mismatch")
    if int(summary.get("exact_duplicate_groups", -1)) != 1229:
        raise ValueError("Phase A exact duplicate group count mismatch")
    if int(truth.get("conflict_groups", -1)) != 28:
        raise ValueError("Phase A truth conflict group count mismatch")
    if int(truth.get("conflict_members", -1)) != 76:
        raise ValueError("Phase A truth conflict member count mismatch")
    return {
        "run_id": PHASE_A_RUN_ID,
        "artifact_id": PHASE_A_ARTIFACT_ID,
        "audit_sha": audit_sha,
        "cleanup_plan_sha256": sha256_file(plan_path),
        "plan": plan,
        "metadata": metadata,
        "coverage": coverage,
    }


def _groups(authority: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normal: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for group in authority["plan"].get("groups") or []:
        if group.get("manual_conflict") or group.get("truth_classification") == "conflict":
            conflicts.append(group)
        else:
            if not group.get("proposed_canonical_id"):
                raise ValueError(f"non-conflict group has no canonical: {group.get('sha256')}")
            normal.append(group)
    return normal, conflicts


def _member_expectations(group: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(member["image_asset_id"]): member for member in group.get("members") or []}


def _plan_ids(authority: dict[str, Any]) -> set[int]:
    normal, conflicts = _groups(authority)
    ids: set[int] = set()
    for group in normal + conflicts:
        ids.update(_member_expectations(group))
    return ids


def _asset_payload(image: ImageAsset) -> dict[str, Any]:
    return {
        "id": image.id,
        "batch_id": image.batch_id,
        "image_id": image.image_id,
        "truth_species": image.truth_species,
        "truth_status": image.truth_status,
        "review_status": image.review_status,
        "training_eligible": bool(image.training_eligible),
        "training_exclusion_reason": image.training_exclusion_reason,
        "duplicate_of_image_asset_id": image.duplicate_of_image_asset_id,
    }


def _expected_member_ids(group: dict[str, Any]) -> set[int]:
    return {int(member["image_asset_id"]) for member in group.get("members") or []}


def _live_drift_gate(db, authority: dict[str, Any]) -> dict[str, Any]:
    normal, conflicts = _groups(authority)
    plan = normal + conflicts
    summary = authority["plan"].get("summary") or {}
    expected_total = int(summary.get("total_image_assets", 0))
    actual_total = int(db.scalar(select(func.count()).select_from(ImageAsset)) or 0)
    reasons: list[str] = []
    if actual_total != expected_total:
        reasons.append(f"ImageAsset count changed: expected={expected_total}, actual={actual_total}")
    target_ids = _plan_ids(authority)
    assets = {image.id: image for image in db.scalars(select(ImageAsset).where(ImageAsset.id.in_(target_ids))).all()}
    missing_ids = sorted(target_ids - set(assets))
    if missing_ids:
        reasons.append(f"target ImageAsset rows disappeared: {missing_ids[:20]}")
    membership_rows = db.execute(
        select(GlobalImageDuplicateMember.sha256, GlobalImageDuplicateMember.image_asset_id)
        .where(GlobalImageDuplicateMember.sha256.in_([str(group.get("sha256")) for group in plan]))
    ).all()
    membership: dict[str, set[int]] = defaultdict(set)
    for sha, image_id in membership_rows:
        membership[str(sha)].add(int(image_id))
    for group in plan:
        sha = str(group.get("sha256"))
        expected = _expected_member_ids(group)
        if membership.get(sha, set()) != expected:
            reasons.append(f"membership changed for sha256={sha}")
        for member in group.get("members") or []:
            image = assets.get(int(member["image_asset_id"]))
            if image is None:
                continue
            if image.truth_species != member.get("truth_species") or image.truth_status != member.get("truth_status"):
                reasons.append(f"truth classification changed for image_asset_id={image.id}")
    if reasons:
        raise PhaseAPlanDrift("; ".join(reasons))
    return {
        "status": "PASS",
        "total_image_assets": actual_total,
        "target_image_assets": len(target_ids),
        "target_groups": len(plan),
        "missing_target_ids": [],
        "membership_checked": True,
        "truth_classification_checked": True,
    }


def _counts(db) -> dict[str, int]:
    total = int(db.scalar(select(func.count()).select_from(ImageAsset)) or 0)
    eligible = int(db.scalar(select(func.count()).select_from(ImageAsset).where(ImageAsset.training_eligible.is_(True))) or 0)
    excluded = db.execute(
        select(ImageAsset.training_exclusion_reason, func.count())
        .where(ImageAsset.training_eligible.is_(False))
        .group_by(ImageAsset.training_exclusion_reason)
    ).all()
    result = {
        "total_image_assets": total,
        "training_eligible": eligible,
        "training_ineligible": total - eligible,
        REASON_DUPLICATE: 0,
        REASON_CONFLICT: 0,
    }
    for reason, count in excluded:
        if reason in REASONS:
            result[str(reason)] = int(count)
    return result


def _frozen_snapshot(db) -> dict[str, Any]:
    versions = [
        {
            "dataset_version": row.dataset_version,
            "parent_version": row.parent_version,
            "manifest_uri": row.manifest_uri,
            "train_count": row.train_count,
            "val_count": row.val_count,
            "test_count": row.test_count,
            "git_commit": row.git_commit,
            "status": row.status,
        }
        for row in db.scalars(select(DatasetVersion).where(func.upper(DatasetVersion.status) == "FROZEN").order_by(DatasetVersion.dataset_version)).all()
    ]
    items = [
        {
            "id": row.id,
            "dataset_version": row.dataset_version,
            "image_asset_id": row.image_asset_id,
            "batch_id": row.batch_id,
            "image_id": row.image_id,
            "gcs_uri": row.gcs_uri,
            "split": row.split,
            "species_key": row.species_key,
            "species_name": row.species_name,
        }
        for row in db.scalars(
            select(DatasetItem)
            .join(DatasetVersion, DatasetItem.dataset_version == DatasetVersion.dataset_version)
            .where(func.upper(DatasetVersion.status) == "FROZEN")
            .order_by(DatasetItem.id)
        ).all()
    ]
    return {
        "frozen_dataset_versions": len(versions),
        "frozen_dataset_items": len(items),
        "dataset_versions_sha256": _sha256_bytes(_json_bytes(versions)),
        "dataset_items_sha256": _sha256_bytes(_json_bytes(items)),
        "versions": versions,
    }


def _apply(db, authority: dict[str, Any], *, fail_after: int | None = None) -> dict[str, Any]:
    normal, conflicts = _groups(authority)
    source = (
        "historical_exact_duplicate_cleanup_v1;"
        f"run={authority['run_id']};audit_sha={authority['audit_sha']};"
        f"cleanup_plan_sha256={authority['cleanup_plan_sha256']}"
    )
    changed_assets: list[int] = []
    newly_excluded: list[int] = []
    pointer_changes: list[int] = []
    conflict_changes: list[int] = []
    registry_pointer_changes: list[str] = []
    mutation_count = 0
    db.rollback()
    try:
        with db.begin():
            _live_drift_gate(db, authority)
            now = _now()
            for group in normal:
                canonical_id = int(group["proposed_canonical_id"])
                canonical = db.get(ImageAsset, canonical_id)
                if canonical is None:
                    raise PhaseAPlanDrift(f"canonical ImageAsset disappeared: {canonical_id}")
                desired = (True, None, None, None)
                current = (bool(canonical.training_eligible), canonical.training_exclusion_reason, canonical.duplicate_of_image_asset_id, canonical.training_eligibility_source)
                if current != desired:
                    canonical.training_eligible, canonical.training_exclusion_reason, canonical.duplicate_of_image_asset_id = desired[:3]
                    canonical.training_eligibility_source = None
                    canonical.training_eligibility_updated_at = now
                    changed_assets.append(canonical.id)
                    mutation_count += 1
                for redundant_id in group.get("proposed_redundant_image_asset_ids") or []:
                    image = db.get(ImageAsset, int(redundant_id))
                    if image is None:
                        raise PhaseAPlanDrift(f"redundant ImageAsset disappeared: {redundant_id}")
                    desired = (False, REASON_DUPLICATE, canonical_id, source)
                    current = (bool(image.training_eligible), image.training_exclusion_reason, image.duplicate_of_image_asset_id, image.training_eligibility_source)
                    if current != desired:
                        if image.training_eligible:
                            newly_excluded.append(image.id)
                        if image.duplicate_of_image_asset_id != canonical_id:
                            pointer_changes.append(image.id)
                        image.training_eligible, image.training_exclusion_reason, image.duplicate_of_image_asset_id = desired[:3]
                        image.training_eligibility_source = source
                        image.training_eligibility_updated_at = now
                        changed_assets.append(image.id)
                        mutation_count += 1
                    if fail_after is not None and mutation_count >= fail_after:
                        raise RuntimeError("injected Phase B transaction failure")
                for registry in db.scalars(select(GlobalImageContent).where(GlobalImageContent.sha256 == str(group["sha256"]))).all():
                    desired_registry = (canonical.batch_id, canonical.image_id, canonical.id, canonical.object_name)
                    current_registry = (registry.canonical_batch_id, registry.canonical_image_id, registry.canonical_image_asset_id, registry.canonical_object_name)
                    if current_registry != desired_registry:
                        registry.canonical_batch_id, registry.canonical_image_id, registry.canonical_image_asset_id, registry.canonical_object_name = desired_registry
                        registry_pointer_changes.append(str(group["sha256"]))
            for group in conflicts:
                for member_id in _expected_member_ids(group):
                    image = db.get(ImageAsset, member_id)
                    if image is None:
                        raise PhaseAPlanDrift(f"conflict ImageAsset disappeared: {member_id}")
                    desired = (False, REASON_CONFLICT, None, source)
                    current = (bool(image.training_eligible), image.training_exclusion_reason, image.duplicate_of_image_asset_id, image.training_eligibility_source)
                    if current != desired:
                        if image.training_eligible:
                            newly_excluded.append(image.id)
                        conflict_changes.append(image.id)
                        image.training_eligible, image.training_exclusion_reason, image.duplicate_of_image_asset_id = desired[:3]
                        image.training_eligibility_source = source
                        image.training_eligibility_updated_at = now
                        changed_assets.append(image.id)
                        mutation_count += 1
                    if fail_after is not None and mutation_count >= fail_after:
                        raise RuntimeError("injected Phase B transaction failure")
    except Exception:
        db.rollback()
        raise
    return {
        "status": "APPLIED",
        "changed_image_asset_ids": sorted(set(changed_assets)),
        "newly_excluded_image_asset_ids": sorted(set(newly_excluded)),
        "newly_excluded": len(set(newly_excluded)),
        "canonical_pointer_changes": sorted(set(pointer_changes)),
        "canonical_pointer_change_count": len(set(pointer_changes)),
        "truth_conflict_rows_changed": len(set(conflict_changes)),
        "global_content_pointer_changes": sorted(set(registry_pointer_changes)),
        "global_content_pointer_change_count": len(set(registry_pointer_changes)),
    }


def _post_audit(db, authority: dict[str, Any]) -> dict[str, Any]:
    normal, conflicts = _groups(authority)
    eligible_duplicate_groups = 0
    eligible_duplicate_excess = 0
    eligible_conflict_members = 0
    canonical_failures: list[int] = []
    redundant_failures: list[int] = []
    conflict_failures: list[int] = []
    for group in normal:
        ids = _expected_member_ids(group)
        assets = {row.id: row for row in db.scalars(select(ImageAsset).where(ImageAsset.id.in_(ids))).all()}
        canonical_id = int(group["proposed_canonical_id"])
        canonical = assets.get(canonical_id)
        if canonical is None or not canonical.training_eligible or canonical.training_exclusion_reason is not None or canonical.duplicate_of_image_asset_id is not None:
            canonical_failures.append(canonical_id)
        redundant = [assets[int(item)] for item in group.get("proposed_redundant_image_asset_ids") or [] if int(item) in assets]
        eligible_redundant = [row for row in redundant if row.training_eligible]
        if eligible_redundant:
            eligible_duplicate_groups += 1
            eligible_duplicate_excess += len(eligible_redundant)
        redundant_failures.extend(
            row.id for row in redundant
            if row.training_eligible or row.training_exclusion_reason != REASON_DUPLICATE or row.duplicate_of_image_asset_id != canonical_id
        )
    for group in conflicts:
        ids = _expected_member_ids(group)
        members = db.scalars(select(ImageAsset).where(ImageAsset.id.in_(ids))).all()
        eligible_conflict_members += sum(1 for row in members if row.training_eligible)
        conflict_failures.extend(row.id for row in members if row.training_eligible or row.training_exclusion_reason != REASON_CONFLICT or row.duplicate_of_image_asset_id is not None)
    counts = _counts(db)
    return {
        **counts,
        "eligible_exact_duplicate_groups": eligible_duplicate_groups,
        "eligible_exact_duplicate_member_excess": eligible_duplicate_excess,
        "eligible_truth_conflict_members": eligible_conflict_members,
        "eligible_truth_conflict_groups": sum(1 for group in conflicts if any(
            row.training_eligible for row in db.scalars(select(ImageAsset).where(ImageAsset.id.in_(_expected_member_ids(group)))).all()
        )),
        "canonical_failures": sorted(set(canonical_failures)),
        "redundant_failures": sorted(set(redundant_failures)),
        "conflict_failures": sorted(set(conflict_failures)),
        "post_apply_gate": not (
            canonical_failures or redundant_failures or conflict_failures or eligible_duplicate_groups or
            eligible_conflict_members
        ),
    }


def _pool_snapshot(db) -> dict[str, Any]:
    try:
        rows, digest = _pool_manifest_rows()
    except Exception as exc:
        return {"status": "UNAVAILABLE", "error": str(exc), "active_count": 0, "manifest_sha256": "", "rows": []}
    return {"status": "PASS", "active_count": len(rows), "manifest_sha256": digest, "rows": rows}


def _run_pool_sync(db, authority: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    job = start_accepted_pool_sync(db)
    for _ in range(100000):
        if str(job.get("status") or "").upper() in {"SUCCESS", "FAILED"}:
            break
        job = step_accepted_pool_job(job["job_id"])
    if str(job.get("status") or "").upper() != "SUCCESS":
        raise RuntimeError(f"Accepted Pool sync failed: {job}")
    after = _pool_snapshot(db)
    if after.get("status") != "PASS":
        raise RuntimeError("Accepted Pool post-sync manifest is unavailable")
    normal, conflicts = _groups(authority)
    duplicate_ids = {int(item) for group in normal for item in group.get("proposed_redundant_image_asset_ids") or []}
    conflict_ids = {int(item) for group in conflicts for item in _expected_member_ids(group)}
    before_keys = {f"{row.get('source_batch') or row.get('batch_id')}:{row.get('image_id')}" for row in before.get("rows") or []}
    after_keys = {f"{row.get('source_batch') or row.get('batch_id')}:{row.get('image_id')}" for row in after.get("rows") or []}
    removed = before_keys - after_keys
    by_key = {
        f"{image.batch_id}:{image.image_id}": image
        for image in db.scalars(select(ImageAsset)).all()
    }
    removed_duplicate = sum(1 for key in removed if by_key.get(key) and by_key[key].id in duplicate_ids)
    removed_conflict = sum(1 for key in removed if by_key.get(key) and by_key[key].id in conflict_ids)
    active_excluded: list[int] = []
    active_image_ids: set[int] = set()
    for row in after.get("rows") or []:
        image = by_key.get(f"{row.get('source_batch') or row.get('batch_id')}:{row.get('image_id')}")
        if image is None or not image.training_eligible:
            active_excluded.append(image.id if image else -1)
        else:
            active_image_ids.add(image.id)
    if active_excluded:
        raise RuntimeError(f"Accepted Pool contains ineligible ImageAsset rows: {active_excluded[:20]}")
    eligible_duplicate_groups = 0
    eligible_duplicate_excess = 0
    for group in normal:
        group_count = len(active_image_ids & _expected_member_ids(group))
        if group_count > 1:
            eligible_duplicate_groups += 1
            eligible_duplicate_excess += group_count - 1
    eligible_conflict_members = sum(
        1 for group in conflicts if active_image_ids & _expected_member_ids(group)
        for _image_id in active_image_ids & _expected_member_ids(group)
    )
    if eligible_duplicate_groups or eligible_conflict_members:
        raise RuntimeError(
            "Accepted Pool eligibility gate failed: "
            f"duplicate_groups={eligible_duplicate_groups}, "
            f"duplicate_excess={eligible_duplicate_excess}, "
            f"conflict_members={eligible_conflict_members}"
        )
    return {
        "status": "PASS",
        "job": {key: value for key, value in job.items() if key not in {"source_refs", "active_keys"}},
        "before": {key: value for key, value in before.items() if key != "rows"},
        "after": {key: value for key, value in after.items() if key != "rows"},
        "removed_from_pool_due_to_duplicate": removed_duplicate,
        "removed_from_pool_due_to_truth_conflict": removed_conflict,
        "eligible_exact_duplicate_groups": eligible_duplicate_groups,
        "eligible_exact_duplicate_member_excess": eligible_duplicate_excess,
        "eligible_truth_conflict_members": eligible_conflict_members,
    }


def execute_phase_b(
    db,
    authority: dict[str, Any],
    output_dir: Path,
    *,
    mode: str,
    sync_pool: bool = False,
    idempotency_check: bool = False,
    fail_after: int | None = None,
    stage_callback=None,
) -> dict[str, Any]:
    def stage(value: str) -> None:
        if stage_callback is not None:
            stage_callback(value)

    output_dir.mkdir(parents=True, exist_ok=True)
    normal, conflicts = _groups(authority)
    expected = {
        "target_non_conflict_groups": len(normal),
        "target_redundant_images": sum(len(group.get("proposed_redundant_image_asset_ids") or []) for group in normal),
        "target_truth_conflict_groups": len(conflicts),
        "target_truth_conflict_images": sum(len(_expected_member_ids(group)) for group in conflicts),
    }
    expected["target_total_exclusions"] = expected["target_redundant_images"] + expected["target_truth_conflict_images"]
    expected["expected_training_eligible"] = int((authority["plan"].get("summary") or {}).get("total_image_assets", 0)) - expected["target_total_exclusions"]
    stage("LIVE_DRIFT_GATE")
    drift = _live_drift_gate(db, authority)
    stage("FROZEN_SNAPSHOT")
    before = _counts(db)
    frozen_before = _frozen_snapshot(db)
    stage("ACCEPTED_POOL_SNAPSHOT")
    pool_before = _pool_snapshot(db)
    plan_evidence = {
        "phase": "B",
        "mode": mode,
        "phase_a_run_id": authority["run_id"],
        "phase_a_artifact_id": authority["artifact_id"],
        "phase_a_audit_sha": authority["audit_sha"],
        "cleanup_plan_sha256": authority["cleanup_plan_sha256"],
        "drift_gate": drift,
        "before": before,
        "expected": expected,
        "non_conflict_canonical_mappings": [
            {"sha256": group["sha256"], "canonical_image_asset_id": group["proposed_canonical_id"], "redundant_image_asset_ids": group.get("proposed_redundant_image_asset_ids") or []}
            for group in normal
        ],
        "truth_conflict_groups": [{"sha256": group["sha256"], "image_asset_ids": sorted(_expected_member_ids(group))} for group in conflicts],
        "frozen_before": frozen_before,
    }
    stage("DRY_RUN_PLAN" if mode == "dry-run" else "APPLY")
    _write_json(output_dir / "phase_b_plan.json", plan_evidence)
    if mode == "dry-run":
        post = _post_audit(db, authority)
        _write_json(output_dir / "phase_b_apply.json", {"status": "NOT_EXECUTED", "reason": "dry-run"})
        _write_json(output_dir / "phase_b_post_audit.json", {"status": "NOT_EXECUTED", "current": post})
        _write_json(output_dir / "accepted_pool_post_sync.json", {"status": "NOT_EXECUTED", "before": pool_before})
        _write_json(output_dir / "execution_metadata.json", {
            "status": "DRY_RUN",
            "phase": "B",
            "phase_a_run_id": authority["run_id"],
            "phase_a_audit_sha": authority["audit_sha"],
            "cleanup_plan_sha256": authority["cleanup_plan_sha256"],
            "before": before,
            "expected": expected,
            "gcs_source_images_deleted": 0,
            "app_git_commit": os.getenv("APP_GIT_COMMIT", ""),
        })
        return {"status": "DRY_RUN", "expected": expected, "before": before}

    apply_result = _apply(db, authority, fail_after=fail_after)
    stage("POOL_SYNC")
    pool_result = _run_pool_sync(db, authority, pool_before) if sync_pool else {"status": "NOT_EXECUTED"}
    pool_after_snapshot = _pool_snapshot(db) if sync_pool else None
    stage("POST_AUDIT")
    post = _post_audit(db, authority)
    frozen_after = _frozen_snapshot(db)
    if frozen_after["dataset_versions_sha256"] != frozen_before["dataset_versions_sha256"] or frozen_after["dataset_items_sha256"] != frozen_before["dataset_items_sha256"]:
        raise RuntimeError("frozen dataset lineage changed during Phase B")
    second = {"status": "NOT_EXECUTED"}
    if idempotency_check:
        stage("IDEMPOTENCY")
        second = _apply(db, authority)
        second_post = _post_audit(db, authority)
        second_pool = _run_pool_sync(db, authority, pool_after_snapshot) if sync_pool and pool_after_snapshot else {"status": "NOT_EXECUTED"}
        pool_changed = bool(
            sync_pool
            and (
                second_pool.get("removed_from_pool_due_to_duplicate")
                or second_pool.get("removed_from_pool_due_to_truth_conflict")
                or (second_pool.get("after") or {}).get("manifest_sha256") != (pool_result.get("after") or {}).get("manifest_sha256")
            )
        )
        if second["newly_excluded"] or second["canonical_pointer_change_count"] or second["truth_conflict_rows_changed"] or second_post != post or pool_changed:
            raise RuntimeError(f"Phase B idempotency gate failed: {second}")
        second = {**second, "post_audit": second_post, "accepted_pool": second_pool, "accepted_pool_changes": 0}
    _write_json(output_dir / "phase_b_apply.json", apply_result)
    _write_json(output_dir / "phase_b_post_audit.json", {"status": "PASS", "post": post, "frozen_after": frozen_after})
    _write_json(output_dir / "accepted_pool_post_sync.json", pool_result)
    _write_json(output_dir / "execution_metadata.json", {
        "status": "COMPLETE",
        "phase": "B",
        "phase_a_run_id": authority["run_id"],
        "phase_a_audit_sha": authority["audit_sha"],
        "cleanup_plan_sha256": authority["cleanup_plan_sha256"],
        "before": before,
        "apply": apply_result,
        "after": post,
        "frozen_before": frozen_before,
        "frozen_after": frozen_after,
        "second_apply": second,
        "gcs_source_images_deleted": 0,
        "phase_c_started": False,
        "model_training_started": False,
        "app_git_commit": os.getenv("APP_GIT_COMMIT", ""),
    })
    return {"status": "COMPLETE", "apply": apply_result, "after": post, "second_apply": second, "pool": pool_result}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a-dir", default=os.getenv("PHASE_A_ARTIFACT_DIR", ""))
    parser.add_argument("--output-dir", default="/tmp/historical-exact-duplicate-phase-b")
    parser.add_argument("--mode", choices=("dry-run", "apply"), required=True)
    parser.add_argument("--sync-pool", action="store_true")
    parser.add_argument("--idempotency-check", action="store_true")
    parser.add_argument("--fail-after", type=int)
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = args.mode
    stage = "LOAD_PHASE_A"
    authority = None
    schema_before = None
    schema_after = None
    db = None
    exit_code = 0
    caught_exception = None
    caught_traceback = ""

    def set_stage(value: str) -> None:
        nonlocal stage
        stage = value

    try:
        root = Path(args.phase_a_dir) if args.phase_a_dir else Path("/tmp/phase-a")
        if not root.exists():
            root = _download_phase_a(os.getenv("PHASE_A_PLAN_GCS_PREFIX", ""), root)
        authority = load_phase_a_artifact(root)

        set_stage("SCHEMA_CONTRACT")
        schema_before = _schema_snapshot()
        # Phase A runs against an existing production schema and deliberately
        # does not call SQLAlchemy create_all().  Phase B must do the same:
        # apply only this additive, idempotent contract.
        _ensure_training_eligibility_columns()
        schema_after = _schema_snapshot()

        set_stage("DB_CONNECT")
        db = SessionLocal()
        result = execute_phase_b(
            db,
            authority,
            output_dir,
            mode=mode,
            sync_pool=args.sync_pool,
            idempotency_check=args.idempotency_check,
            fail_after=args.fail_after,
            stage_callback=set_stage,
        )
        metadata_path = output_dir / "execution_metadata.json"
        metadata = _read_json(metadata_path)
        metadata.update({
            "mode": mode,
            "stage": "PERSIST_EVIDENCE",
            "schema_before": schema_before,
            "schema_after": schema_after,
            "schema_diff": _schema_diff(schema_before or {}, schema_after or {}),
            "app_git_commit": os.getenv("APP_GIT_COMMIT", ""),
        })
        _write_json(metadata_path, metadata)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    except PhaseAPlanDrift as exc:
        caught_exception = exc
        caught_traceback = traceback.format_exc()
        exit_code = 20
        if db is not None:
            db.rollback()
        print(f"BLOCKED_DEPENDENCY — {PhaseAPlanDrift.code}: {exc}")
    except Exception as exc:
        caught_exception = exc
        caught_traceback = traceback.format_exc()
        exit_code = 1
        if db is not None:
            db.rollback()
        print(f"PHASE_B_FAILED: {type(exc).__name__}: {exc}")
    finally:
        if db is not None:
            db.close()
        if exit_code:
            failure = _failure_metadata(
                mode=mode,
                stage=stage,
                exc=caught_exception or RuntimeError("unknown Phase B failure"),
                schema_before=schema_before,
                schema_after=schema_after,
                authority=authority,
            )
            _write_json(output_dir / "execution_metadata.json", failure)
            _write_json(output_dir / "failure.json", {
                "status": "FAILED",
                "stage": stage,
                "exception_type": failure["exception_type"],
                "error_message": failure["error_message"],
                "traceback": _sanitize_text(caught_traceback),
            })
        try:
            upload = _upload_evidence(output_dir, os.getenv("PHASE_B_OUTPUT_GCS_PREFIX", ""))
            print(json.dumps({"evidence_upload": upload}, ensure_ascii=False, sort_keys=True))
        except Exception as upload_exc:
            print(f"PHASE_B_EVIDENCE_UPLOAD_FAILED: {_sanitize_text(str(upload_exc))}")
            if not exit_code:
                exit_code = 1
                failure = _failure_metadata(
                    mode=mode,
                    stage="PERSIST_EVIDENCE",
                    exc=upload_exc,
                    schema_before=schema_before,
                    schema_after=schema_after,
                    authority=authority,
                )
                _write_json(output_dir / "execution_metadata.json", failure)
                _write_json(output_dir / "failure.json", {
                    "status": "FAILED",
                    "stage": "PERSIST_EVIDENCE",
                    "exception_type": failure["exception_type"],
                    "error_message": failure["error_message"],
                    "traceback": _sanitize_text(traceback.format_exc()),
                })
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
