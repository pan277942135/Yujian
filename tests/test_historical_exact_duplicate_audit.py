from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import app.dedupe  # noqa: F401 - registers ImageFingerprint before create_all
import app.platform.models  # noqa: F401 - registers platform tables before create_all
from app.db import Base
from app.dataset_models import DatasetItem
from app.exact_dedupe import bootstrap_global_registry
from app.models import (
    Batch,
    BatchCropReview,
    DatasetVersion,
    GlobalImageContent,
    GlobalImageDuplicateMember,
    ImageAsset,
)
from scripts import historical_exact_duplicate_audit as audit


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def _batch(db, batch_id: str):
    db.add(Batch(batch_id=batch_id, source="test", manifest_uri="manifest", raw_uri="raw"))
    db.flush()


def _image(
    db,
    *,
    batch_id: str,
    image_id: str,
    sha: str | None = None,
    created_at: datetime | None = None,
    review_status: str = "pending",
    truth_species: str | None = None,
    truth_status: str = "UNCERTAIN",
    reviewed_by: str | None = None,
    reviewed_at: datetime | None = None,
) -> ImageAsset:
    if db.get(Batch, batch_id) is None:
        _batch(db, batch_id)
    image = ImageAsset(
        batch_id=batch_id,
        image_id=image_id,
        file_name=f"{image_id}.jpg",
        object_name=f"raw/{batch_id}/{image_id}.jpg",
        gcs_uri=f"gs://test/raw/{batch_id}/{image_id}.jpg",
        created_at=created_at or datetime(2026, 1, 1, tzinfo=timezone.utc),
        review_status=review_status,
        truth_species=truth_species,
        truth_status=truth_status,
        reviewed_by=reviewed_by,
        reviewed_at=reviewed_at,
    )
    db.add(image)
    db.flush()
    if sha:
        now = datetime(2026, 1, 2, tzinfo=timezone.utc)
        if db.scalar(select(GlobalImageContent).where(GlobalImageContent.sha256 == sha)) is None:
            db.add(
                GlobalImageContent(
                    sha256=sha,
                    lifecycle_status="ACTIVE",
                    canonical_batch_id=batch_id,
                    canonical_image_id=image_id,
                    canonical_image_asset_id=image.id,
                    canonical_object_name=image.object_name,
                    source="test",
                    first_seen_at=image.created_at,
                    created_at=now,
                    updated_at=now,
                )
            )
        db.add(
            GlobalImageDuplicateMember(
                sha256=sha,
                image_asset_id=image.id,
                batch_id=batch_id,
                image_id=image_id,
                object_name=image.object_name,
            )
        )
    return image


def _fingerprint(db, image: ImageAsset, sha: str):
    from app.dedupe import ImageFingerprint

    db.add(
        ImageFingerprint(
            image_asset_id=image.id,
            batch_id=image.batch_id,
            sha256=sha,
            phash_json="[]",
            dhash="0" * 16,
            crop_hash="",
            histogram_json="[]",
            width=1,
            height=1,
            fingerprint_version="test",
        )
    )


def _dataset_item(db, version: DatasetVersion, image: ImageAsset, *, split: str = "train"):
    db.add(
        DatasetItem(
            dataset_version=version.dataset_version,
            image_asset_id=image.id,
            batch_id=image.batch_id,
            image_id=image.image_id,
            gcs_uri=image.gcs_uri,
            species_key="grass",
            species_name="草鱼",
            class_index=0,
            split=split,
        )
    )


def _seed_bootstrap_images(db):
    same = "a" * 64
    unique = "b" * 64
    first = _image(db, batch_id="B1", image_id="A", sha=None)
    second = _image(db, batch_id="B2", image_id="B", sha=None)
    third = _image(db, batch_id="B3", image_id="C", sha=None)
    _fingerprint(db, first, same)
    _fingerprint(db, second, same)
    _fingerprint(db, third, unique)
    db.commit()
    return same, unique


def _bootstrap_and_audit(db, monkeypatch):
    result = bootstrap_global_registry(db, bucket_name="test")
    monkeypatch.setattr(audit, "_pool_manifest_rows", lambda: ([], ""))
    return result, audit.audit_registry(db, environment={"name": "test"})


def test_bootstrap_unique_sha_is_complete(db):
    same, unique = _seed_bootstrap_images(db)
    result = bootstrap_global_registry(db, bucket_name="test")
    assert result["processed"] == 3
    assert result["missing"] == 0
    assert result["coverage_complete"] is True
    assert {row.sha256 for row in db.scalars(select(GlobalImageContent)).all()} == {same, unique}


def test_bootstrap_same_sha_creates_one_global_content_and_members(db):
    same, _ = _seed_bootstrap_images(db)
    bootstrap_global_registry(db, bucket_name="test")
    assert db.scalar(select(GlobalImageContent).where(GlobalImageContent.sha256 == same)) is not None
    assert len(db.scalars(select(GlobalImageDuplicateMember).where(GlobalImageDuplicateMember.sha256 == same)).all()) == 2


def test_bootstrap_is_idempotent_without_duplicate_members(db):
    _seed_bootstrap_images(db)
    first = bootstrap_global_registry(db, bucket_name="test")
    content_count = len(db.scalars(select(GlobalImageContent)).all())
    member_count = len(db.scalars(select(GlobalImageDuplicateMember)).all())
    canonical_before = {
        row.sha256: row.canonical_image_asset_id
        for row in db.scalars(select(GlobalImageContent)).all()
    }
    second = bootstrap_global_registry(db, bucket_name="test")
    assert first["created"] == 2
    assert second["created"] == 0
    assert len(db.scalars(select(GlobalImageContent)).all()) == content_count
    assert len(db.scalars(select(GlobalImageDuplicateMember)).all()) == member_count
    canonical_after = {
        row.sha256: row.canonical_image_asset_id
        for row in db.scalars(select(GlobalImageContent)).all()
    }
    assert canonical_before == canonical_after


def test_audit_same_batch_group(db, monkeypatch):
    sha = "1" * 64
    _image(db, batch_id="B1", image_id="A", sha=sha)
    _image(db, batch_id="B1", image_id="B", sha=sha)
    db.commit()
    _, result = _bootstrap_and_audit(db, monkeypatch)
    assert result["exact_duplicate_summary"]["same_batch_groups"] == 1
    assert result["exact_duplicate_summary"]["cross_batch_groups"] == 0


def test_audit_cross_batch_group(db, monkeypatch):
    sha = "2" * 64
    _image(db, batch_id="B1", image_id="A", sha=sha)
    _image(db, batch_id="B2", image_id="B", sha=sha)
    db.commit()
    _, result = _bootstrap_and_audit(db, monkeypatch)
    assert result["exact_duplicate_summary"]["cross_batch_groups"] == 1


def test_consistent_truth_is_not_a_conflict(db, monkeypatch):
    sha = "3" * 64
    _image(db, batch_id="B1", image_id="A", sha=sha, truth_species="草鱼", truth_status="LIKELY_CORRECT")
    _image(db, batch_id="B2", image_id="B", sha=sha, truth_species="草鱼", truth_status="LIKELY_CORRECT")
    db.commit()
    _, result = _bootstrap_and_audit(db, monkeypatch)
    assert result["truth_conflicts"]["conflict_groups"] == 0
    assert result["groups"][0]["truth_classification"] == "consistent"


def test_conflicting_truth_is_listed(db, monkeypatch):
    sha = "4" * 64
    _image(db, batch_id="B1", image_id="A", sha=sha, truth_species="草鱼", truth_status="LIKELY_CORRECT")
    _image(db, batch_id="B2", image_id="B", sha=sha, truth_species="鲤鱼", truth_status="LIKELY_CORRECT")
    db.commit()
    _, result = _bootstrap_and_audit(db, monkeypatch)
    assert result["truth_conflicts"]["conflict_groups"] == 1
    assert result["truth_conflicts"]["groups"][0]["sha256"] == sha


def test_conflict_has_no_auto_redundant_list_or_canonical(db, monkeypatch):
    sha = "5" * 64
    _image(db, batch_id="B1", image_id="A", sha=sha, truth_species="草鱼")
    _image(db, batch_id="B2", image_id="B", sha=sha, truth_species="鲤鱼")
    db.commit()
    _, result = _bootstrap_and_audit(db, monkeypatch)
    group = result["groups"][0]
    assert group["manual_conflict"] is True
    assert group["proposed_canonical_id"] is None
    assert group["proposed_redundant_image_asset_ids"] is None


def test_frozen_member_wins_canonical_comparator(db, monkeypatch):
    sha = "6" * 64
    frozen = _image(db, batch_id="B1", image_id="A", sha=sha, created_at=datetime(2026, 2, 1, tzinfo=timezone.utc))
    other = _image(db, batch_id="B2", image_id="B", sha=sha, created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), review_status="approved", truth_species="草鱼", truth_status="LIKELY_CORRECT")
    version = DatasetVersion(dataset_version="DS_FROZEN", manifest_uri="gs://test/manifest.csv", git_commit="frozen", status="FROZEN")
    db.add(version)
    db.flush()
    _dataset_item(db, version, frozen, split="test")
    db.commit()
    _, result = _bootstrap_and_audit(db, monkeypatch)
    assert result["groups"][0]["proposed_canonical_id"] == frozen.id
    assert result["groups"][0]["proposed_canonical_id"] != other.id


def test_approved_canonical_truth_and_bbox_wins(db, monkeypatch):
    sha = "7" * 64
    first = _image(db, batch_id="B1", image_id="A", sha=sha, review_status="approved", truth_species="草鱼", truth_status="LIKELY_CORRECT")
    second = _image(db, batch_id="B2", image_id="B", sha=sha, review_status="approved", truth_species="草鱼", truth_status="LIKELY_CORRECT")
    db.add(BatchCropReview(batch_id=second.batch_id, image_asset_id=second.id, image_id=second.image_id, status="ACCEPTED", accepted_bbox_json="[0.1,0.1,0.5,0.5]"))
    db.commit()
    _, result = _bootstrap_and_audit(db, monkeypatch)
    assert result["groups"][0]["proposed_canonical_id"] == second.id
    assert result["groups"][0]["proposed_canonical_id"] != first.id


def test_canonical_tie_break_is_lowest_image_asset_id(db, monkeypatch):
    sha = "8" * 64
    first = _image(db, batch_id="B2", image_id="Z", sha=sha, created_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    second = _image(db, batch_id="B1", image_id="A", sha=sha, created_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    db.commit()
    _, result = _bootstrap_and_audit(db, monkeypatch)
    assert result["groups"][0]["proposed_canonical_id"] == min(first.id, second.id)


def test_accepted_pool_source_exposure_is_read_only(db, monkeypatch):
    sha = "9" * 64
    first = _image(db, batch_id="B1", image_id="A", sha=sha, review_status="approved", truth_species="草鱼", truth_status="LIKELY_CORRECT")
    second = _image(db, batch_id="B2", image_id="B", sha=sha, review_status="approved", truth_species="草鱼", truth_status="LIKELY_CORRECT")
    db.add(BatchCropReview(batch_id=first.batch_id, image_asset_id=first.id, image_id=first.image_id, status="ACCEPTED", accepted_bbox_json="[0.1,0.1,0.5,0.5]"))
    db.add(BatchCropReview(batch_id=second.batch_id, image_asset_id=second.id, image_id=second.image_id, status="ACCEPTED", accepted_bbox_json="[0.1,0.1,0.5,0.5]"))
    db.commit()
    monkeypatch.setattr(audit, "_pool_manifest_rows", lambda: ([{"source_batch": "B1", "source_image_id": "A", "pool_status": "ACTIVE"}], "manifest"))
    result = audit.audit_registry(db, environment={"name": "test"})
    assert result["accepted_pool_exposure"]["source_images"] == 2
    assert result["accepted_pool_exposure"]["exact_duplicate_groups_with_source"] == 1
    assert result["accepted_pool_exposure"]["resync_performed"] is False


def test_frozen_dataset_exposure_is_counted(db, monkeypatch):
    sha = "a" * 64
    first = _image(db, batch_id="B1", image_id="A", sha=sha)
    second = _image(db, batch_id="B2", image_id="B", sha=sha)
    version = DatasetVersion(dataset_version="DS_FROZEN", manifest_uri="gs://test/manifest.csv", git_commit="frozen", status="FROZEN")
    db.add(version)
    db.flush()
    _dataset_item(db, version, first)
    db.commit()
    _, result = _bootstrap_and_audit(db, monkeypatch)
    assert result["frozen_dataset_exposure"]["frozen_image_asset_count"] == 1
    assert result["frozen_dataset_exposure"]["exact_duplicate_groups_with_frozen_member"] == 1
    assert second.id not in {first.id}


def test_cross_split_exact_duplicate_is_exposed(db, monkeypatch):
    sha = "b" * 64
    first = _image(db, batch_id="B1", image_id="A", sha=sha)
    second = _image(db, batch_id="B2", image_id="B", sha=sha)
    version = DatasetVersion(dataset_version="DS_FROZEN", manifest_uri="gs://test/manifest.csv", git_commit="frozen", status="FROZEN")
    db.add(version)
    db.flush()
    db.add_all([
        DatasetItem(dataset_version=version.dataset_version, image_asset_id=first.id, batch_id=first.batch_id, image_id=first.image_id, gcs_uri=first.gcs_uri, species_key="grass", species_name="草鱼", class_index=0, split="train"),
        DatasetItem(dataset_version=version.dataset_version, image_asset_id=second.id, batch_id=second.batch_id, image_id=second.image_id, gcs_uri=second.gcs_uri, species_key="grass", species_name="草鱼", class_index=0, split="test"),
    ])
    db.commit()
    _, result = _bootstrap_and_audit(db, monkeypatch)
    assert result["frozen_dataset_exposure"]["cross_split_exact_duplicate_groups"] == 1


def test_non_frozen_manifest_is_excluded_from_frozen_snapshot(db, monkeypatch):
    version = DatasetVersion(
        dataset_version="DS_BBOX",
        manifest_uri="gs://test/missing.csv",
        git_commit="bbox",
        status="BBOX_PROCESSING",
    )
    db.add(version)
    db.commit()

    class NoStorageAccess:
        def __init__(self):
            raise AssertionError("non-FROZEN manifest must not be read")

    monkeypatch.setattr(audit.storage, "Client", NoStorageAccess)
    snapshot = audit._frozen_manifest_snapshot(db)
    assert snapshot["records"] == []
    assert snapshot["read_error"] is None


def test_frozen_missing_manifest_is_read_error_and_blocks_production(db, monkeypatch, tmp_path):
    version = DatasetVersion(
        dataset_version="DS_FROZEN_MISSING",
        manifest_uri="gs://test/missing.csv",
        git_commit="frozen",
        status="FROZEN",
    )
    db.add(version)
    db.commit()

    class MissingBlob:
        def download_as_bytes(self):
            raise FileNotFoundError("missing manifest")

    class MissingStorage:
        def bucket(self, _name):
            return self

        def blob(self, _name):
            return MissingBlob()

    monkeypatch.setattr(audit.storage, "Client", lambda: MissingStorage())
    monkeypatch.setattr(audit, "SessionLocal", lambda: db)
    monkeypatch.setattr(audit, "_pool_manifest_rows", lambda: ([], ""))
    monkeypatch.setenv("GCS_BUCKET", "test")
    monkeypatch.setenv("CLOUD_SQL_CONNECTION_NAME", "project:region:instance")
    result = audit.run(
        audit.argparse.Namespace(
            bootstrap=False,
            audit=True,
            production=True,
            output_dir=str(tmp_path),
            gcs_output_prefix=None,
            as_json=False,
            limit=None,
        )
    )
    assert result["status"] == "BLOCKED_INFRA"
    assert result["coverage"]["safety"]["frozen_dataset_changed"] is False


def test_non_frozen_lineage_does_not_elevate_canonical_rank(db, monkeypatch):
    sha = "e" * 64
    bbox_image = _image(db, batch_id="B1", image_id="BBOX", sha=sha, created_at=datetime(2026, 1, 2, tzinfo=timezone.utc))
    canonical = _image(
        db,
        batch_id="B2",
        image_id="CANONICAL",
        sha=sha,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        review_status="approved",
        truth_species="草鱼",
        truth_status="LIKELY_CORRECT",
    )
    version = DatasetVersion(dataset_version="DS_BBOX", manifest_uri="gs://test/missing.csv", git_commit="bbox", status="BBOX_PROCESSING")
    db.add(version)
    db.flush()
    _dataset_item(db, version, bbox_image)
    db.commit()

    _, result = _bootstrap_and_audit(db, monkeypatch)
    group = result["groups"][0]
    bbox_member = next(row for row in group["members"] if row["image_asset_id"] == bbox_image.id)
    assert bbox_member["frozen_refs"] == []
    assert len(bbox_member["non_frozen_refs"]) == 1
    assert group["proposed_canonical_id"] == canonical.id
    assert result["frozen_dataset_exposure"]["duplicate_groups_present_in_frozen_datasets"] == 0
    assert result["non_frozen_dataset_exposure"]["non_frozen_duplicate_groups"] == 1
    assert result["non_frozen_dataset_exposure"]["dataset_versions"][0]["status"] == "BBOX_PROCESSING"


def test_same_image_non_frozen_and_frozen_lineage_keeps_only_frozen_refs(db, monkeypatch):
    sha = "f" * 64
    image = _image(db, batch_id="B1", image_id="A", sha=sha)
    other = _image(db, batch_id="B2", image_id="B", sha=sha)
    bbox_version = DatasetVersion(dataset_version="DS_BBOX", manifest_uri="gs://test/bbox.csv", git_commit="bbox", status="BBOX_PROCESSING")
    frozen_version = DatasetVersion(dataset_version="DS_FROZEN", manifest_uri="gs://test/frozen.csv", git_commit="frozen", status="FROZEN")
    db.add_all([bbox_version, frozen_version])
    db.flush()
    _dataset_item(db, bbox_version, image, split="train")
    _dataset_item(db, frozen_version, image, split="test")
    db.commit()

    _, result = _bootstrap_and_audit(db, monkeypatch)
    member = next(row for row in result["groups"][0]["members"] if row["image_asset_id"] == image.id)
    assert [row["dataset_version"] for row in member["frozen_refs"]] == ["DS_FROZEN"]
    assert [row["dataset_version"] for row in member["non_frozen_refs"]] == ["DS_BBOX"]
    assert result["frozen_dataset_exposure"]["cross_split_exact_duplicate_groups"] == 0
    assert result["non_frozen_dataset_exposure"]["non_frozen_duplicate_groups"] == 1


def test_cross_split_metrics_ignore_non_frozen_dataset_items(db, monkeypatch):
    sha = "0" * 64
    first = _image(db, batch_id="B1", image_id="A", sha=sha)
    second = _image(db, batch_id="B2", image_id="B", sha=sha)
    frozen_version = DatasetVersion(dataset_version="DS_FROZEN", manifest_uri="gs://test/frozen.csv", git_commit="frozen", status="FROZEN")
    bbox_version = DatasetVersion(dataset_version="DS_BBOX", manifest_uri="gs://test/bbox.csv", git_commit="bbox", status="BBOX_PROCESSING")
    db.add_all([frozen_version, bbox_version])
    db.flush()
    _dataset_item(db, frozen_version, first, split="train")
    _dataset_item(db, bbox_version, second, split="test")
    db.commit()

    _, result = _bootstrap_and_audit(db, monkeypatch)
    assert result["frozen_dataset_exposure"]["duplicate_groups_present_in_frozen_datasets"] == 1
    assert result["frozen_dataset_exposure"]["duplicate_members_present_in_frozen_datasets"] == 1
    assert result["frozen_dataset_exposure"]["cross_split_exact_duplicate_groups"] == 0
    assert result["non_frozen_dataset_exposure"]["non_frozen_duplicate_groups"] == 1


def test_dry_run_preserves_business_counts(db, monkeypatch):
    sha = "c" * 64
    _image(db, batch_id="B1", image_id="A", sha=sha, review_status="approved")
    _image(db, batch_id="B2", image_id="B", sha=sha, review_status="pending")
    db.commit()
    before = audit.business_counts(db, 0)
    _, result = _bootstrap_and_audit(db, monkeypatch)
    after = audit.business_counts(db, 0)
    assert before == after
    assert result["dry_run_preview"]["manual_conflict_groups_untouched"] == 0


def test_near_duplicate_different_sha_is_excluded(db, monkeypatch):
    first = _image(db, batch_id="B1", image_id="A", sha="d" * 64)
    second = _image(db, batch_id="B2", image_id="B", sha="e" * 64)
    db.commit()
    _, result = _bootstrap_and_audit(db, monkeypatch)
    assert first.id != second.id
    assert result["exact_duplicate_summary"]["exact_duplicate_groups"] == 0
    assert result["exact_duplicate_summary"]["unique_sha256"] == 2


def test_report_artifact_contract_has_no_apply_and_required_sections(db, monkeypatch, tmp_path):
    sha = hashlib.sha256(b"same").hexdigest()
    _image(db, batch_id="B1", image_id="A", sha=sha)
    _image(db, batch_id="B2", image_id="B", sha=sha)
    db.commit()
    monkeypatch.setattr(audit, "_pool_manifest_rows", lambda: ([], ""))
    result = audit.audit_registry(db, environment={"name": "test"})
    plan = {
        "apply_available": False,
        "groups": result["groups"],
    }
    (tmp_path / "cleanup_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    assert plan["apply_available"] is False
    assert "proposed_canonical_id" in plan["groups"][0]
    assert "members" in plan["groups"][0]
    assert all(section in audit._report(environment={"name": "test"}, coverage={"total_image_assets": 2, "processed": 2, "missing": 0, "created_global_content_rows": 0, "historical_duplicate_members": 0, "coverage": 1.0, "coverage_percent": 100.0, "coverage_complete": True, "status": "COMPLETE"}, audit=result, before_counts={}, after_counts={}) for section in ("# Coverage", "# Exact Duplicate Summary", "# Ground Truth Conflicts", "# Phase B Preconditions"))
