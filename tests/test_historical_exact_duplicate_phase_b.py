from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.dataset_models import DatasetItem
from app.freeze_policy import _training_eligibility_gate
from app.models import Batch, BatchCropReview, DatasetVersion, GlobalImageContent, GlobalImageDuplicateMember, ImageAsset, SpeciesCatalog
from app.accepted_pool import _source_rows
from app.platform.services.crop_dataset import _accepted_pool_rows
from app.presence import FishPresenceResult
from scripts.historical_exact_duplicate_phase_b import (
    PhaseAPlanDrift,
    _apply,
    _counts,
    _post_audit,
    _upload_evidence,
    execute_phase_b,
    main,
)


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    tables = [
        Batch.__table__,
        ImageAsset.__table__,
        GlobalImageContent.__table__,
        GlobalImageDuplicateMember.__table__,
        BatchCropReview.__table__,
        SpeciesCatalog.__table__,
        FishPresenceResult.__table__,
        DatasetVersion.__table__,
        DatasetItem.__table__,
    ]
    Base.metadata.create_all(engine, tables=tables)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def _authority(db):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    db.add(Batch(batch_id="B1", source="test", manifest_uri="manifest", raw_uri="raw"))
    db.add(Batch(batch_id="B2", source="test", manifest_uri="manifest", raw_uri="raw"))
    db.flush()
    canonical = ImageAsset(
        batch_id="B1", image_id="canonical", file_name="c.jpg", object_name="c.jpg", gcs_uri="gs://c",
        review_status="approved", truth_species="草鱼", truth_status="LIKELY_CORRECT", created_at=now,
    )
    redundant = ImageAsset(
        batch_id="B2", image_id="redundant", file_name="r.jpg", object_name="r.jpg", gcs_uri="gs://r",
        review_status="approved", truth_species="草鱼", truth_status="LIKELY_CORRECT", created_at=now,
    )
    conflict_a = ImageAsset(
        batch_id="B1", image_id="conflict-a", file_name="a.jpg", object_name="a.jpg", gcs_uri="gs://a",
        review_status="approved", truth_species="青鱼", truth_status="LIKELY_CORRECT", created_at=now,
    )
    conflict_b = ImageAsset(
        batch_id="B2", image_id="conflict-b", file_name="b.jpg", object_name="b.jpg", gcs_uri="gs://b",
        review_status="approved", truth_species="草鱼", truth_status="LIKELY_CORRECT", created_at=now,
    )
    db.add_all([canonical, redundant, conflict_a, conflict_b])
    db.flush()
    for sha, rows in {"sha-normal": [canonical, redundant], "sha-conflict": [conflict_a, conflict_b]}.items():
        db.add(
            GlobalImageContent(
                sha256=sha, lifecycle_status="ACTIVE", canonical_batch_id=rows[0].batch_id,
                canonical_image_id=rows[0].image_id, canonical_image_asset_id=rows[0].id,
                canonical_object_name=rows[0].object_name, first_seen_at=now, created_at=now, updated_at=now,
            )
        )
        for row in rows:
            db.add(GlobalImageDuplicateMember(sha256=sha, image_asset_id=row.id, batch_id=row.batch_id, image_id=row.image_id, object_name=row.object_name))
    db.commit()
    return {
        "run_id": "37101698837",
        "artifact_id": "11265843128",
        "audit_sha": "4de6de59704586fb52b9ae626bfe8f8b1c4d9abf",
        "cleanup_plan_sha256": "plan-sha",
        "plan": {
            "summary": {"total_image_assets": 4},
            "groups": [
                {
                    "sha256": "sha-normal", "truth_classification": "consistent", "manual_conflict": False,
                    "proposed_canonical_id": canonical.id, "proposed_redundant_image_asset_ids": [redundant.id],
                    "members": [
                        {"image_asset_id": canonical.id, "truth_species": "草鱼", "truth_status": "LIKELY_CORRECT"},
                        {"image_asset_id": redundant.id, "truth_species": "草鱼", "truth_status": "LIKELY_CORRECT"},
                    ],
                },
                {
                    "sha256": "sha-conflict", "truth_classification": "conflict", "manual_conflict": True,
                    "proposed_canonical_id": None, "proposed_redundant_image_asset_ids": None,
                    "members": [
                        {"image_asset_id": conflict_a.id, "truth_species": "青鱼", "truth_status": "LIKELY_CORRECT"},
                        {"image_asset_id": conflict_b.id, "truth_species": "草鱼", "truth_status": "LIKELY_CORRECT"},
                    ],
                },
            ],
        },
    }


def test_phase_b_quarantines_redundant_and_whole_conflict_group_without_review_mutation(db):
    authority = _authority(db)
    before_truth = {row.id: (row.truth_species, row.truth_status, row.review_status) for row in db.scalars(select(ImageAsset)).all()}
    result = execute_phase_b(db, authority, Path("var/phase-b-test-evidence"), mode="apply")
    assert result["status"] == "COMPLETE"
    assert result["apply"]["newly_excluded"] == 3
    assert result["after"]["training_eligible"] == 1
    assert result["after"]["eligible_exact_duplicate_groups"] == 0
    assert result["after"]["eligible_truth_conflict_members"] == 0
    for row in db.scalars(select(ImageAsset)).all():
        assert (row.truth_species, row.truth_status, row.review_status) == before_truth[row.id]
    canonical = db.scalar(select(ImageAsset).where(ImageAsset.image_id == "canonical"))
    redundant = db.scalar(select(ImageAsset).where(ImageAsset.image_id == "redundant"))
    assert canonical.training_eligible is True
    assert redundant.training_eligible is False
    assert redundant.training_exclusion_reason == "GLOBAL_EXACT_DUPLICATE"
    assert redundant.duplicate_of_image_asset_id == canonical.id
    assert all(row.training_exclusion_reason == "GLOBAL_EXACT_TRUTH_CONFLICT" for row in db.scalars(select(ImageAsset).where(ImageAsset.image_id.like("conflict-%"))).all())


def test_phase_b_second_apply_is_idempotent(db):
    authority = _authority(db)
    _apply(db, authority)
    second = _apply(db, authority)
    assert second["newly_excluded"] == 0
    assert second["canonical_pointer_change_count"] == 0
    assert second["truth_conflict_rows_changed"] == 0
    assert _post_audit(db, authority)["post_apply_gate"] is True


def test_phase_b_rolls_back_on_partial_failure(db):
    authority = _authority(db)
    with pytest.raises(RuntimeError, match="injected Phase B transaction failure"):
        _apply(db, authority, fail_after=1)
    assert _counts(db)["training_eligible"] == 4
    assert not db.scalar(select(ImageAsset).where(ImageAsset.training_exclusion_reason.is_not(None)))


def test_phase_a_drift_blocks_before_mutation(db):
    authority = _authority(db)
    image = db.scalar(select(ImageAsset).where(ImageAsset.image_id == "redundant"))
    image.truth_species = "鲤鱼"
    db.commit()
    with pytest.raises(PhaseAPlanDrift):
        _apply(db, authority)
    assert _counts(db)["training_eligible"] == 4


def test_accepted_pool_source_requires_training_eligibility(db):
    authority = _authority(db)
    _apply(db, authority)
    for row in db.scalars(select(ImageAsset).where(ImageAsset.image_id.in_(["canonical", "redundant"]))).all():
        db.add(
            BatchCropReview(
                batch_id=row.batch_id, image_asset_id=row.id, image_id=row.image_id,
                accepted_bbox_json="[0.1,0.1,0.5,0.5]", species_name=row.truth_species, status="ACCEPTED",
            )
        )
    db.commit()
    rows, _invalid = _source_rows(db)
    assert [row["image_id"] for row in rows] == ["canonical"]


def test_crop_dataset_source_requires_training_eligibility(db):
    authority = _authority(db)
    _apply(db, authority)
    for row in db.scalars(select(ImageAsset).where(ImageAsset.image_id.in_(["canonical", "redundant"]))).all():
        db.add(
            BatchCropReview(
                batch_id=row.batch_id, image_asset_id=row.id, image_id=row.image_id,
                accepted_bbox_json="[0.1,0.1,0.5,0.5]", species_name=row.truth_species, status="ACCEPTED",
            )
        )
    db.commit()
    rows = _accepted_pool_rows(db)
    assert [row[0].image_id for row in rows] == ["canonical"]


def test_freeze_gate_detects_reenabled_exact_duplicate(db):
    authority = _authority(db)
    redundant = db.scalar(select(ImageAsset).where(ImageAsset.image_id == "redundant"))
    redundant.training_exclusion_reason = "GLOBAL_EXACT_DUPLICATE"
    db.commit()
    with pytest.raises(ValueError, match="Training eligibility gate failed"):
        _training_eligibility_gate(db)


class _FakeBlob:
    def __init__(self, name, uploaded):
        self.name = name
        self.uploaded = uploaded

    def upload_from_filename(self, filename):
        self.uploaded.append((self.name, Path(filename).read_text(encoding="utf-8")))

    def exists(self, _client):
        return any(name == self.name for name, _content in self.uploaded)


class _FakeBucket:
    def __init__(self, uploaded):
        self.uploaded = uploaded

    def blob(self, name):
        return _FakeBlob(name, self.uploaded)


class _FakeStorageClient:
    def __init__(self):
        self.uploaded = []

    def bucket(self, _name):
        return _FakeBucket(self.uploaded)


def test_success_evidence_upload_is_verified(tmp_path):
    (tmp_path / "execution_metadata.json").write_text("{}", encoding="utf-8")
    client = _FakeStorageClient()
    result = _upload_evidence(tmp_path, "gs://bucket/phase-b", storage_client_factory=lambda: client)
    assert result["status"] == "PASS"
    assert result["uploaded"] == ["execution_metadata.json"]
    assert client.uploaded == [("phase-b/execution_metadata.json", "{}")]


def test_failure_before_db_session_writes_durable_failure_evidence(tmp_path, monkeypatch):
    def fail_load(_root):
        raise RuntimeError("load failed password=not-a-real-password")

    monkeypatch.setattr("scripts.historical_exact_duplicate_phase_b.load_phase_a_artifact", fail_load)
    client = _FakeStorageClient()
    monkeypatch.setattr("google.cloud.storage.Client", lambda: client)
    monkeypatch.setenv("PHASE_B_OUTPUT_GCS_PREFIX", "gs://bucket/failed-phase-b")
    assert main(["--mode", "dry-run", "--phase-a-dir", str(tmp_path), "--output-dir", str(tmp_path)]) == 1
    metadata = json.loads((tmp_path / "execution_metadata.json").read_text(encoding="utf-8"))
    failure = json.loads((tmp_path / "failure.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "FAILED"
    assert metadata["stage"] == "LOAD_PHASE_A"
    assert metadata["exception_type"] == "RuntimeError"
    assert "not-a-real-password" not in failure["traceback"]
    uploaded_names = {name.rsplit("/", 1)[-1] for name, _content in client.uploaded}
    assert {"execution_metadata.json", "failure.json"} <= uploaded_names


def test_schema_contract_failure_is_reported_before_db_session(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.historical_exact_duplicate_phase_b.load_phase_a_artifact", lambda _root: {"run_id": "r", "audit_sha": "a", "cleanup_plan_sha256": "p"})
    monkeypatch.setattr("scripts.historical_exact_duplicate_phase_b._schema_snapshot", lambda: {"status": "PASS", "table_exists": True, "columns": [], "indexes": [], "foreign_keys": []})
    monkeypatch.setattr("scripts.historical_exact_duplicate_phase_b._ensure_training_eligibility_columns", lambda: (_ for _ in ()).throw(RuntimeError("ALTER TABLE denied")))
    assert main(["--mode", "dry-run", "--phase-a-dir", str(tmp_path), "--output-dir", str(tmp_path)]) == 1
    metadata = json.loads((tmp_path / "execution_metadata.json").read_text(encoding="utf-8"))
    assert metadata["stage"] == "SCHEMA_CONTRACT"
    assert metadata["error_message"] == "ALTER TABLE denied"


def test_live_drift_failure_is_reported_with_stage(tmp_path, db, monkeypatch):
    authority = _authority(db)
    monkeypatch.setattr("scripts.historical_exact_duplicate_phase_b.load_phase_a_artifact", lambda _root: authority)
    monkeypatch.setattr("scripts.historical_exact_duplicate_phase_b._schema_snapshot", lambda: {"status": "PASS", "table_exists": True, "columns": [], "indexes": [], "foreign_keys": []})
    monkeypatch.setattr("scripts.historical_exact_duplicate_phase_b._ensure_training_eligibility_columns", lambda: None)
    monkeypatch.setattr("scripts.historical_exact_duplicate_phase_b.SessionLocal", lambda: db)
    monkeypatch.setattr("scripts.historical_exact_duplicate_phase_b._live_drift_gate", lambda _db, _authority: (_ for _ in ()).throw(RuntimeError("live drift")))
    monkeypatch.setattr("scripts.historical_exact_duplicate_phase_b._upload_evidence", lambda output, prefix: {"status": "PASS"})
    assert main(["--mode", "dry-run", "--phase-a-dir", str(tmp_path), "--output-dir", str(tmp_path)]) == 1
    metadata = json.loads((tmp_path / "execution_metadata.json").read_text(encoding="utf-8"))
    assert metadata["stage"] == "LIVE_DRIFT_GATE"
    assert metadata["error_message"] == "live drift"


def test_dry_run_never_calls_apply_or_pool_sync(tmp_path, db, monkeypatch):
    authority = _authority(db)
    monkeypatch.setattr("scripts.historical_exact_duplicate_phase_b._apply", lambda *args, **kwargs: pytest.fail("dry-run called _apply"))
    monkeypatch.setattr("scripts.historical_exact_duplicate_phase_b._run_pool_sync", lambda *args, **kwargs: pytest.fail("dry-run called pool sync"))
    result = execute_phase_b(db, authority, tmp_path, mode="dry-run")
    assert result["status"] == "DRY_RUN"
    assert json.loads((tmp_path / "phase_b_apply.json").read_text(encoding="utf-8"))["status"] == "NOT_EXECUTED"
    assert json.loads((tmp_path / "accepted_pool_post_sync.json").read_text(encoding="utf-8"))["status"] == "NOT_EXECUTED"


def test_workflow_captures_execution_id_before_polling():
    workflow = Path(".github/workflows/historical-exact-duplicate-phase-b-production.yml").read_text(encoding="utf-8")
    execute_start = workflow.index("id: execute")
    diagnostics_start = workflow.index("name: Persist execution diagnostics")
    execute_block = workflow[execute_start:diagnostics_start]
    assert "--async" in execute_block
    assert "--format='value(metadata.name)'" in execute_block
    assert "EXECUTION=$" in execute_block
    assert "executions describe \"$EXECUTION\"" in execute_block
    assert "gcloud run jobs execute \"$JOB_NAME\" --project \"$PROJECT_ID\" --region \"$REGION\" --update-env-vars=\"PHASE_B_MODE=${MODE}\" --wait" not in execute_block


def test_direct_script_import_bootstrap():
    repository_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "scripts/historical_exact_duplicate_phase_b.py", "--help"],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
