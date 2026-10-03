from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from app import models  # noqa: F401
from app.platform import models as _platform_models  # noqa: F401
from app.bulk_review import BulkReviewApply, BulkReviewItem, api_bulk_apply
from app.db import Base
from app.historical_duplicate_closure import (
    HistoricalDuplicateClosureWriteFenceLocked,
    PROTECTED_ROUTE_PATHS,
)
from app.main import (
    BatchSync,
    DatasetFreeze,
    FeedbackCreate,
    FeedbackMaterialize,
    ReviewUpdate,
    api_materialize_feedback,
    api_record_feedback,
    batch_sync,
    dataset_freeze,
    update_review,
)
from app.models import Batch, FeedbackEvent, ImageAsset, SpeciesCatalog
from app.p0_automation import maybe_auto_materialize_feedback


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'closure-fence.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = factory()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture(autouse=True)
def active_fence(monkeypatch):
    monkeypatch.setenv("HISTORICAL_DUPLICATE_CLOSURE_WRITE_FENCE", "true")


def _feedback(event_id: str, feedback_type: str, corrected: str | None = None) -> FeedbackCreate:
    return FeedbackCreate(
        source_event_id=event_id,
        feedback_type=feedback_type,
        source="console_inference_test",
        image_gcs_uri="gs://test-bucket/inference/image.jpg",
        model_version="MODEL_M1_v0.2",
        predicted_species="草鱼",
        confidence=0.9,
        corrected_species=corrected,
    )


def test_fence_active_confirmed_feedback_is_allowed(db: Session):
    result = api_record_feedback(_feedback("closure-confirmed-001", "confirmed"), db)

    assert result["feedback_type"] == "confirmed"
    assert db.scalar(select(func.count()).select_from(FeedbackEvent)) == 1


def test_fence_active_known_corrected_species_is_allowed(db: Session):
    db.add(SpeciesCatalog(species_key="common_carp", catalog_order=3, common_name_zh="鲤鱼", status="active"))
    db.commit()

    result = api_record_feedback(_feedback("closure-corrected-001", "corrected", "鲤鱼"), db)

    assert result["feedback_type"] == "corrected"
    assert result["corrected_species"] == "鲤鱼"
    assert db.scalar(select(func.count()).select_from(SpeciesCatalog).where(SpeciesCatalog.common_name_zh == "鲤鱼")) == 1


def test_fence_active_unknown_feedback_is_allowed(db: Session):
    result = api_record_feedback(_feedback("closure-unknown-001", "unknown"), db)

    assert result["feedback_type"] == "unknown"
    assert db.scalar(select(func.count()).select_from(FeedbackEvent)) == 1


def test_fence_active_unknown_corrected_species_is_blocked_without_catalog_creation(db: Session):
    before_events = db.scalar(select(func.count()).select_from(FeedbackEvent)) or 0
    before_candidates = db.scalar(
        select(func.count()).select_from(SpeciesCatalog).where(SpeciesCatalog.status == "candidate")
    ) or 0

    with pytest.raises(HistoricalDuplicateClosureWriteFenceLocked) as caught:
        api_record_feedback(_feedback("closure-new-species-001", "corrected", "闭环测试鱼"), db)

    assert caught.value.code == "HISTORICAL_DUPLICATE_CLOSURE_IN_PROGRESS"
    assert caught.value.message == "Historical duplicate cleanup maintenance window is active."
    assert (db.scalar(select(func.count()).select_from(FeedbackEvent)) or 0) == before_events
    assert (
        db.scalar(select(func.count()).select_from(SpeciesCatalog).where(SpeciesCatalog.status == "candidate")) or 0
    ) == before_candidates


def test_fence_active_feedback_materialize_is_blocked(db: Session):
    with pytest.raises(HistoricalDuplicateClosureWriteFenceLocked):
        api_materialize_feedback(FeedbackMaterialize(batch_id="BATCH_CLOSURE_001"), db)


def test_feedback_event_does_not_change_image_asset_authority_fingerprint(db: Session):
    batch = Batch(
        batch_id="BATCH_CLOSURE_001",
        source="test",
        image_count=1,
        manifest_uri="gs://test-bucket/manifest.csv",
        raw_uri="gs://test-bucket/raw",
        status="INGESTED",
    )
    image = ImageAsset(
        batch_id=batch.batch_id,
        image_id="IMAGE_CLOSURE_001",
        file_name="images/image.jpg",
        object_name="incoming/BATCH_CLOSURE_001/images/image.jpg",
        gcs_uri="gs://test-bucket/incoming/BATCH_CLOSURE_001/images/image.jpg",
        claimed_species="草鱼",
        truth_species="草鱼",
        truth_status="LIKELY_CORRECT",
        review_status="approved",
        notes="frozen-authority-fixture",
        created_at=datetime.now(timezone.utc),
    )
    db.add_all([batch, image])
    db.commit()
    db.refresh(image)
    before = (image.id, image.batch_id, image.image_id, image.truth_species, image.truth_status, image.review_status, image.notes, image.updated_at)

    api_record_feedback(_feedback("closure-fingerprint-001", "confirmed"), db)

    db.expire_all()
    after_image = db.get(ImageAsset, image.id)
    after = (
        after_image.id,
        after_image.batch_id,
        after_image.image_id,
        after_image.truth_species,
        after_image.truth_status,
        after_image.review_status,
        after_image.notes,
        after_image.updated_at,
    )
    assert after == before
    assert db.scalar(select(func.count()).select_from(FeedbackEvent)) == 1


def test_fence_blocks_review_ingestion_and_dataset_freeze_writes(db: Session):
    with pytest.raises(HistoricalDuplicateClosureWriteFenceLocked):
        update_review("BATCH_CLOSURE_001", "IMAGE_CLOSURE_001", ReviewUpdate(truth_species="鲤鱼"), db)
    with pytest.raises(HistoricalDuplicateClosureWriteFenceLocked):
        api_bulk_apply(
            BulkReviewApply(
                batch_id="BATCH_CLOSURE_001",
                items=[BulkReviewItem(image_id="IMAGE_CLOSURE_001", review_status="approved")],
            ),
            db,
        )
    with pytest.raises(HistoricalDuplicateClosureWriteFenceLocked):
        batch_sync(BatchSync(batch_id="BATCH_CLOSURE_001"), db)
    with pytest.raises(HistoricalDuplicateClosureWriteFenceLocked):
        dataset_freeze(DatasetFreeze(dataset_version="DS_CLOSURE_001"), db)


def test_feedback_materialization_automation_is_suppressed_by_fence(db: Session):
    result = maybe_auto_materialize_feedback(db)

    assert result["triggered"] is False
    assert result["blocked_by_closure"] is True


def test_protected_routes_exclude_plain_feedback_and_model_prediction():
    assert "/api/feedback" not in PROTECTED_ROUTE_PATHS
    assert "/api/feedback/materialize" in PROTECTED_ROUTE_PATHS
    assert "/api/inference/predict" not in PROTECTED_ROUTE_PATHS
    assert "/api/dataset-accepted-bbox/{batch_id}/{image_id}" in PROTECTED_ROUTE_PATHS
    assert "/api/platform/review/species" in PROTECTED_ROUTE_PATHS


def test_structured_frontend_error_message_is_used():
    template = open("app/templates/inference.html", encoding="utf-8").read()

    assert "detail&&typeof detail==='object'?detail.message" in template
    assert "data?.message" in template
    assert "data?.code" in template
    assert "`HTTP ${status}`" in template


def test_fence_disabled_preserves_unknown_species_feedback_behavior(db: Session, monkeypatch):
    monkeypatch.setenv("HISTORICAL_DUPLICATE_CLOSURE_WRITE_FENCE", "false")

    result = api_record_feedback(_feedback("closure-disabled-new-species-001", "corrected", "关闭围栏测试鱼"), db)

    assert result["feedback_type"] == "new_species_candidate"
    assert result["corrected_species"] == "关闭围栏测试鱼"
    candidate = db.scalar(select(SpeciesCatalog).where(SpeciesCatalog.common_name_zh == "关闭围栏测试鱼"))
    assert candidate is not None
    assert candidate.status == "candidate"
