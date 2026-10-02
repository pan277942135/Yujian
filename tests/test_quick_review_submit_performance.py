from __future__ import annotations

from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session
from fastapi import HTTPException

from app.bulk_review import BulkReviewApply, BulkReviewItem, api_bulk_apply, api_bulk_images
from app.dedupe import ImageFingerprint
from app.models import (
    AcceptedPoolSyncRequest,
    Batch,
    BatchCropReview,
    FeedbackEvent,
    ImageAsset,
    ReviewEvent,
    SpeciesCatalog,
)
from app.presence import FishPresenceResult


TABLES = [
    Batch,
    SpeciesCatalog,
    ImageAsset,
    BatchCropReview,
    FeedbackEvent,
    FishPresenceResult,
    ImageFingerprint,
    ReviewEvent,
    AcceptedPoolSyncRequest,
]


def _session():
    engine = create_engine("sqlite:///:memory:")
    for model in TABLES:
        model.__table__.create(engine)
    return engine, Session(engine)


def _seed(db: Session, *, batches: int = 1, images_per_batch: int = 30):
    db.add(SpeciesCatalog(species_key="grass", catalog_order=1, common_name_zh="草鱼", status="active"))
    for batch_number in range(batches):
        batch_id = f"batch-{batch_number}"
        db.add(Batch(batch_id=batch_id, source="test", manifest_uri="m", raw_uri="r"))
        for image_number in range(images_per_batch):
            image_id = f"image-{batch_number}-{image_number}"
            image = ImageAsset(
                batch_id=batch_id,
                image_id=image_id,
                file_name=f"{image_id}.jpg",
                object_name=image_id,
                gcs_uri=f"gs://test/{image_id}",
                claimed_species="草鱼",
            )
            db.add(image)
            db.flush()
            db.add(
                BatchCropReview(
                    batch_id=batch_id,
                    image_asset_id=image.id,
                    image_id=image_id,
                    candidate_bbox_json="[0,0,1,1]",
                )
            )
    db.commit()


def test_images_use_real_cross_batch_page_and_presence_filter():
    _engine, db = _session()
    try:
        _seed(db, batches=2, images_per_batch=15)
        page = api_bulk_images(
            batch_id=None,
            species="草鱼",
            status="pending",
            presence="not_scanned",
            limit=24,
            offset=0,
            db=db,
        )
        assert page["total"] == 30
        assert len(page["items"]) == 24
        assert {item["batch_id"] for item in page["items"]} == {"batch-0", "batch-1"}
    finally:
        db.close()


def test_bulk_apply_prefetch_query_count_does_not_scale_with_items():
    counts = []
    for item_count in (1, 24):
        engine, db = _session()
        _seed(db, images_per_batch=max(item_count, 24))
        statements = []

        def capture(_conn, _cursor, statement, _parameters, _context, _executemany):
            if statement.lstrip().upper().startswith("SELECT"):
                statements.append(statement)

        event.listen(engine, "before_cursor_execute", capture)
        try:
            result = api_bulk_apply(
                BulkReviewApply(
                    batch_id="batch-0",
                    items=[
                        BulkReviewItem(
                            image_id=f"image-0-{index}",
                            review_status="approved",
                            truth_species="草鱼",
                            accepted_bbox=[0, 0, 1, 1],
                        )
                        for index in range(item_count)
                    ],
                ),
                db,
            )
            assert result["updated"] == item_count
            counts.append(len(statements))
        finally:
            db.close()
    assert counts[0] == counts[1]
    assert counts[1] <= 6


def test_accepted_pool_enqueue_is_only_a_durable_signal(monkeypatch):
    from app.accepted_pool import enqueue_accepted_pool_sync

    _engine, db = _session()
    try:
        def heavy_scan_should_not_run(_db):
            raise AssertionError("Accepted Pool scan ran in the review request")

        monkeypatch.setattr("app.accepted_pool.start_accepted_pool_sync", heavy_scan_should_not_run)
        signal = enqueue_accepted_pool_sync(db)
        db.commit()
        assert signal["status"] == "PENDING"
        assert db.scalar(select(AcceptedPoolSyncRequest.status)) == "PENDING"
    finally:
        db.close()


def test_bulk_apply_preserves_validation_and_mixed_status_semantics():
    _engine, db = _session()
    try:
        _seed(db, images_per_batch=3)
        invalid_cases = [
            BulkReviewItem(image_id="image-0-0", review_status="approved"),
            BulkReviewItem(image_id="image-0-0", review_status="approved", truth_species="不存在", accepted_bbox=[0, 0, 1, 1]),
            BulkReviewItem(image_id="image-0-0", review_status="approved", truth_species="草鱼", accepted_bbox=[0, 0, 0, 0]),
        ]
        for item in invalid_cases:
            try:
                api_bulk_apply(BulkReviewApply(batch_id="batch-0", items=[item]), db)
            except HTTPException:
                db.rollback()
            else:
                raise AssertionError("invalid bulk review item was accepted")

        result = api_bulk_apply(
            BulkReviewApply(
                batch_id="batch-0",
                items=[
                    BulkReviewItem(image_id="image-0-0", review_status="approved", truth_species="草鱼", accepted_bbox=[0, 0, 1, 1]),
                    BulkReviewItem(image_id="image-0-1", review_status="rejected", truth_species="草鱼"),
                    BulkReviewItem(image_id="image-0-2", review_status="pending", truth_species="草鱼"),
                ],
            ),
            db,
        )
        assert result["updated"] == 3
        rows = db.scalars(select(ImageAsset).order_by(ImageAsset.image_id)).all()
        assert [row.review_status for row in rows] == ["approved", "rejected", "pending"]
        assert rows[0].truth_status == "LIKELY_CORRECT"
        assert db.scalar(select(ReviewEvent.id)) is not None
    finally:
        db.close()


def test_bulk_review_template_has_one_reload_path_and_no_fetch_all_architecture():
    source = open("app/templates/bulk_review.html", encoding="utf-8").read()
    assert "fetchAllImages" not in source
    assert "allImageCache" not in source
    assert "await loadSpecies();if(speciesRows.length&&currentSpecies)await loadImages()" not in source
    assert "await loadSpecies()" in source
    assert "提交中…" in source


def test_batches_endpoint_uses_bounded_aggregation_queries():
    from app.main import batches

    engine, db = _session()
    _seed(db, batches=2, images_per_batch=3)
    statements = []

    def capture(_conn, _cursor, statement, _parameters, _context, _executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    event.listen(engine, "before_cursor_execute", capture)
    try:
        result = batches(db)
        assert len(result) == 2
        assert all(item["image_count"] == 3 for item in result)
        assert len(statements) == 2
    finally:
        db.close()
