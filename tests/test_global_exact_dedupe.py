from datetime import datetime, timezone

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.dedupe import ImageFingerprint
from app.exact_dedupe import bootstrap_global_registry, claim_global_image
from app.models import (
    Batch,
    GlobalDuplicateAudit,
    GlobalImageContent,
    GlobalImageDuplicateMember,
    ImageAsset,
)


def _session():
    import app.platform.models  # noqa: F401

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_global_claim_is_exact_and_resume_safe():
    db = _session()
    try:
        digest = "a" * 64
        first = claim_global_image(
            db,
            sha256=digest,
            batch_id="BATCH_A",
            incoming_path="images/a.jpg",
            object_name="incoming/BATCH_A/images/a.jpg",
            source="manual",
        )
        assert first.status == "CLAIMED"
        db.commit()

        retry = claim_global_image(
            db,
            sha256=digest,
            batch_id="BATCH_A",
            incoming_path="images/a.jpg",
            object_name="incoming/BATCH_A/images/a.jpg",
            source="manual",
        )
        assert retry.status == "SKIP"

        different_content = claim_global_image(
            db,
            sha256="b" * 64,
            batch_id="BATCH_B",
            incoming_path="images/b.jpg",
            object_name="incoming/BATCH_B/images/b.jpg",
            source="manual",
        )
        assert different_content.status == "CLAIMED"

        duplicate = claim_global_image(
            db,
            sha256=digest,
            batch_id="BATCH_B",
            incoming_path="images/renamed.jpg",
            object_name="incoming/BATCH_B/images/renamed.jpg",
            source="zip",
        )
        assert duplicate.status == "DUPLICATE_BLOCKED"
        db.commit()

        audit = db.scalar(select(GlobalDuplicateAudit).where(GlobalDuplicateAudit.sha256 == digest))
        assert audit is not None
        assert audit.reason == "GLOBAL_EXACT_DUPLICATE"
        assert audit.incoming_batch_id == "BATCH_B"
        assert db.scalar(select(GlobalImageContent).where(GlobalImageContent.sha256 == digest)) is not None
    finally:
        db.close()


def test_bootstrap_reuses_fingerprints_and_records_historical_members():
    db = _session()
    try:
        created = datetime(2026, 1, 1, tzinfo=timezone.utc)
        db.add_all(
            [
                Batch(batch_id="BATCH_1", source="legacy", manifest_uri="m1", raw_uri="r1", image_count=1),
                Batch(batch_id="BATCH_2", source="legacy", manifest_uri="m2", raw_uri="r2", image_count=1),
            ]
        )
        db.flush()
        first = ImageAsset(
            batch_id="BATCH_1",
            image_id="A",
            file_name="a.jpg",
            object_name="raw/a.jpg",
            gcs_uri="gs://bucket/raw/a.jpg",
            created_at=created,
        )
        second = ImageAsset(
            batch_id="BATCH_2",
            image_id="B",
            file_name="b.jpg",
            object_name="raw/b.jpg",
            gcs_uri="gs://bucket/raw/b.jpg",
            created_at=created,
        )
        db.add_all([first, second])
        db.flush()
        for image in (first, second):
            db.add(
                ImageFingerprint(
                    image_asset_id=image.id,
                    batch_id=image.batch_id,
                    sha256="c" * 64,
                    phash_json="{}",
                    dhash="d" * 64,
                    crop_hash="e" * 64,
                    histogram_json="[]",
                    width=1,
                    height=1,
                    fingerprint_version="test",
                    created_at=created,
                    updated_at=created,
                )
            )
        db.commit()

        result = bootstrap_global_registry(db, bucket_name="bucket")
        assert result["coverage_complete"] is True
        assert result["created"] == 1
        assert result["historical_duplicate_members"] == 1
        registry = db.scalar(select(GlobalImageContent).where(GlobalImageContent.sha256 == "c" * 64))
        assert registry is not None
        assert registry.canonical_image_asset_id == first.id
        assert db.scalar(select(GlobalImageDuplicateMember).where(GlobalImageDuplicateMember.sha256 == "c" * 64)) is not None
        assert len(db.scalars(select(GlobalImageDuplicateMember)).all()) == 2
    finally:
        db.close()
