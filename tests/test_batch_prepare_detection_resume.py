import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import app.dedupe as dedupe
import app.presence as presence
from app.db import Base
from app.dedupe import ImageFingerprint
from app.models import Batch, ImageAsset
from app.presence import FishPresenceResult


class ScanBlob:
    def download_as_bytes(self, **_kwargs):
        return b"resume-safe-image"


class ScanBucket:
    def blob(self, _name):
        return ScanBlob()


class ScanStorageClient:
    def bucket(self, _name):
        return ScanBucket()


@pytest.fixture
def scan_db():
    import app.platform.models  # noqa: F401

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def _seed_image(db, batch_id: str):
    object_name = f"raw/batches/{batch_id}/images/fish.jpg"
    db.add(
        Batch(
            batch_id=batch_id,
            source="test",
            image_count=1,
            manifest_uri=f"gs://test-bucket/raw/batches/{batch_id}/metadata/fish_manifest.csv",
            raw_uri=f"gs://test-bucket/raw/batches/{batch_id}/",
            status="REGISTERED",
        )
    )
    db.flush()
    db.add(
        ImageAsset(
            batch_id=batch_id,
            image_id=f"{batch_id}_001",
            file_name="fish.jpg",
            object_name=object_name,
            gcs_uri=f"gs://test-bucket/{object_name}",
            review_status="pending",
            truth_status="UNCERTAIN",
        )
    )
    db.commit()


def test_prepare_resume_does_not_duplicate_detection_results(scan_db, monkeypatch):
    monkeypatch.setattr(dedupe.storage, "Client", lambda: ScanStorageClient())
    monkeypatch.setattr(presence.storage, "Client", lambda: ScanStorageClient())
    monkeypatch.setattr(dedupe, "get_bucket_name", lambda: "test-bucket")
    monkeypatch.setattr(presence, "get_bucket_name", lambda: "test-bucket")
    monkeypatch.setattr(
        dedupe,
        "fingerprint_bytes",
        lambda _content: {
            "sha256": "a" * 64,
            "phashes": ["0" * 16],
            "dhash": "0" * 16,
            "crop_hash": "",
            "histogram": [0.25, 0.25, 0.25, 0.25],
            "width": 100,
            "height": 100,
        },
    )
    monkeypatch.setattr(presence.vision, "ImageAnnotatorClient", lambda: object())
    monkeypatch.setattr(
        presence,
        "_vision_evidence",
        lambda _client, _content: {
            "status": "single_fish",
            "fish_score": 0.9,
            "fish_count": 1,
            "max_box_area_ratio": 0.3,
        },
    )

    batch_id = "BATCH_PREPARE_DETECTION_RESUME_001"
    with scan_db() as db:
        _seed_image(db, batch_id)
        first_dedupe = dedupe.scan_batch(db, batch_id, limit=100, rescan=False)
        second_dedupe = dedupe.scan_batch(db, batch_id, limit=100, rescan=False)
        first_presence = presence.scan_batch(db, batch_id, limit=40, rescan=False)
        second_presence = presence.scan_batch(db, batch_id, limit=40, rescan=False)

        fingerprint_rows = db.scalars(
            select(ImageFingerprint).where(ImageFingerprint.batch_id == batch_id)
        ).all()
        presence_rows = db.scalars(
            select(FishPresenceResult).where(FishPresenceResult.batch_id == batch_id)
        ).all()

    assert first_dedupe["processed"] == 1
    assert second_dedupe["processed"] == 0
    assert first_presence["vision_processed"] == 1
    assert second_presence["vision_processed"] == 0
    assert len(fingerprint_rows) == 1
    assert len(presence_rows) == 1
