from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.accepted_pool import (
    ACCEPTED_POOL_MANIFEST_NAME,
    accepted_pool_freeze_preview,
    accepted_pool_summary,
    freeze_accepted_pool_dataset,
    start_accepted_pool_sync,
    step_accepted_pool_job,
)
from app.db import Base
from app.models import Batch, BatchCropReview, DatasetVersion, ImageAsset
from app.platform.services import crop_dataset


class MemoryBlob:
    def __init__(self, bucket: "MemoryBucket", name: str):
        self.bucket = bucket
        self.name = name
        self.data: bytes | None = None

    def exists(self, _client=None):
        return self.data is not None

    def download_as_text(self, encoding="utf-8"):
        return (self.data or b"").decode(encoding)

    def download_as_bytes(self, **_kwargs):
        return self.data or b""

    def upload_from_string(self, data, **_kwargs):
        self.data = data.encode("utf-8") if isinstance(data, str) else bytes(data)

    def copy_to(self, bucket: "MemoryBucket", destination: str):
        bucket.blob(destination).data = self.data


class MemoryBucket:
    def __init__(self):
        self._blobs: dict[str, MemoryBlob] = {}

    def blob(self, name: str):
        return self._blobs.setdefault(name, MemoryBlob(self, name))

    def list_blobs(self, prefix: str = ""):
        return [blob for name, blob in self._blobs.items() if name.startswith(prefix) and blob.data is not None]


class MemoryClient:
    def __init__(self, bucket: MemoryBucket):
        self.bucket_obj = bucket

    def bucket(self, _name: str):
        return self.bucket_obj


def _image_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (120, 80), (35, 100, 160)).save(output, format="JPEG")
    return output.getvalue()


def _session(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'accepted-pool-incremental.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def _add_accepted(db, bucket: MemoryBucket, index: int, *, bbox=None):
    batch_id = f"BATCH_ACCEPTED_{index:03d}"
    image_id = f"accepted-{index:03d}"
    db.add(Batch(batch_id=batch_id, source="upload", manifest_uri="gs://pool/manifest.csv", raw_uri="gs://pool/raw"))
    image = ImageAsset(
        batch_id=batch_id,
        image_id=image_id,
        file_name=f"{image_id}.jpg",
        object_name=f"source/{image_id}.jpg",
        gcs_uri=f"gs://pool/source/{image_id}.jpg",
        truth_species="草鱼",
        review_status="approved",
    )
    db.add(image)
    db.flush()
    db.add(
        BatchCropReview(
            batch_id=batch_id,
            image_asset_id=image.id,
            image_id=image_id,
            accepted_bbox_json=json.dumps(bbox or [0.15, 0.10, 0.50, 0.60]),
            species_name="草鱼",
            status="ACCEPTED",
        )
    )
    bucket.blob(f"source/{image_id}.jpg").data = _image_bytes()


def _run_to_terminal(job):
    for _ in range(100):
        if job["status"] in {"SUCCESS", "FAILED"}:
            return job
        job = step_accepted_pool_job(job["job_id"])
    raise AssertionError(f"Accepted Pool job did not finish: {job}")


def test_accepted_pool_is_persistent_incremental_and_freeze_is_explicit(monkeypatch, tmp_path: Path):
    bucket = MemoryBucket()
    client = MemoryClient(bucket)
    Session = _session(tmp_path)
    monkeypatch.setattr(crop_dataset, "_storage", lambda: (client, bucket))
    monkeypatch.setattr(crop_dataset, "get_bucket_name", lambda: "pool")
    monkeypatch.setattr("app.accepted_pool.SessionLocal", Session)
    monkeypatch.setattr("app.accepted_pool._jobs", {})
    monkeypatch.setattr("app.accepted_pool._job_locks", {})

    crop_calls: list[list[float]] = []
    original_letterbox = crop_dataset._letterbox

    def counted_letterbox(data, box, *, crop_scale):
        crop_calls.append(list(box))
        return original_letterbox(data, box, crop_scale=crop_scale)

    monkeypatch.setattr(crop_dataset, "_letterbox", counted_letterbox)
    monkeypatch.setattr(
        crop_dataset,
        "_accepted_pool_detector_bbox",
        lambda _data: (_ for _ in ()).throw(AssertionError("Accepted Pool sync must not rerun Detector")),
    )

    db = Session()
    try:
        for index in range(3):
            _add_accepted(db, bucket, index)
        db.commit()

        first = _run_to_terminal(start_accepted_pool_sync(db))
        assert first["status"] == "SUCCESS", first
        assert first["source_count"] == 3
        assert first["added_count"] == 3
        assert first["reused_count"] == 0
        assert first["bbox_generated"] == 3
        assert first["crop_generated"] == 3
        assert first["dataset_count"] == 3
        assert db.get(DatasetVersion, "DS_CROP_M1_v0.2") is None
        assert not any(name.startswith("datasets/DS_CROP_M1_v0.2/") for name in bucket._blobs)

        pool_rows = list(csv.DictReader(io.StringIO(bucket.blob(ACCEPTED_POOL_MANIFEST_NAME).download_as_text())))
        assert len(pool_rows) == 3
        assert all(row["bbox_source"] == "accepted_bbox" for row in pool_rows)
        assert all(row["detector_bbox"] == "" for row in pool_rows)
        old_crop_objects = {
            row["pool_key"]: bucket.blob(f"datasets/accepted_pool/{row['crop_path']}").data
            for row in pool_rows
        }
        assert accepted_pool_summary(db)["pending_count"] == 0

        no_op = _run_to_terminal(start_accepted_pool_sync(db))
        assert no_op["status"] == "SUCCESS", no_op
        assert no_op["pending_count"] == 0
        assert no_op["added_count"] == 0
        assert no_op["updated_count"] == 0
        assert no_op["reused_count"] == 3
        assert len(crop_calls) == 3

        _add_accepted(db, bucket, 3)
        db.commit()
        second = _run_to_terminal(start_accepted_pool_sync(db))
        assert second["status"] == "SUCCESS", second
        assert second["source_count"] == 4
        assert second["pending_count"] == 1
        assert second["added_count"] == 1
        assert second["reused_count"] == 3
        assert second["bbox_generated"] == 1
        assert second["crop_generated"] == 1
        assert len(crop_calls) == 4

        pool_rows = list(csv.DictReader(io.StringIO(bucket.blob(ACCEPTED_POOL_MANIFEST_NAME).download_as_text())))
        assert len(pool_rows) == 4
        for key, data in old_crop_objects.items():
            row = next(item for item in pool_rows if item["pool_key"] == key)
            assert bucket.blob(f"datasets/accepted_pool/{row['crop_path']}").data == data

        preview = accepted_pool_freeze_preview(db, dataset_version="DS_CROP_M1_v0.2")
        assert preview["freeze_ready"] is True
        assert preview["image_count"] == 4
        assert preview["source_mode"] == "ACCEPTED_POOL"
        assert db.get(DatasetVersion, "DS_CROP_M1_v0.2") is None

        frozen = freeze_accepted_pool_dataset(
            db,
            dataset_version="DS_CROP_M1_v0.2",
            preview_hash=preview["selection_hash"],
            git_commit="test-sha",
        )
        assert frozen["status"] == "FROZEN"
        assert frozen["image_count"] == 4
        assert db.get(DatasetVersion, "DS_CROP_M1_v0.2").status == "FROZEN"
        assert len(crop_calls) == 4

        frozen_rows = list(csv.DictReader(io.StringIO(bucket.blob("datasets/DS_CROP_M1_v0.2/manifest.csv").download_as_text())))
        assert len(frozen_rows) == 4
        assert all(row["source_manifest_uri"].endswith("datasets/accepted_pool/manifest.csv") for row in frozen_rows)
        assert all(bucket.blob(f"datasets/DS_CROP_M1_v0.2/{row['crop_path']}").exists(client) for row in frozen_rows)
    finally:
        db.close()


def test_legacy_dataset_page_uses_explicit_freeze_for_both_sources():
    source = Path("app/templates/datasets.html").read_text(encoding="utf-8")
    assert 'id="sourceMode"' in source
    assert 'value="ORIGINAL"' in source
    assert 'value="ACCEPTED_POOL"' in source
    assert "/api/platform/datasets/crop/create" not in source
    assert "不会自动创建 DatasetVersion" in source
