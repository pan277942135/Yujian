from __future__ import annotations

import csv
import io
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.accepted_pool import ACCEPTED_POOL_MANIFEST_NAME
from app.db import Base
from app.inspect import inspect_accepted_pool_media, inspect_images
from app.models import Batch, ImageAsset
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


class MemoryBucket:
    def __init__(self):
        self.blobs: dict[str, MemoryBlob] = {}

    def blob(self, name: str):
        return self.blobs.setdefault(name, MemoryBlob(self, name))


class MemoryClient:
    def __init__(self, bucket: MemoryBucket):
        self.bucket_obj = bucket

    def bucket(self, _name: str):
        return self.bucket_obj


def _session(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'accepted-pool-inspect.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def _manifest_bytes() -> bytes:
    rows = [
        {
            "pool_key": "BATCH_POOL_001:pool-image-001",
            "batch_id": "BATCH_POOL_001",
            "source_batch": "BATCH_POOL_001",
            "image_id": "pool-image-001",
            "species": "草鱼",
            "species_name": "草鱼",
            "source_image": "gs://pool-bucket/source/pool-image-001.jpg",
            "crop_path": "images/pool-image-001_crop.jpg",
            "accepted_bbox": "[0.15,0.2,0.5,0.6]",
            "pool_status": "ACTIVE",
        }
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def test_accepted_pool_inspect_uses_materialised_crop_and_keeps_source_traceability(monkeypatch, tmp_path: Path):
    bucket = MemoryBucket()
    client = MemoryClient(bucket)
    monkeypatch.setattr(crop_dataset, "_storage", lambda: (client, bucket))
    monkeypatch.setattr(crop_dataset, "get_bucket_name", lambda: "pool-bucket")
    bucket.blob(ACCEPTED_POOL_MANIFEST_NAME).data = _manifest_bytes()
    crop_bytes = b"materialised-accepted-pool-crop"
    source_bytes = b"original-source-image"
    bucket.blob("datasets/accepted_pool/images/pool-image-001_crop.jpg").data = crop_bytes
    bucket.blob("source/pool-image-001.jpg").data = source_bytes

    Session = _session(tmp_path)
    db = Session()
    try:
        db.add(Batch(batch_id="BATCH_POOL_001", source="upload", manifest_uri="gs://pool-bucket/manifest.csv", raw_uri="gs://pool-bucket/raw"))
        db.add(
            ImageAsset(
                batch_id="BATCH_POOL_001",
                image_id="pool-image-001",
                file_name="pool-image-001.jpg",
                object_name="source/pool-image-001.jpg",
                gcs_uri="gs://pool-bucket/source/pool-image-001.jpg",
                truth_species="草鱼",
                review_status="approved",
            )
        )
        db.commit()

        result = inspect_images(
            species="草鱼",
            review_status="approved",
            source="accepted_pool",
            limit=24,
            offset=0,
            db=db,
        )
        assert result["source"] == "accepted_pool"
        assert result["total"] == 1
        item = result["items"][0]
        assert item["accepted_pool"] is True
        assert item["media_source"] == "accepted_pool_crop"
        assert item["accepted_bbox"] == [0.15, 0.2, 0.5, 0.6]
        assert "/api/inspect/accepted-pool/media" in item["media_url"]
        assert "/media/BATCH_POOL_001/pool-image-001" not in item["media_url"]
        assert "variant=thumbnail" in item["thumbnail_url"]
        assert item["source_image_url"] == "/media/BATCH_POOL_001/pool-image-001"

        response = inspect_accepted_pool_media(pool_key="BATCH_POOL_001:pool-image-001", variant=None)
        assert response.status_code == 200
        assert response.body == crop_bytes
        assert response.body != source_bytes
    finally:
        db.close()


def test_original_inspect_source_still_uses_legacy_media_gateway(tmp_path: Path):
    Session = _session(tmp_path)
    db = Session()
    try:
        db.add(Batch(batch_id="BATCH_ORIGINAL_001", source="upload", manifest_uri="gs://pool/manifest.csv", raw_uri="gs://pool/raw"))
        db.add(
            ImageAsset(
                batch_id="BATCH_ORIGINAL_001",
                image_id="original-001",
                file_name="original-001.jpg",
                object_name="source/original-001.jpg",
                gcs_uri="gs://pool/source/original-001.jpg",
                truth_species="鲫鱼",
                review_status="approved",
            )
        )
        db.commit()
        result = inspect_images(review_status="approved", source="original", limit=24, offset=0, db=db)
        assert result["source"] == "original"
        assert result["items"][0]["media_url"] == "/media/BATCH_ORIGINAL_001/original-001"
        assert result["items"][0]["thumbnail_url"].endswith("?variant=thumbnail")
    finally:
        db.close()


def test_dataset_page_routes_accepted_pool_tags_to_crop_inspection():
    source = Path("app/templates/datasets.html").read_text(encoding="utf-8")
    inspect_template = Path("app/templates/inspect.html").read_text(encoding="utf-8")
    assert "/inspect?source=accepted_pool&review_status=approved&species=" in source
    assert 'href="/inspect?source=accepted_pool&review_status=approved"' in source
    assert "仅显示 Accepted Pool 裁剪图" in inspect_template
    assert "x.thumbnail_url||x.media_url" in inspect_template
    assert "Accepted Pool Crop" in inspect_template
