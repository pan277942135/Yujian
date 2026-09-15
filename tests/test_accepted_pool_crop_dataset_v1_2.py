from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import Batch, DatasetVersion, ImageAsset
from app.platform.routes.api import CropDatasetCreate, platform_crop_dataset_create
from app.platform.services import crop_dataset
from app.training_api import TrainingCreate, queue_training_run
from trainer.crop_dataset_validator import validate_crop_rows


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

    def download_to_filename(self, filename, **_kwargs):
        Path(filename).write_bytes(self.data or b"")

    def upload_from_string(self, data, **_kwargs):
        self.data = data.encode("utf-8") if isinstance(data, str) else bytes(data)

    def copy_to(self, bucket: "MemoryBucket", destination: str):
        bucket.blob(destination).data = self.data


class MemoryBucket:
    def __init__(self):
        self.blobs: dict[str, MemoryBlob] = {}

    def blob(self, name: str):
        return self.blobs.setdefault(name, MemoryBlob(self, name))

    def list_blobs(self, prefix: str = ""):
        return [blob for name, blob in self.blobs.items() if name.startswith(prefix)]


class MemoryClient:
    def __init__(self, bucket: MemoryBucket):
        self._bucket = bucket

    def bucket(self, _name: str):
        return self._bucket


def _image_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (120, 80), (40, 100, 160)).save(output, format="JPEG")
    return output.getvalue()


def _db(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'accepted-pool-v1-2.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def test_platform_accepted_pool_full_requires_explicit_legacy_freeze():
    with pytest.raises(ValueError, match="ACCEPTED_POOL_FULL_REQUIRES_LEGACY_DATASET_FREEZE"):
        crop_dataset.start_crop_dataset_job(
            source="ACCEPTED_POOL",
            dataset_name="DS_CROP_M1_v0.2",
            expand_ratio=1.0,
            size=416,
            mode="FULL",
        )
    with pytest.raises(HTTPException) as error:
        platform_crop_dataset_create(CropDatasetCreate(source="accepted_pool", mode="FULL"))
    assert error.value.status_code == 409
    assert error.value.detail["error"] == "EXPLICIT_DATASET_FREEZE_REQUIRED"


def test_v12_qa_selects_risk_rows_without_good_filter():
    rows = []
    for index in range(60):
        split = "train" if index < 40 else ("val" if index < 50 else "test")
        rows.append(
            {
                "image_id": f"risk-{index:03d}",
                "batch_id": "BATCH_ACCEPTED_001",
                "quality_status": "WARNING",
                "quality_reason": "crop触边",
                "split": split,
            }
        )

    first = crop_dataset.select_random_50_qa_rows(rows, "DS_CROP_M1_v0.2")
    second = crop_dataset.select_random_50_qa_rows(rows, "DS_CROP_M1_v0.2")
    assert len(first) == 50
    assert [row["image_id"] for row in first] == [row["image_id"] for row in second]
    assert all(row["quality_status"] == "WARNING" for row in first)
    assert sum(row["split"] == "train" for row in first) == 35
    assert sum(row["split"] == "val" for row in first) == 8
    assert sum(row["split"] == "test" for row in first) == 7


def test_v12_release_qa_keeps_fixed_50_snapshot_and_opens_gate(monkeypatch, tmp_path: Path):
    bucket = MemoryBucket()
    client = MemoryClient(bucket)
    Session = _db(tmp_path)
    monkeypatch.setattr(crop_dataset, "_storage", lambda: (client, bucket))
    monkeypatch.setattr(crop_dataset, "get_bucket_name", lambda: "pool-bucket")
    rows = [
        {
            "image_id": f"qa-{index:03d}",
            "batch_id": "BATCH_ACCEPTED_001",
            "quality_status": "WARNING",
            "quality_reason": "risk-only-test",
            "split": "train" if index < 35 else ("val" if index < 43 else "test"),
            "crop_path": f"images/qa-{index:03d}.jpg",
            "source_image": f"gs://pool-bucket/source/qa-{index:03d}.jpg",
            "bbox": "[0.1,0.1,0.5,0.5]",
        }
        for index in range(50)
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    bucket.blob("datasets/DS_CROP_M1_v0.2/manifest.csv").data = output.getvalue().encode("utf-8")

    db = Session()
    try:
        db.add(
            DatasetVersion(
                dataset_version="DS_CROP_M1_v0.2",
                manifest_uri="gs://pool-bucket/datasets/DS_CROP_M1_v0.2/manifest.csv",
                train_count=35,
                val_count=8,
                test_count=7,
                species_count=1,
                git_commit="test-sha",
                selection_mode="ACCEPTED_POOL_DETECTOR_CROP",
                status="RELEASE_QA_PENDING",
                pipeline_type="CROP_CLASSIFIER_V1",
                metadata_json=json.dumps({"source": "ACCEPTED_POOL"}),
            )
        )
        db.commit()

        first = crop_dataset.start_random_50_qa("DS_CROP_M1_v0.2", db)
        second = crop_dataset.start_random_50_qa("DS_CROP_M1_v0.2", db)
        assert first["sample_size"] == 50
        assert [item["item_id"] for item in first["items"]] == [item["item_id"] for item in second["items"]]

        final = first
        for index in range(50):
            final = crop_dataset.review_random_50_qa("DS_CROP_M1_v0.2", index, "PASS", "通过", db)
        assert final["status"] == "PASS"
        assert final["final_release_gate"] == "PASS"
        assert final["reviewed_count"] == 50
        db.refresh(db.get(DatasetVersion, "DS_CROP_M1_v0.2"))
        assert db.get(DatasetVersion, "DS_CROP_M1_v0.2").status == "READY_FOR_TRAINING"
    finally:
        db.close()


def test_v12_training_gate_is_409_until_release_qa_pass(monkeypatch, tmp_path: Path):
    bucket = MemoryBucket()
    client = MemoryClient(bucket)
    monkeypatch.setattr(crop_dataset, "_storage", lambda: (client, bucket))
    monkeypatch.setattr(crop_dataset, "get_bucket_name", lambda: "pool-bucket")
    Session = _db(tmp_path)
    db = Session()
    try:
        dataset = DatasetVersion(
            dataset_version="DS_CROP_M1_v0.2",
            manifest_uri="gs://pool-bucket/datasets/DS_CROP_M1_v0.2/manifest.csv",
            class_map_uri="gs://pool-bucket/datasets/DS_CROP_M1_v0.2/metadata/class_map.json",
            train_count=35,
            val_count=8,
            test_count=7,
            species_count=1,
            git_commit="test-sha",
            selection_mode="ACCEPTED_POOL_DETECTOR_CROP",
            status="RELEASE_QA_PENDING",
            pipeline_type="CROP_CLASSIFIER_V1",
            metadata_json=json.dumps({"source": "ACCEPTED_POOL"}),
        )
        db.add(dataset)
        db.commit()
        payload = TrainingCreate(
            dataset_version="DS_CROP_M1_v0.2",
            run_id="RUN_DS_CROP_M1_v0.2_GATE",
            model_version="MODEL_DS_CROP_M1_v0.2_GATE",
            pipeline_type="CROP_CLASSIFIER_V1",
        )
        with pytest.raises(HTTPException) as error:
            queue_training_run(db, payload, launcher=lambda *_args: {"name": "unused"})
        assert error.value.status_code == 409

        dataset.status = "READY_FOR_TRAINING"
        dataset.metadata_json = json.dumps(
            {
                "source": "ACCEPTED_POOL",
                "release_gate": {
                    "random_50_qa": {"status": "PASS", "sample_size": 50, "reviewed_count": 50, "pass_count": 50, "issue_count": 0},
                    "final_release_gate": "PASS",
                },
            }
        )
        db.commit()
        result = queue_training_run(db, payload, launcher=lambda *_args: {"name": "unused"})
        assert result["run_id"] == payload.run_id
        assert result["dataset_version"] == "DS_CROP_M1_v0.2"
    finally:
        db.close()
