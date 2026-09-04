from __future__ import annotations

import asyncio
import hashlib
import io
import json
from pathlib import Path

import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.datastructures import Headers, UploadFile

from app import models  # noqa: F401
from app.db import Base
from app.inference_upload_api import (
    InferenceContractError,
    _put_if_absent,
    _read_record,
    review_inference_asset,
    upload_inference_asset,
)
from app.models import FeedbackEvent, InferenceAsset


class FakeBlob:
    def __init__(self):
        self.data: bytes | None = None
        self.upload_count = 0

    def exists(self, _client=None):
        return self.data is not None

    def download_as_bytes(self):
        return self.data or b""

    def upload_from_string(self, data, **_kwargs):
        self.data = bytes(data)
        self.upload_count += 1


class FakeBucket:
    def __init__(self):
        self.blobs: dict[str, FakeBlob] = {}

    def blob(self, name):
        return self.blobs.setdefault(name, FakeBlob())


def _record(image_id="yj_img_123456"):
    return {
        "contract_version": "INFERENCE_RECORD_V2",
        "image_id": image_id,
        "timestamp": "2026-09-01T00:00:00.000Z",
        "source": "camera",
        "source_image_path": "/private/image.jpg",
        "detection": {
            "image_id": image_id,
            "detector_version": "DET_FISH_v0.1",
            "image_width": 10,
            "image_height": 8,
            "candidate_bbox": [0.1, 0.2, 0.5, 0.5],
            "confidence": 0.9,
            "bbox_area_ratio": 0.25,
            "source": "android_detector",
        },
        "crop": {
            "source_image_id": image_id,
            "crop_path": "/private/crop.jpg",
            "expand_ratio": 0.15,
            "crop_width": 6,
            "crop_height": 5,
        },
        "classification": {
            "model_version": "MODEL_M1_v0.2",
            "prediction_species": "grass_carp",
            "confidence": 0.82,
            "latency_ms": 11,
        },
    }


def _image_bytes() -> bytes:
    image = Image.new("RGB", (10, 8), (50, 100, 150))
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


def _upload(data: bytes, filename: str, content_type: str) -> UploadFile:
    return UploadFile(
        file=io.BytesIO(data),
        filename=filename,
        headers=Headers({"content-type": content_type}),
    )


def test_record_rejects_ground_truth_in_detector_payload():
    document = _record()
    document["detection"]["ground_truth_bbox"] = [0.1, 0.2, 0.5, 0.5]
    with pytest.raises(InferenceContractError, match="candidate_bbox"):
        _read_record(json.dumps(document).encode())


def test_record_rejects_nested_identity_mismatch():
    document = _record()
    document["detection"]["image_id"] = "yj_img_other"
    with pytest.raises(InferenceContractError, match="detection.image_id"):
        _read_record(json.dumps(document).encode())


def test_put_if_absent_is_hash_idempotent_and_conflict_safe():
    blob = FakeBlob()
    client = object()
    first = _put_if_absent(blob, b"same", content_type="text/plain", client=client, digest=hashlib.sha256(b"same").hexdigest())
    second = _put_if_absent(blob, b"same", content_type="text/plain", client=client, digest=hashlib.sha256(b"same").hexdigest())
    assert first == "CREATED"
    assert second == "SKIP"
    assert blob.upload_count == 1
    with pytest.raises(InferenceContractError, match="different hash"):
        _put_if_absent(blob, b"other", content_type="text/plain", client=client, digest=hashlib.sha256(b"other").hexdigest())
    assert blob.upload_count == 1


def test_upload_persists_candidate_and_duplicate_without_overwrite(monkeypatch, tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'registry.db'}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    bucket = FakeBucket()

    class Client:
        def bucket(self, name):
            assert name == "test-bucket"
            return bucket

    monkeypatch.setattr("app.inference_upload_api.storage.Client", Client)
    monkeypatch.setattr("app.inference_upload_api.get_bucket_name", lambda: "test-bucket")
    image = _image_bytes()
    record = _record()
    record_bytes = json.dumps(record, ensure_ascii=False).encode()
    result = asyncio.run(
        upload_inference_asset(
            record=_upload(record_bytes, "InferenceRecord.json", "application/json"),
            image=_upload(image, "original.jpg", "image/jpeg"),
            crop=_upload(image, "crop.jpg", "image/jpeg"),
            db=session,
        )
    )
    assert result["status"] == "REVIEW_REQUIRED"
    assert result["storage"]["record"] == "CREATED"
    assert result["record_gcs_uri"].endswith("/yj_img_123456.json")
    row = session.get(InferenceAsset, "yj_img_123456")
    assert row is not None
    assert row.status == "REVIEW_REQUIRED"
    assert row.accepted_bbox_json is None

    duplicate = asyncio.run(
        upload_inference_asset(
            record=_upload(record_bytes, "InferenceRecord.json", "application/json"),
            image=_upload(image, "original.jpg", "image/jpeg"),
            crop=_upload(image, "crop.jpg", "image/jpeg"),
            db=session,
        )
    )
    assert duplicate["duplicate"] is True
    assert all(blob.upload_count == 1 for blob in bucket.blobs.values())

    reviewed = review_inference_asset(
        "yj_img_123456",
        __import__("app.inference_upload_api", fromlist=["InferenceReviewRequest"]).InferenceReviewRequest(
            decision="ACCEPTED", reviewer="reviewer-1", accepted_bbox=[0.1, 0.2, 0.5, 0.5], accepted_species="grass_carp"
        ),
        session,
    )
    assert reviewed["status"] == "ACCEPTED"
    assert session.get(InferenceAsset, "yj_img_123456").accepted_species == "grass_carp"
    session.close()


def test_upload_materializes_nested_feedback_as_existing_review_pool_event(monkeypatch, tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'registry-feedback.db'}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    bucket = FakeBucket()

    class Client:
        def bucket(self, name):
            return bucket

    monkeypatch.setattr("app.inference_upload_api.storage.Client", Client)
    monkeypatch.setattr("app.inference_upload_api.get_bucket_name", lambda: "test-bucket")
    image = _image_bytes()
    record = _record("yj_img_feedback_001")
    record["feedback"] = {
        "source_event_id": "APP_feedback_001",
        "ai_prediction": "草鱼",
        "user_label": "鲤鱼",
        "is_error": True,
        "hard_case": True,
        "feedback_type": "corrected",
        "user_note": "用户确认是鲤鱼",
    }
    result = asyncio.run(
        upload_inference_asset(
            record=_upload(json.dumps(record, ensure_ascii=False).encode(), "InferenceRecord.json", "application/json"),
            image=_upload(image, "original.jpg", "image/jpeg"),
            crop=None,
            db=session,
        )
    )
    assert result["feedback_event"]["source_event_id"] == "APP_feedback_001"
    event = session.query(FeedbackEvent).filter_by(source_event_id="APP_feedback_001").one()
    assert event.feedback_type == "corrected"
    assert event.corrected_species == "鲤鱼"
    assert event.pipeline_status == "NEW"
    session.close()
