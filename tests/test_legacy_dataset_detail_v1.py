from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.requests import Request

from app.db import Base
from app.main import _legacy_qa_with_media, legacy_dataset_detail, legacy_dataset_operations
from app.models import DatasetVersion
from app.platform.models import PlatformOperationLog
from app.platform.services import crop_dataset
from app.training_api import TrainingCreate, queue_training_run


class FakeBlob:
    def __init__(self, bucket: "FakeBucket", name: str):
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

    def copy_to(self, bucket: "FakeBucket", destination: str):
        bucket.blob(destination).data = self.data


class FakeBucket:
    def __init__(self):
        self.blobs: dict[str, FakeBlob] = {}
        self.requested: list[str] = []

    def blob(self, name: str):
        self.requested.append(name)
        return self.blobs.setdefault(name, FakeBlob(self, name))


class FakeClient:
    def __init__(self, bucket: FakeBucket):
        self._bucket = bucket

    def bucket(self, _name):
        return self._bucket


def _db(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy-detail.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)()


def _install_storage(monkeypatch: pytest.MonkeyPatch, bucket: FakeBucket):
    monkeypatch.setattr(crop_dataset, "_storage", lambda: (FakeClient(bucket), bucket))
    monkeypatch.setattr(crop_dataset, "get_bucket_name", lambda: "test-bucket")


def _csv(rows: list[dict[str, str]], fields: list[str] | None = None) -> bytes:
    fields = fields or list(rows[0])
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _qa_rows(count: int = 50) -> list[dict[str, str]]:
    rows = []
    for index in range(count):
        split = "train" if index < 35 else ("val" if index < 43 else "test")
        rows.append(
            {
                "image_id": f"image-{index:03d}",
                "batch_id": "BATCH_FREEZE_001",
                "crop_path": f"crops/image-{index:03d}.jpg",
                "species": "草鱼",
                "source_image": f"source/image-{index:03d}.jpg",
                "bbox": "[0.1,0.1,0.5,0.5]",
                "pixel_bbox": "[10,10,60,60]",
                "source_size": "[100,100]",
                "quality_status": "GOOD",
                "quality_reason": "",
                "split": split,
            }
        )
    return rows


def _dataset(*, uri: str, status: str = "READY_FOR_TRAINING", metadata: dict | None = None):
    return DatasetVersion(
        dataset_version="DS_CROP_M1_v0.1",
        manifest_uri=uri,
        train_count=35,
        val_count=8,
        test_count=7,
        species_count=1,
        git_commit="test-sha",
        selection_mode="ACCEPTED_BBOX_CROP",
        status=status,
        pipeline_type="CROP_CLASSIFIER_V1",
        metadata_json=json.dumps(metadata or {}, ensure_ascii=False),
    )


def test_quality_analysis_reads_only_current_manifest_and_invalidates_old_report(monkeypatch):
    bucket = FakeBucket()
    _install_storage(monkeypatch, bucket)
    rows = [
        {"image_id": "good", "batch_id": "B1", "quality_status": "GOOD", "quality_reason": "", "crop_path": "crops/good.jpg"},
        {"image_id": "warning", "batch_id": "B1", "quality_status": "WARNING", "quality_reason": "crop触边", "crop_path": "crops/warning.jpg"},
        {"image_id": "invalid", "batch_id": "B1", "quality_status": "INVALID", "quality_reason": "图片不可用", "crop_path": "crops/invalid.jpg"},
        {"image_id": "good-2", "batch_id": "B1", "quality_status": "GOOD", "quality_reason": "", "crop_path": "crops/good-2.jpg"},
    ]
    fields = ["image_id", "batch_id", "quality_status", "quality_reason", "crop_path"]
    bucket.blob("datasets/DS_CROP_M1_v0.1/manifest.csv").data = _csv(rows, fields)
    bucket.blob("datasets/DS_CROP_M1_v0.1/manifest_all.csv").data = _csv(
        [{"image_id": "wrong", "quality_status": "GOOD"}], ["image_id", "quality_status"]
    )
    bucket.blob("datasets/DS_CROP_M1_v0.1/reports/quality_gate_analysis.json").data = json.dumps(
        {
            "schema_version": "QUALITY_GATE_ANALYSIS_V1",
            "source": {
                "manifest_uri": "gs://test-bucket/datasets/DS_CROP_M1_v0.1/manifest_all.csv",
                "source_is_frozen_manifest_all": True,
                "source_count": 624798,
            },
            "totals": {"TOTAL": 624798, "GOOD": 624798, "WARNING": 0, "INVALID": 0},
        }
    ).encode("utf-8")
    for row in rows:
        bucket.blob(f"datasets/DS_CROP_M1_v0.1/{row['crop_path']}").data = b"crop"
    request_start = len(bucket.requested)

    report = crop_dataset.generate_quality_gate_analysis("DS_CROP_M1_v0.1")

    assert report["source"]["manifest_uri"].endswith("/datasets/DS_CROP_M1_v0.1/manifest.csv")
    assert report["source"]["source_is_frozen_manifest"] is True
    assert report["source"]["source_count"] == 4
    assert report["totals"]["TOTAL"] == 4
    assert report["totals"]["GOOD"] + report["totals"]["WARNING"] + report["totals"]["INVALID"] == 4
    assert report["quality_sum_check"] is True
    assert "manifest_all.csv" not in report["source"]["manifest_uri"]
    assert "datasets/DS_CROP_M1_v0.1/manifest_all.csv" not in bucket.requested[request_start:]
    assert "UNKNOWN" not in json.dumps(report, ensure_ascii=False)
    assert "NO_REASON" not in json.dumps(report, ensure_ascii=False)


def test_quality_analysis_reports_unavailable_quality_field_without_unknown_tokens(monkeypatch):
    bucket = FakeBucket()
    _install_storage(monkeypatch, bucket)
    bucket.blob("datasets/DS_CROP_M1_v0.1/manifest.csv").data = _csv(
        [{"image_id": "one", "species": "鲫鱼"}, {"image_id": "two", "species": "鲤鱼"}],
        ["image_id", "species"],
    )

    report = crop_dataset.generate_quality_gate_analysis("DS_CROP_M1_v0.1")

    assert report["quality_field_available"] is False
    assert report["totals"] == {"TOTAL": 2, "GOOD": 0, "WARNING": 0, "INVALID": 0, "UNAVAILABLE": 2}
    assert report["quality_sum_check"] is False
    assert report["reasons"][0]["status"] == "UNAVAILABLE"
    assert report["reasons"][0]["reason"] == "质量字段不可用"
    serialized = json.dumps(report, ensure_ascii=False)
    assert "UNKNOWN" not in serialized
    assert "NO_REASON" not in serialized


def test_registered_manifest_counts_are_preferred_to_stale_dataset_counters(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "image_id,split\n1,train\n2,train\n3,val\n4,test\n",
        encoding="utf-8",
    )
    db = _db(tmp_path)
    try:
        dataset = _dataset(uri=str(manifest))
        dataset.train_count = 999
        db.add(dataset)
        db.commit()
        result = legacy_dataset_detail(dataset.dataset_version, db)
        assert result["sample_count"] == 4
        assert result["train_count"] == 2
        assert result["val_count"] == 1
        assert result["test_count"] == 1
        assert "release_gate" not in result
        assert "release_qa_checked" not in result["processing"]
        assert "release_qa_total" not in result["processing"]
    finally:
        db.close()


def test_release_qa_uses_one_persistent_50_row_snapshot_and_saves_progress(monkeypatch, tmp_path: Path):
    bucket = FakeBucket()
    _install_storage(monkeypatch, bucket)
    rows = _qa_rows()
    bucket.blob("datasets/DS_CROP_M1_v0.1/manifest.csv").data = _csv(rows)
    bucket.blob("datasets/DS_CROP_M1_v0.1/manifest_all.csv").data = b"must not be read"
    request_start = len(bucket.requested)
    db = _db(tmp_path)
    try:
        db.add(_dataset(uri="gs://test-bucket/datasets/DS_CROP_M1_v0.1/manifest.csv"))
        db.commit()
        first = crop_dataset.start_random_50_qa("DS_CROP_M1_v0.1", db)
        requested_after_first = list(bucket.requested[request_start:])
        second = crop_dataset.start_random_50_qa("DS_CROP_M1_v0.1", db)
        assert first["sample_size"] == 50
        assert len(first["items"]) == 50
        assert [item["item_id"] for item in first["items"]] == [item["item_id"] for item in second["items"]]
        assert bucket.blob("datasets/DS_CROP_M1_v0.1/qa/random_50_qa.json").exists()
        assert bucket.blob("datasets/DS_CROP_M1_v0.1/qa/random_50_qa.csv").exists()
        assert "datasets/DS_CROP_M1_v0.1/manifest_all.csv" not in requested_after_first

        reviewed = crop_dataset.review_random_50_qa("DS_CROP_M1_v0.1", 0, "PASS", "检查通过", db)
        assert reviewed["checked"] == 1
        assert reviewed["passed"] == 1
        persisted = crop_dataset.get_random_50_qa("DS_CROP_M1_v0.1")
        assert persisted["reviewed_count"] == 1
        assert persisted["items"][0]["decision"] == "PASS"
    finally:
        db.close()


def _jpeg(color=(20, 80, 140), size=(100, 80)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format="JPEG")
    return output.getvalue()


def test_release_qa_media_exposes_source_overlay_and_training_crop(monkeypatch):
    bucket = FakeBucket()
    _install_storage(monkeypatch, bucket)
    item = {
        "qa_index": 0,
        "source_image": "source.jpg",
        "crop_path": "crop.jpg",
        "bbox": "[0.1,0.1,0.5,0.5]",
        "pixel_bbox": "[10,10,60,60]",
        "source_size": "[100,80]",
    }
    qa = {"source_manifest": "manifest.csv", "sample_size": 50, "items": [item] + [{"qa_index": i} for i in range(1, 50)]}
    bucket.blob("datasets/DS_CROP_M1_v0.1/qa/random_50_qa.json").data = json.dumps(qa).encode("utf-8")
    bucket.blob("datasets/DS_CROP_M1_v0.1/source.jpg").data = _jpeg()
    bucket.blob("datasets/DS_CROP_M1_v0.1/crop.jpg").data = _jpeg((200, 20, 20), (40, 40))

    source = crop_dataset.read_random_50_qa_media("DS_CROP_M1_v0.1", 0, kind="source")
    overlay = crop_dataset.read_random_50_qa_media("DS_CROP_M1_v0.1", 0, kind="source_bbox")
    crop = crop_dataset.read_random_50_qa_media("DS_CROP_M1_v0.1", 0, kind="crop")
    media = _legacy_qa_with_media("DS_CROP_M1_v0.1", qa)["items"][0]

    assert Image.open(io.BytesIO(source)).size == (100, 80)
    assert Image.open(io.BytesIO(overlay)).size == (100, 80)
    assert Image.open(io.BytesIO(crop)).size == (40, 40)
    assert source != overlay
    assert media["source_image_url"].endswith("kind=source")
    assert media["bbox_overlay_url"].endswith("kind=source_bbox")
    assert media["crop_media_url"].endswith("kind=crop")


def test_training_does_not_require_release_qa_after_dataset_freeze(monkeypatch, tmp_path: Path):
    bucket = FakeBucket()
    _install_storage(monkeypatch, bucket)
    db = _db(tmp_path)
    try:
        db.add(
            _dataset(
                uri="gs://test-bucket/datasets/DS_CROP_M1_v0.1/manifest.csv",
                status="FROZEN",
            )
        )
        db.commit()
        result = queue_training_run(
            db,
            TrainingCreate(
                dataset_version="DS_CROP_M1_v0.1",
                run_id="RUN_CROP_M1_v0.1_GATE",
                model_version="MODEL_CROP_M1_v0.1_GATE",
                pipeline_type="CROP_CLASSIFIER_V1",
            ),
            launcher=lambda *_args: {"name": "unused"},
        )
        assert result["dataset_version"] == "DS_CROP_M1_v0.1"
    finally:
        db.close()


def test_dataset_detail_page_and_operation_timeline_contract(tmp_path: Path):
    from app.main import templates as legacy_templates

    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_id,split\n1,train\n", encoding="utf-8")
    db = _db(tmp_path)
    try:
        db.add(_dataset(uri=str(manifest), metadata={"source_batch_id": "BATCH_FREEZE_001"}))
        db.add(
            PlatformOperationLog(
                operation_type="RANDOM_50_QA_START",
                resource_type="dataset_release",
                resource_id="DS_CROP_M1_v0.1",
                status="SUCCESS",
            )
        )
        db.commit()
        operations = legacy_dataset_operations("DS_CROP_M1_v0.1", limit=50, db=db)
        assert operations[0]["label"] == "创建QA Snapshot"
        request = Request({"type": "http", "method": "GET", "path": "/datasets/DS_CROP_M1_v0.1", "query_string": b"", "headers": []})
        rendered = legacy_templates.env.get_template("legacy_dataset_detail.html").render(
            request=request,
            dataset_version="DS_CROP_M1_v0.1",
        )
        for text in (
            "数据集详情",
            "A 数据集概览",
            "B 冻结数据质量分析",
            "C 模型训练",
            "D 操作记录",
            "创建训练任务",
        ):
            assert text in rendered
        for text in (
            "发布前质量确认",
            "Release QA",
            "/release-qa",
            "通过并下一张",
            "标记问题",
            "BBox Overlay",
        ):
            assert text not in rendered
        assert "UNKNOWN" not in rendered
        assert "NO_REASON" not in rendered
        assert "严重问题" not in rendered
    finally:
        db.close()
