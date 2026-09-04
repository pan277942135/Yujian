from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.dataset_crop_review import DatasetCropReviewUpdate, crop_preview, items, summary, update
import app.dataset_crop_review as dataset_crop_review
from app.frozen_crop_bridge import crop_readiness, load_frozen_dataset
from app.models import DatasetCropReview, DatasetVersion
from app.recognition_pipeline import BBox, Detection
from trainer.build_reviewed_datasets import build_crop_dataset


def _db(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'registry.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)()


def _parent(tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / "dataset_manifest.csv"
    for name in ("img_a.jpg", "img_b.jpg"):
        (tmp_path / name).write_bytes(b"test-image")
    manifest.write_text(
        "batch_id,image_id,gcs_uri,species_key,species,truth_species,review_status,class_index,split,group_id\n"
        f"BATCH_1,img_a,{tmp_path / 'img_a.jpg'},grass_carp,草鱼,草鱼,approved,4,train,g1\n"
        f"BATCH_1,img_b,{tmp_path / 'img_b.jpg'},common_carp,鲤鱼,鲤鱼,approved,9,test,g2\n",
        encoding="utf-8",
    )
    class_map = tmp_path / "class_map.json"
    class_map.write_text(
        json.dumps({"classes": [{"class_index": 4, "species_key": "grass_carp"}, {"class_index": 9, "species_key": "common_carp"}]}),
        encoding="utf-8",
    )
    return manifest, class_map


def test_frozen_manifest_is_the_registered_source_and_readiness_is_dynamic(tmp_path: Path):
    manifest, class_map = _parent(tmp_path)
    db = _db(tmp_path)
    try:
        db.add(DatasetVersion(dataset_version="DS_M1_v0.5", manifest_uri=str(manifest), class_map_uri=str(class_map), git_commit="sha", status="FROZEN"))
        db.commit()
        loaded = load_frozen_dataset(db, "DS_M1_v0.5")
        assert len(loaded["rows"]) == 2
        assert loaded["rows"][0]["class_index"] == 4
        assert loaded["rows"][0]["split"] == "train"
        ready = crop_readiness(db, "DS_M1_v0.5")
        assert ready["ground_truth_confirmed"] is True
        assert ready["crop_ready"] is False
        assert ready["images"] == 2
        assert summary("DS_M1_v0.5", db)["bbox_required"] == 2
    finally:
        db.close()


def test_frozen_review_never_requests_species_and_preserves_parent_metadata(tmp_path: Path, monkeypatch):
    manifest, class_map = _parent(tmp_path)
    db = _db(tmp_path)
    try:
        db.add(DatasetVersion(dataset_version="DS_M1_v0.5", manifest_uri=str(manifest), class_map_uri=str(class_map), git_commit="sha", status="FROZEN"))
        db.commit()
        rows = items("DS_M1_v0.5", status="BBOX_REQUIRED", db=db)["items"]
        assert rows[0]["species_name"] == "草鱼"
        assert rows[0]["split"] == "train"
        image_bytes = io.BytesIO()
        Image.new("RGB", (32, 24), (10, 20, 30)).save(image_bytes, format="JPEG")
        monkeypatch.setattr(dataset_crop_review, "_read_uri", lambda _uri: (image_bytes.getvalue(), None))
        result = update("DS_M1_v0.5", "img_a", DatasetCropReviewUpdate(decision="ACCEPTED", accepted_bbox=[0.1, 0.1, 0.7, 0.7]), db)
        assert result["status"] == "ACCEPTED"
        assert result["species_key"] == "grass_carp"
        assert result["accepted_bbox"] == [0.1, 0.1, 0.7, 0.7]
        assert result["bbox_source"] == "accepted_review"
        assert result["crop_status"] == "READY"
        assert result["crop_uri"].endswith("img_a_crop_preview.jpg")
        assert result["preview_url"].endswith("/DS_M1_v0.5/img_a/crop")
        preview = crop_preview("DS_M1_v0.5", "img_a", db)
        assert preview.media_type == "image/jpeg"
        assert preview.body
        persisted = db.scalar(
            select(DatasetCropReview).where(
                DatasetCropReview.source_dataset_version == "DS_M1_v0.5",
                DatasetCropReview.image_id == "img_a",
            )
        )
        assert persisted.accepted_bbox_json == "[0.1,0.1,0.7,0.7]"
        assert persisted.crop_status == "READY"
        assert summary("DS_M1_v0.5", db)["accepted"] == 1
    finally:
        db.close()


def test_frozen_review_paginates_and_generates_candidate_bbox_without_accepting_it(tmp_path: Path, monkeypatch):
    manifest = tmp_path / "dataset_manifest.csv"
    class_map = tmp_path / "class_map.json"
    rows = [
        f"BATCH_1,img_{index},{tmp_path / f'img_{index}.jpg'},grass_carp,草鱼,草鱼,approved,4,train,g1"
        for index in range(55)
    ]
    manifest.write_text(
        "batch_id,image_id,gcs_uri,species_key,species,truth_species,review_status,class_index,split,group_id\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )
    class_map.write_text(
        json.dumps({"classes": [{"class_index": 4, "species_key": "grass_carp"}]}),
        encoding="utf-8",
    )
    db = _db(tmp_path)
    try:
        db.add(DatasetVersion(dataset_version="DS_M1.v0.5", manifest_uri=str(manifest), class_map_uri=str(class_map), git_commit="sha", status="FROZEN"))
        db.commit()
        image_bytes = io.BytesIO()
        Image.new("RGB", (16, 16), (10, 20, 30)).save(image_bytes, format="JPEG")
        monkeypatch.setattr(dataset_crop_review, "_read_uri", lambda _uri: (image_bytes.getvalue(), None))
        monkeypatch.setattr(
            "app.detector_runtime.detect",
            lambda _image: SimpleNamespace(
                model_version="DET_FISH_v0.1",
                detections=(Detection(confidence=0.9, box=BBox(0.1, 0.2, 0.6, 0.6)),),
            ),
        )
        first = items("DS_M1.v0.5", db=db)
        assert first["total"] == 55
        assert first["page"] == 1
        assert first["page_size"] == 50
        assert len(first["items"]) == 50
        assert first["items"][0]["candidate_bbox"] == [0.1, 0.2, 0.5, 0.4]
        assert first["items"][0]["detector_version"] == "DET_FISH_v0.1"
        assert first["items"][0]["detector_confidence"] == 0.9
        assert first["items"][0]["quality_status"] == "GOOD"
        assert first["items"][0]["accepted_bbox"] is None
        counts = summary("DS_M1.v0.5", db)
        assert counts["candidate_bbox_count"] == 50
        assert counts["accepted_bbox_count"] == 0

        second = items("DS_M1.v0.5", page=2, page_size=50, db=db)
        assert second["total"] == 55
        assert len(second["items"]) == 5
    finally:
        db.close()


def test_detector_audit_is_read_only_and_uses_quality_gate(tmp_path: Path, monkeypatch):
    manifest, class_map = _parent(tmp_path)
    db = _db(tmp_path)
    try:
        db.add(DatasetVersion(dataset_version="DS_M1_v0.5", manifest_uri=str(manifest), class_map_uri=str(class_map), git_commit="sha", status="FROZEN"))
        db.commit()
        image_bytes = io.BytesIO()
        Image.new("RGB", (32, 24), (10, 20, 30)).save(image_bytes, format="JPEG")
        monkeypatch.setattr(dataset_crop_review, "_read_uri", lambda _uri: (image_bytes.getvalue(), None))
        monkeypatch.setattr(
            "app.detector_runtime.detect",
            lambda _image: SimpleNamespace(
                model_version="DET_FISH_v0.1",
                detections=(Detection(confidence=0.9, box=BBox(0.1, 0.2, 0.6, 0.8)),),
            ),
        )
        result = dataset_crop_review.detector_audit("DS_M1_v0.5", sample_size=2, seed=7, db=db)
        assert result["read_only"] is True
        assert result["total"] == 2
        assert result["sample_size"] == 2
        assert result["detected"] == 2
        assert result["quality_good"] == 2
        assert db.query(DatasetCropReview).count() == 0
    finally:
        db.close()


def test_legacy_candidate_is_refreshed_without_touching_human_decision(tmp_path: Path, monkeypatch):
    manifest, class_map = _parent(tmp_path)
    db = _db(tmp_path)
    try:
        db.add(DatasetVersion(dataset_version="DS_M1_v0.5", manifest_uri=str(manifest), class_map_uri=str(class_map), git_commit="sha", status="FROZEN"))
        db.commit()
        items("DS_M1_v0.5", status="BBOX_REQUIRED", db=db)
        row = db.scalar(
            select(DatasetCropReview).where(
                DatasetCropReview.source_dataset_version == "DS_M1_v0.5",
                DatasetCropReview.image_id == "img_a",
            )
        )
        row.candidate_bbox_json = "[0.01,0.01,0.1,0.1]"
        row.accepted_bbox_json = "[0.2,0.2,0.5,0.5]"
        row.review_status = "ACCEPTED"
        db.commit()
        image_bytes = io.BytesIO()
        Image.new("RGB", (32, 24), (10, 20, 30)).save(image_bytes, format="JPEG")
        monkeypatch.setattr(dataset_crop_review, "_read_uri", lambda _uri: (image_bytes.getvalue(), None))
        monkeypatch.setattr(
            "app.detector_runtime.detect",
            lambda _image: SimpleNamespace(
                model_version="DET_FISH_v0.1",
                detections=(Detection(confidence=0.9, box=BBox(0.1, 0.2, 0.6, 0.6)),),
            ),
        )
        result = items("DS_M1_v0.5", status="ALL", page_size=50, db=db)
        refreshed = next(item for item in result["items"] if item["image_id"] == "img_a")
        assert refreshed["candidate_bbox"] == [0.1, 0.2, 0.5, 0.4]
        assert refreshed["accepted_bbox"] == [0.2, 0.2, 0.5, 0.5]
        assert refreshed["status"] == "ACCEPTED"
    finally:
        db.close()


def test_accept_keeps_human_bbox_when_preview_materialization_fails(tmp_path: Path, monkeypatch):
    manifest, class_map = _parent(tmp_path)
    db = _db(tmp_path)
    try:
        db.add(DatasetVersion(dataset_version="DS_M1_v0.5", manifest_uri=str(manifest), class_map_uri=str(class_map), git_commit="sha", status="FROZEN"))
        db.commit()
        items("DS_M1_v0.5", status="BBOX_REQUIRED", db=db)
        monkeypatch.setattr(dataset_crop_review, "_persist_crop_preview", lambda _base, _box: (_ for _ in ()).throw(RuntimeError("source unavailable")))
        result = update("DS_M1_v0.5", "img_a", DatasetCropReviewUpdate(decision="ACCEPTED", accepted_bbox=[0.1, 0.1, 0.7, 0.7]), db)
        assert result["status"] == "ACCEPTED"
        assert result["accepted_bbox"] == [0.1, 0.1, 0.7, 0.7]
        assert result["crop_status"] == "ERROR"
        assert result["crop_uri"] is None
        assert "source unavailable" in result["crop_error"]
        assert result["preview_url"] is None
    finally:
        db.close()


def test_crop_builder_preserves_frozen_split_and_class_map(tmp_path: Path):
    image = io.BytesIO()
    Image.new("RGB", (100, 80), (10, 20, 30)).save(image, format="JPEG")
    records = [
        {"image_id": "a", "status": "ACCEPTED", "accepted_bbox": [0.1, 0.1, 0.5, 0.5], "accepted_species_key": "grass_carp", "accepted_species_name": "草鱼", "class_index": 4, "split": "val", "source_image": "gs://b/a.jpg"},
        {"image_id": "b", "status": "ACCEPTED", "accepted_bbox": [0.1, 0.1, 0.5, 0.5], "accepted_species_key": "common_carp", "accepted_species_name": "鲤鱼", "class_index": 9, "split": "test", "source_image": "gs://b/b.jpg"},
    ]
    report = build_crop_dataset(records, tmp_path / "crop", image_loader=lambda _uri: image.getvalue(), preserve_parent_split=True, preserve_parent_class_map=True, input_type="crop_image")
    assert report["validation"]["valid"] is True
    rows = list(csv.DictReader((tmp_path / "crop" / "metadata" / "crop_manifest.csv").open(encoding="utf-8")))
    assert {row["split"] for row in rows} == {"val", "test"}
    assert {row["class_index"] for row in rows} == {"4", "9"}
