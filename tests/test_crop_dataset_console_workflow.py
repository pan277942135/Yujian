from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi import HTTPException
from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app import crop_dataset_api
from app.crop_review import CropReviewUpdate, crop_review_items, update_crop_review
from app.db import Base
from app.models import Batch, BatchCropReview, ImageAsset
from app.presence import FishPresenceResult
from trainer.crop_dataset_pipeline import load_reviewed_crop_records


def _jpeg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (80, 60), (50, 90, 130))
    output = io.BytesIO()
    image.save(output, format="JPEG")
    path.write_bytes(output.getvalue())


def _db(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'registry.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)()


def _add_reviewed(db, tmp_path: Path, batch_id: str, image_id: str, status: str = "ACCEPTED"):
    source = tmp_path / f"{image_id}.jpg"
    _jpeg(source)
    if not db.get(Batch, batch_id):
        db.add(
            Batch(
                batch_id=batch_id,
                source="P5",
                manifest_uri="gs://bucket/manifest.csv",
                raw_uri="gs://bucket/raw",
                image_count=1,
            )
        )
        db.flush()
    image = ImageAsset(
        batch_id=batch_id,
        image_id=image_id,
        file_name=source.name,
        object_name=source.name,
        gcs_uri=str(source),
        claimed_species="草鱼",
    )
    db.add(image)
    db.flush()
    db.add(
        BatchCropReview(
            batch_id=batch_id,
            image_asset_id=image.id,
            image_id=image_id,
            accepted_bbox_json="[0.1,0.1,0.5,0.5]" if status in {"ACCEPTED", "TRAINING_READY"} else None,
            species_key="grass_carp" if status in {"ACCEPTED", "TRAINING_READY"} else None,
            species_name="草鱼" if status in {"ACCEPTED", "TRAINING_READY"} else None,
            status=status,
        )
    )
    db.commit()


def test_batch_filter_happens_before_limit_and_only_accepted_reviews_are_loaded(tmp_path: Path):
    db = _db(tmp_path)
    try:
        _add_reviewed(db, tmp_path, "BATCH_TARGET_001", "target_001")
        _add_reviewed(db, tmp_path, "BATCH_OTHER_001", "other_001")
        records = load_reviewed_crop_records(db, limit=1, batch_id="BATCH_TARGET_001")
        assert [record["image_id"] for record in records] == ["target_001"]
        assert records[0]["source_batch"] == "BATCH_TARGET_001"
        assert records[0]["accepted_bbox_json"]
    finally:
        db.close()


def test_candidate_or_pending_review_never_enters_crop_loader(tmp_path: Path):
    db = _db(tmp_path)
    try:
        _add_reviewed(db, tmp_path, "BATCH_PENDING_001", "pending_001", status="REVIEW_REQUIRED")
        assert load_reviewed_crop_records(db, batch_id="BATCH_PENDING_001") == []
    finally:
        db.close()


def test_crop_review_requires_explicit_bbox_and_species(tmp_path: Path):
    db = _db(tmp_path)
    try:
        _add_reviewed(db, tmp_path, "BATCH_REVIEW_001", "review_001", status="REVIEW_REQUIRED")
        with pytest.raises(HTTPException) as no_box:
            update_crop_review("BATCH_REVIEW_001", "review_001", CropReviewUpdate(decision="ACCEPTED"), db)
        assert no_box.value.detail["error"] == "ACCEPTED_BBOX_REQUIRED"
        with pytest.raises(HTTPException) as no_species:
            update_crop_review(
                "BATCH_REVIEW_001",
                "review_001",
                CropReviewUpdate(decision="ACCEPTED", accepted_bbox=[0.1, 0.1, 0.5, 0.5]),
                db,
            )
        assert no_species.value.detail["error"] == "SPECIES_REQUIRED"
        result = update_crop_review(
            "BATCH_REVIEW_001",
            "review_001",
            CropReviewUpdate(
                decision="ACCEPTED",
                accepted_bbox=[0.1, 0.1, 0.5, 0.5],
                species_key="grass_carp",
                species_name="草鱼",
            ),
            db,
        )
        assert result["status"] == "ACCEPTED"
        assert result["accepted_bbox"] == [0.1, 0.1, 0.5, 0.5]
        assert db.get(BatchCropReview, 1).status == "ACCEPTED"
    finally:
        db.close()


def test_crop_review_items_exposes_candidate_without_promoting_it(tmp_path: Path):
    db = _db(tmp_path)
    try:
        _add_reviewed(db, tmp_path, "BATCH_CANDIDATE_001", "candidate_001", status="REVIEW_REQUIRED")
        image = db.scalar(select(ImageAsset).where(ImageAsset.image_id == "candidate_001"))
        db.add(
            FishPresenceResult(
                image_asset_id=image.id,
                batch_id=image.batch_id,
                status="single_fish",
                fish_score=0.9,
                fish_count=1,
                evidence_json='{"objects":[{"name":"fish","score":0.9,"vertices":[{"x":0.1,"y":0.2},{"x":0.6,"y":0.2},{"x":0.6,"y":0.7},{"x":0.1,"y":0.7}]}]}',
            )
        )
        db.commit()
        result = crop_review_items("BATCH_CANDIDATE_001", db=db)
        assert result["items"][0]["candidate_bbox"] == pytest.approx([0.1, 0.2, 0.5, 0.5])
        assert result["items"][0]["accepted_bbox"] is None
    finally:
        db.close()


def test_build_request_rejects_whole_image_pipeline():
    payload = crop_dataset_api.CropDatasetBuildRequest(pipeline="WHOLE_IMAGE_V1")
    with pytest.raises(HTTPException) as caught:
        crop_dataset_api.build_crop_dataset_endpoint(payload, None)
    assert caught.value.status_code == 400
    assert caught.value.detail["error"] == "PIPELINE_NOT_SUPPORTED"


def test_validation_reports_new_when_no_staging_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CROP_DATASET_STAGING_ROOT", str(tmp_path / "var" / "crop_datasets"))
    result = crop_dataset_api.validate_crop_dataset_endpoint("DS_CROP_M1_v0.1")
    assert result["state"] == "NEW"
    assert result["manifest_path"].endswith("DS_CROP_M1_v0.1/metadata/crop_manifest.csv")
