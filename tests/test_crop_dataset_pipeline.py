from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import DatasetVersion
from trainer.crop_dataset_pipeline import (
    CROP_DATASET_VERSION,
    CROP_INPUT_TYPE,
    CROP_PIPELINE_TYPE,
    build_reviewed_crop_dataset,
    freeze_crop_dataset,
)


def _image_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (100, 80), (40, 100, 160)).save(output, format="JPEG")
    return output.getvalue()


def _record(image_id: str, status: str = "ACCEPTED", species: str = "grass_carp") -> dict:
    return {
        "image_id": image_id,
        "status": status,
        "source_image": f"gs://bucket/original/{image_id}.jpg",
        "accepted_species_key": species,
        "accepted_species_name": "草鱼" if species == "grass_carp" else "鲤鱼",
        "accepted_bbox": [0.1, 0.15, 0.65, 0.65],
        "detection": {"candidate_bbox": [0.8, 0.8, 0.1, 0.1]},
    }


def test_production_crop_builder_is_accepted_only_and_has_required_contract(tmp_path: Path):
    report = build_reviewed_crop_dataset(
        [_record("yj_img_001"), _record("yj_img_candidate", "CANDIDATE")],
        tmp_path / CROP_DATASET_VERSION,
        image_loader=lambda _uri: _image_bytes(),
    )
    assert report["written"] == 1
    assert report["excluded"]["candidate_excluded"] == 1
    assert report["validation"]["valid"] is True

    manifest = tmp_path / CROP_DATASET_VERSION / "metadata" / "crop_manifest.csv"
    row = next(csv.DictReader(manifest.open(encoding="utf-8")))
    required = {
        "image_id",
        "source_image_id",
        "crop_image_path",
        "species_key",
        "species_name",
        "accepted_bbox",
        "expand_ratio",
        "crop_width",
        "crop_height",
        "review_status",
        "created_at",
    }
    assert required <= set(row)
    assert row["input_type"] == CROP_INPUT_TYPE
    assert row["pipeline_type"] == CROP_PIPELINE_TYPE
    assert row["bbox_source"] == "accepted_review"
    assert row["source_image_id"] == "yj_img_001"
    assert (tmp_path / CROP_DATASET_VERSION / row["crop_image_path"]).is_file()


def test_freeze_writes_metadata_and_registers_ready_for_training(tmp_path: Path):
    root = tmp_path / CROP_DATASET_VERSION
    build_reviewed_crop_dataset(
        [_record("yj_img_002", species="grass_carp"), _record("yj_img_003", species="common_carp")],
        root,
        image_loader=lambda _uri: _image_bytes(),
    )
    engine = create_engine(f"sqlite:///{tmp_path / 'registry.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    try:
        result = freeze_crop_dataset(
            root,
            dataset_version=CROP_DATASET_VERSION,
            db=db,
            git_commit="test-sha",
        )
        assert result["status"] == "READY_FOR_TRAINING"
        assert result["source"] == "accepted_bbox"
        assert result["pipeline"] == CROP_PIPELINE_TYPE
        assert result["image_count"] == 2
        metadata = result["freeze_metadata"]
        assert metadata["input_type"] == CROP_INPUT_TYPE
        assert metadata["crop_expand_ratio"] == 0.15
        assert (root / "metadata" / "dataset.json").is_file()
        assert (root / "metadata" / "freeze_report.json").is_file()
        row = db.get(DatasetVersion, CROP_DATASET_VERSION)
        assert row is not None
        assert row.status == "READY_FOR_TRAINING"
        assert row.pipeline_type == CROP_PIPELINE_TYPE
        assert json.loads(row.metadata_json)["source"] == "accepted_bbox"
    finally:
        db.close()
