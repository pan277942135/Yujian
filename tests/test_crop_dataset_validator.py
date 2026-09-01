from __future__ import annotations

import csv
import io
import json
from types import SimpleNamespace

from PIL import Image

from trainer.build_reviewed_datasets import build_crop_dataset
from trainer.crop_dataset_validator import validate_crop_dataset, validate_crop_manifest, validate_crop_rows


def _jpeg_bytes() -> bytes:
    image = Image.new("RGB", (64, 48), (60, 100, 140))
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


def _row(path: str = "images/grass_carp/yj_img_001_crop.jpg", image_id: str = "yj_img_001") -> dict[str, str]:
    return {
        "image_id": image_id,
        "species_key": "grass_carp",
        "local_path": path,
        "input_type": "crop",
        "pipeline_type": "CROP_CLASSIFIER_V1",
        "source_image_id": image_id,
        "accepted_bbox": "[0.1, 0.2, 0.5, 0.5]",
    }


def test_validator_accepts_reviewed_crop_and_checks_local_file(tmp_path):
    crop = tmp_path / "images" / "grass_carp" / "yj_img_001_crop.jpg"
    crop.parent.mkdir(parents=True)
    crop.write_bytes(_jpeg_bytes())

    report = validate_crop_rows([_row()], tmp_path)

    assert report["valid"] is True
    assert report["checks"] == {
        "crop_exists": True,
        "species_present": True,
        "accepted_bbox_present": True,
        "image_id_unique": True,
    }


def test_validator_rejects_duplicate_and_missing_contract_fields(tmp_path):
    crop = tmp_path / "crop.jpg"
    crop.write_bytes(_jpeg_bytes())
    first = _row(path="crop.jpg")
    second = _row(path="crop.jpg")
    second["species_key"] = ""
    second["accepted_bbox"] = ""
    second["candidate_bbox"] = "[0.1,0.1,0.4,0.4]"

    report = validate_crop_rows([first, second], tmp_path)
    codes = {error["code"] for error in report["errors"]}

    assert report["valid"] is False
    assert {"DUPLICATE_IMAGE_ID", "MISSING_SPECIES", "MISSING_ACCEPTED_BBOX"} <= codes
    assert report["safety"]["candidate_bbox_accepted"] is False


def test_validator_rejects_original_input_and_missing_crop(tmp_path):
    row = _row(path="missing.jpg")
    row["input_type"] = "original"
    report = validate_crop_rows([row], tmp_path)
    codes = {error["code"] for error in report["errors"]}
    assert {"ORIGINAL_INPUT_FORBIDDEN", "CROP_NOT_FOUND"} <= codes


def test_validator_prefers_crop_reference_and_rejects_source_reference(tmp_path):
    original = tmp_path / "original.jpg"
    crop = tmp_path / "crop.jpg"
    original.write_bytes(_jpeg_bytes())
    crop.write_bytes(_jpeg_bytes())
    row = _row(path="original.jpg")
    row.update({"crop_path": "crop.jpg", "source_image_path": "original.jpg"})

    report = validate_crop_rows([row], tmp_path)

    assert report["valid"] is True
    assert report["checks"]["crop_exists"] is True

    row["crop_path"] = "original.jpg"
    report = validate_crop_rows([row], tmp_path)
    assert any(error["code"] == "ORIGINAL_INPUT_FORBIDDEN" for error in report["errors"])


def test_crop_builder_writes_provenance_metadata_and_validation(tmp_path):
    def loader(_uri: str) -> bytes:
        return _jpeg_bytes()

    records = [
        {
            "image_id": "yj_img_002",
            "status": "ACCEPTED",
            "accepted_bbox": [0.1, 0.2, 0.5, 0.5],
            "accepted_species": "grass_carp",
            "image_gcs_uri": "gs://bucket/original.jpg",
            "detection": {"candidate_bbox": [0.8, 0.8, 0.1, 0.1]},
        }
    ]
    root = tmp_path / "dataset"
    report = build_crop_dataset(records, root, image_loader=loader)
    manifest_path = root / "metadata" / "crop_manifest.csv"
    row = next(csv.DictReader(manifest_path.open(encoding="utf-8")))

    assert report["written"] == 1
    assert report["validation"]["valid"] is True
    assert row["source_image_id"] == "yj_img_002"
    assert row["bbox_source"] == "accepted_review"
    assert int(row["crop_width"]) > 0
    assert int(row["crop_height"]) > 0
    assert validate_crop_dataset(root)["valid"] is True
    assert validate_crop_manifest(manifest_path)["valid"] is True
    saved_report = json.loads((root / "metadata" / "report.json").read_text(encoding="utf-8"))
    assert saved_report["safety"]["candidate_bbox_used"] is False


def test_crop_qa_item_only_serializes_reviewed_assets():
    from app.crop_qa import crop_qa_item

    candidate = SimpleNamespace(status="CANDIDATE", accepted_bbox_json=None, image_id="yj_img_003")
    assert crop_qa_item(candidate) is None

    accepted = SimpleNamespace(
        status="ACCEPTED",
        accepted_bbox_json="[0.1, 0.2, 0.5, 0.5]",
        accepted_species="grass_carp",
        image_id="yj_img_004",
        record_gcs_uri="",
        crop_gcs_uri="gs://bucket/crop.jpg",
    )
    item = crop_qa_item(accepted)
    assert item is not None
    assert item["bbox_source"] == "human_review"
    assert item["source_image_id"] == "yj_img_004"
    assert item["crop_available"] is True
