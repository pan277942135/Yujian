from __future__ import annotations

import csv
import io
import json

from PIL import Image

from trainer.build_reviewed_datasets import build_crop_dataset, build_reviewed_detector_dataset


def _image_loader(_uri: str) -> bytes:
    image = Image.new("RGB", (100, 80), (80, 120, 160))
    out = io.BytesIO()
    image.save(out, format="JPEG")
    return out.getvalue()


def _record(status: str = "ACCEPTED") -> dict:
    return {
        "image_id": "yj_img_dataset_001",
        "status": status,
        "accepted_bbox": [0.1, 0.2, 0.5, 0.5] if status == "ACCEPTED" else None,
        "accepted_species": "grass_carp",
        "image_gcs_uri": "gs://test-bucket/app_feedback/inference/image.jpg",
        "detection": {"candidate_bbox": [0.9, 0.9, 0.05, 0.05]},
    }


def test_detector_dataset_uses_only_accepted_bbox(tmp_path):
    report = build_reviewed_detector_dataset(
        [_record("CANDIDATE"), _record("ACCEPTED")],
        tmp_path / "detector",
        image_loader=_image_loader,
    )
    assert report["written"] == 1
    assert report["excluded"]["candidate_excluded"] == 1
    label = next((tmp_path / "detector" / "labels").rglob("*.txt"))
    assert label.read_text(encoding="utf-8").startswith("0 0.350000 0.450000 0.500000 0.500000")
    manifest = list(csv.DictReader((tmp_path / "detector" / "metadata" / "manifest.csv").open(encoding="utf-8")))
    assert manifest[0]["bbox_source"] == "accepted_review"
    assert json.loads((tmp_path / "detector" / "metadata" / "report.json").read_text(encoding="utf-8"))["safety"]["candidate_bbox_used_as_label"] is False


def test_crop_dataset_regenerates_crop_and_never_reads_original_as_classifier_input(tmp_path):
    report = build_crop_dataset([_record("ACCEPTED")], tmp_path / "crop", image_loader=_image_loader)
    assert report["written"] == 1
    crop = next((tmp_path / "crop" / "images").rglob("*.jpg"))
    with Image.open(crop) as image:
        assert image.size[0] > 50
        assert image.size[1] > 40
    row = next(csv.DictReader((tmp_path / "crop" / "metadata" / "crop_manifest.csv").open(encoding="utf-8")))
    assert row["input_type"] == "crop"
    assert row["pipeline_type"] == "CROP_CLASSIFIER_V1"
    assert report["safety"]["original_images_used_for_classifier"] is False
