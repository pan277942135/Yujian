from __future__ import annotations

import csv
import hashlib
from io import BytesIO

from PIL import Image

from app.crop_contract import canonical_crop
from app.dataset_crop_review import _crop_preview_bytes
from trainer.build_reviewed_datasets import build_crop_dataset


ACCEPTED_BBOX = [0.13, 0.11, 0.31, 0.43]
EXPECTED_PIXEL_BOX_AFTER_EXIF = (5, 4, 33, 62)


def _oriented_source_bytes() -> bytes:
    image = Image.new("RGB", (101, 67))
    for x in range(image.width):
        for y in range(image.height):
            image.putpixel((x, y), ((x * 3) % 256, (y * 5) % 256, ((x + y) * 7) % 256))
    exif = Image.Exif()
    exif[274] = 6  # Rotate 90 degrees clockwise when normalized.
    output = BytesIO()
    image.save(output, format="JPEG", quality=95, exif=exif)
    return output.getvalue()


def test_canonical_crop_applies_exif_before_pixel_box_math() -> None:
    result = canonical_crop(_oriented_source_bytes(), ACCEPTED_BBOX)

    assert result.pixel_box == EXPECTED_PIXEL_BOX_AFTER_EXIF
    assert (result.width, result.height) == (28, 58)


def test_review_preview_and_training_crop_are_byte_identical(tmp_path) -> None:
    source_bytes = _oriented_source_bytes()
    source_path = tmp_path / "source.jpg"
    source_path.write_bytes(source_bytes)
    dataset_root = tmp_path / "crop_dataset"

    review_bytes = _crop_preview_bytes(source_bytes, ACCEPTED_BBOX)
    report = build_crop_dataset(
        [
            {
                "image_id": "sample-001",
                "source_image_id": "sample-001",
                "source_image_path": str(source_path),
                "status": "ACCEPTED",
                "accepted_bbox": ACCEPTED_BBOX,
                "accepted_species_key": "carp",
                "accepted_species_name": "Carp",
            }
        ],
        dataset_root,
    )

    assert report["written"] == 1
    assert report["failures"] == []

    training_path = dataset_root / "images" / "carp" / "sample-001_crop.jpg"
    training_bytes = training_path.read_bytes()
    assert review_bytes == training_bytes
    assert hashlib.sha256(review_bytes).hexdigest() == hashlib.sha256(training_bytes).hexdigest()

    with (dataset_root / "metadata" / "crop_manifest.csv").open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert tuple(int(row[key]) for key in ("crop_left", "crop_top", "crop_right", "crop_bottom")) == EXPECTED_PIXEL_BOX_AFTER_EXIF
    assert (int(row["crop_width"]), int(row["crop_height"])) == (28, 58)
