from __future__ import annotations

from io import BytesIO

from PIL import Image

from app.crop_audit import AUDIT_VERSION, audit_review_record, compare_crop_bytes
from app.crop_contract import canonical_crop


def _source_with_exif_orientation() -> bytes:
    image = Image.new("RGB", (101, 67))
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            pixels[x, y] = ((x * 3) % 256, (y * 5) % 256, (x + y) % 256)
    exif = Image.Exif()
    exif[274] = 6
    output = BytesIO()
    image.save(output, format="JPEG", quality=95, exif=exif)
    return output.getvalue()


def _legacy_review_crop(source_bytes: bytes, bbox: list[float], expand_ratio: float = 0.15) -> bytes:
    """Reproduce the pre-C.5-C.2 review behavior: no EXIF transpose + round right/bottom."""

    with Image.open(BytesIO(source_bytes)) as source:
        image = source.convert("RGB")
    x, y, width, height = bbox
    left = max(0.0, x - width * expand_ratio)
    top = max(0.0, y - height * expand_ratio)
    right = min(1.0, x + width + width * expand_ratio)
    bottom = min(1.0, y + height + height * expand_ratio)
    pixel_box = (
        max(0, int(left * image.width)),
        max(0, int(top * image.height)),
        min(image.width, max(1, int(round(right * image.width)))),
        min(image.height, max(1, int(round(bottom * image.height)))),
    )
    output = BytesIO()
    image.crop(pixel_box).save(output, format="JPEG", quality=92)
    return output.getvalue()


def test_compare_crop_bytes_matches_canonical_artifact() -> None:
    source = _source_with_exif_orientation()
    bbox = [0.173, 0.219, 0.417, 0.381]
    canonical = canonical_crop(source, bbox)

    result = compare_crop_bytes(source, canonical.jpeg_bytes, bbox)

    assert result["audit_status"] == "MATCH"
    assert result["rebuild_required"] is False
    assert result["existing_sha256"] == result["canonical_sha256"]


def test_compare_crop_bytes_flags_pre_contract_review_crop() -> None:
    source = _source_with_exif_orientation()
    bbox = [0.173, 0.219, 0.417, 0.381]
    legacy = _legacy_review_crop(source, bbox)

    result = compare_crop_bytes(source, legacy, bbox)

    assert result["audit_status"] == "REBUILD_REQUIRED"
    assert result["rebuild_required"] is True
    assert result["existing_sha256"] != result["canonical_sha256"]


def test_audit_review_record_classifies_missing_crop_without_writes() -> None:
    calls: list[str] = []

    def loader(uri: str):
        calls.append(uri)
        raise AssertionError("loader must not be called when crop_uri is missing")

    record = {
        "source_dataset_version": "DS_M1_v0.5",
        "image_id": "fish-001",
        "review_status": "ACCEPTED",
        "crop_status": "NOT_GENERATED",
        "accepted_bbox_json": "[0.1,0.2,0.5,0.4]",
        "source_image_gcs_uri": "gs://bucket/source.jpg",
        "crop_uri": None,
    }

    result = audit_review_record(record, loader)

    assert result["audit_status"] == "MISSING_CROP"
    assert result["rebuild_required"] is True
    assert result["manual_fix_required"] is False
    assert calls == []


def test_crop_audit_router_is_registered() -> None:
    from app.entry import app

    assert AUDIT_VERSION == "HISTORICAL_CROP_AUDIT_V1"
    assert "/api/crop-audit/historical" in app.openapi()["paths"]
