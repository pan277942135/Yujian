from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from scripts.swos_mask_quality_probe import (
    build_multipart,
    decode_data_url,
    digest_bytes,
    summarize_payload,
    verify_expected_digest,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests/fixtures/swos_resource_quality/photo_manifest_v1.json"


def test_public_photo_manifest_is_attributed_and_source_pinned():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sources = manifest["sources"]
    assert len(sources) >= 4
    assert len({source["id"] for source in sources}) == len(sources)
    for source in sources:
        assert source["page_url"].startswith("https://commons.wikimedia.org/wiki/File:")
        assert source["download_url"].startswith("https://commons.wikimedia.org/wiki/Special:Redirect/file/")
        assert source["attribution"]
        assert source["license"] in {"CC BY 4.0", "CC BY-SA 3.0", "CC BY-SA 4.0"}
        assert source["license_url"].startswith("https://creativecommons.org/licenses/")
        assert source["expected_digest"]["algorithm"] in {"sha1", "sha256"}
        assert len(source["expected_digest"]["value"]) == (40 if source["expected_digest"]["algorithm"] == "sha1" else 64)


def test_digest_verification_uses_exact_source_bytes():
    raw = b"fish-photo-fixture"
    expected = {"algorithm": "sha256", "value": digest_bytes(raw, "sha256")}
    verified = verify_expected_digest(raw, expected)
    assert verified["verified"] == "true"
    assert verified["actual"] == expected["value"]
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_expected_digest(raw, {"algorithm": "sha256", "value": "0" * 64})


def test_data_url_decoding_rejects_non_base64_and_decodes_png():
    raw = b"png-test-bytes"
    value = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    assert decode_data_url(value) == raw
    with pytest.raises(ValueError):
        decode_data_url("https://example.invalid/image.png")


def test_multipart_contains_image_field_and_closing_boundary():
    body, content_type = build_multipart(b"jpeg-bytes", "fish.jpg")
    boundary = content_type.split("boundary=", 1)[1]
    assert b'name="image"; filename="fish.jpg"' in body
    assert b"Content-Type: image/jpeg" in body
    assert body.endswith(f"--{boundary}--\r\n".encode())


def test_summary_does_not_promote_heuristic_mask_to_production():
    payload = {
        "runtime": {"commit": "a" * 40, "revision": "candidate-001", "segmentation_model": "SAM_VIT_B"},
        "input": {"image_width": 1200, "image_height": 900, "image_id": "abc"},
        "detector": {"detections_count": 1, "primary_detection_found": True, "assessment": "ready", "detector_processing_ms": 22.1},
        "eligibility": {"single_fish_hero_evaluation": True},
        "segmentation": {"executed": True, "quality": "GOOD", "mask_width": 1200, "mask_height": 900, "processing_ms": 180.2},
    }
    summary = summarize_payload(payload)
    assert summary["segmentation"]["executed"] is True
    assert summary["segmentation"]["quality"] == "GOOD"
    assert summary["runtime"]["commit"] == "a" * 40
    assert "production_qualified" not in summary["segmentation"]

