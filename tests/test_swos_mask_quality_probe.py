from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.swos_mask_quality_probe import (
    build_multipart,
    decode_data_url,
    digest_bytes,
    summarize_payload,
    verify_expected_digest,
    verify_same_request_binding,
    write_mask_artifacts,
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


def test_multipart_can_target_product_inference_file_field():
    body, _ = build_multipart(b"jpeg-bytes", "fish.jpg", field_name="file")
    assert b'name="file"; filename="fish.jpg"' in body


def test_summary_does_not_promote_heuristic_mask_to_production():
    mask_png = b"mask bytes for review fixture"
    mask_sha = digest_bytes(mask_png, "sha256")
    checkpoint_sha = "b" * 64
    source_sha = "a" * 64
    payload = {
        "result_state": "RESULT_HIGH",
        "input": {"source_sha256": source_sha, "width": 1200, "height": 900, "format": "JPEG"},
        "detector": {
            "model_version": "DET_FISH_v0.3",
            "onnx_sha256": "c" * 64,
            "latency_ms": 22.1,
            "coordinate_space": "EXIF_TRANSPOSED_SOURCE_IMAGE_NORMALIZED",
            "detections": [{"bbox": {"x1": 0.1, "y1": 0.2, "x2": 0.9, "y2": 0.8}, "confidence": 0.92}],
            "quality_gate": {"status": "PASS", "primary_bbox": {"x1": 0.1, "y1": 0.2, "x2": 0.9, "y2": 0.8}},
        },
        "classifier": {"model_version": "CLASSIFIER_v1", "top1": {"species": "草鱼", "confidence": 0.91}},
        "timing": {"total_ms": 240.2, "segmentation_ms": 180.2},
        "segmentation": {
            "status": "REVIEW_READY",
            "geometry_trust_state": "REVIEW_PROXY",
            "model_version": "SAM_VIT_B",
            "checkpoint_sha256": checkpoint_sha,
            "source_image_sha256": source_sha,
            "quality": {"status": "GOOD", "reason": "heuristic_only", "mask_area_ratio": 0.2},
            "mask": {
                "encoding": "image/png",
                "data_base64": base64.b64encode(mask_png).decode("ascii"),
                "sha256": mask_sha,
                "width": 1200,
                "height": 900,
                "source_image_sha256": source_sha,
            },
        },
    }
    summary = summarize_payload(payload)
    assert summary["segmentation"]["executed"] is True
    assert summary["segmentation"]["quality"] == "GOOD"
    assert summary["segmentation"]["trust_state"] == "REVIEW_PROXY"
    assert summary["classifier"]["result_state"] == "RESULT_HIGH"
    assert summary["detector"]["model"] == "DET_FISH_v0.3"
    assert "production_qualified" not in summary["segmentation"]


def test_same_request_binding_rejects_mask_from_another_source():
    payload = {
        "input": {"source_sha256": "a" * 64},
        "segmentation": {
            "source_image_sha256": "a" * 64,
            "mask": {"source_image_sha256": "b" * 64},
        },
    }
    with pytest.raises(ValueError, match="mask source SHA"):
        verify_same_request_binding(payload, "a" * 64)


def test_same_request_binding_checks_bbox_and_normalized_dimensions():
    bbox = {"x1": 0.1, "y1": 0.2, "x2": 0.9, "y2": 0.8}
    source_sha = "a" * 64
    payload = {
        "input": {"source_sha256": source_sha, "width": 1200, "height": 900},
        "detector": {"quality_gate": {"primary_bbox": bbox}},
        "segmentation": {
            "source_image_sha256": source_sha,
            "bbox": dict(bbox),
            "mask": {
                "data_base64": "AQ==",
                "source_image_sha256": source_sha,
                "width": 1200,
                "height": 900,
            },
        },
    }
    verify_same_request_binding(payload, source_sha)
    payload["segmentation"]["mask"]["width"] = 900
    with pytest.raises(ValueError, match="dimensions"):
        verify_same_request_binding(payload, source_sha)


def test_mask_artifact_generation_checks_shared_source_dimensions(tmp_path):
    source = Image.new("RGB", (4, 3), (10, 20, 30))
    source_bytes = io.BytesIO()
    source.save(source_bytes, format="PNG")
    mask = Image.new("L", (4, 3), 255)
    mask_bytes = io.BytesIO()
    mask.save(mask_bytes, format="PNG")

    files = write_mask_artifacts(source_bytes.getvalue(), mask_bytes.getvalue(), tmp_path, "fixture")

    assert set(files) == {"mask", "overlay", "cutout"}
    for filename in files.values():
        assert (tmp_path / filename).is_file()
    with pytest.raises(ValueError, match="dimensions differ"):
        wrong = Image.new("L", (3, 3), 255)
        wrong_bytes = io.BytesIO()
        wrong.save(wrong_bytes, format="PNG")
        write_mask_artifacts(source_bytes.getvalue(), wrong_bytes.getvalue(), tmp_path, "bad")

