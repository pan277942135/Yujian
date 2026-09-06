from io import BytesIO
import asyncio

import numpy as np
from fastapi import UploadFile
from PIL import Image

from app import segmentation_api
from app.recognition_pipeline import BBox, Detection
from app.segmentation.quality_gate import SegmentationQuality, assess_mask


def test_quality_gate_marks_empty_mask_invalid():
    quality, metrics = assess_mask(np.zeros((20, 20), dtype=bool), (2, 2, 18, 18))
    assert quality is SegmentationQuality.INVALID
    assert metrics["reason"] == "empty_mask"


def test_quality_gate_accepts_compact_subject():
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:70, 25:75] = True
    quality, metrics = assess_mask(mask, (20, 20, 80, 80))
    assert quality is SegmentationQuality.GOOD
    assert 0.5 < metrics["mask_area_ratio"] < 0.8


def test_segmentation_api_uses_detector_primary_bbox(monkeypatch):
    image = Image.new("RGB", (100, 50), (20, 30, 40))
    detector = type("DetectorRun", (), {
        "model_version": "DET_FISH_v0.1",
        "latency_ms": 1.2,
        "detections": (Detection(0.9, BBox(0.1, 0.2, 0.8, 0.9)),),
    })()
    segmentation = type("SegmentationResult", (), {
        "quality": SegmentationQuality.GOOD,
        "reason": "mask_passed_basic_checks",
        "width": 100,
        "height": 50,
        "mask_area_ratio": 0.55,
        "edge_ratio": 0.1,
        "connected_components": 1,
        "processing_ms": 2.3,
        "mask": np.ones((50, 100), dtype=bool),
        "cutout_png": b"png",
    })()
    monkeypatch.setattr(segmentation_api, "detect", lambda _image: detector)
    monkeypatch.setattr(segmentation_api, "generate_fish_cutout", lambda _image, bbox: segmentation)
    payload = asyncio.run(segmentation_api.fish_segmentation(UploadFile(filename="fish.png", file=BytesIO(_png(image)), headers={"content-type": "image/png"})))
    assert payload["detector"]["bbox_normalized"] == [0.1, 0.2, 0.8, 0.9]
    assert payload["segmentation"]["prompt"] == "DET_FISH_v0.1 primary bbox"
    assert payload["transparent_fish"].startswith("data:image/png;base64,")


def _png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()
