from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.detector_runtime import _prepare_yolox_input, decode_yolox_output, normalize_android_source


def test_yolox_preprocess_is_bgr_top_left_letterbox_with_114_padding():
    image = Image.new("RGB", (100, 50), (10, 20, 30))

    tensor, scale, draw_width, draw_height = _prepare_yolox_input(image, 416)

    assert tensor.shape == (1, 3, 416, 416)
    assert tensor.dtype == np.float32
    assert (scale, draw_width, draw_height) == (4.16, 416, 208)
    # First source pixel is RGB(10, 20, 30), so the detector receives BGR(30, 20, 10).
    assert tensor[0, :, 0, 0].tolist() == [30.0, 20.0, 10.0]
    assert tensor[0, :, 300, 0].tolist() == [114.0, 114.0, 114.0]


def test_android_source_normalization_applies_max_dimension_before_detector():
    image = Image.new("RGB", (4097, 2049), (10, 20, 30))
    normalized = normalize_android_source(image)
    try:
        # Android doubles inSampleSize until 4097 / sample <= 2048: sample=4.
        assert normalized.size == (1024, 512)
        assert normalized.mode == "RGB"
    finally:
        normalized.close()


def test_yolox_decode_maps_boxes_back_to_source_and_applies_contract_nms():
    output = np.array(
        [
            [
                [208.0, 208.0, 208.0, 208.0, 0.9, 0.9],
                [210.0, 210.0, 208.0, 208.0, 0.95, 0.8],  # overlaps first: suppressed by NMS
                [80.0, 80.0, 80.0, 80.0, 0.9, 0.8],
                [300.0, 300.0, 20.0, 20.0, 0.1, 0.9],  # below weak confidence: excluded
            ]
        ],
        dtype=np.float32,
    )

    detections = decode_yolox_output(
        output,
        scale=1.0,
        source_width=416,
        source_height=416,
        nms_iou=0.45,
        min_confidence=0.20,
    )

    assert len(detections) == 2
    assert detections[0].confidence == pytest.approx(0.81)
    assert detections[0].box.x1 == 0.25
    assert detections[0].box.y2 == 0.75
    assert round(detections[1].box.area_ratio, 6) == round((80.0 / 416.0) ** 2, 6)
