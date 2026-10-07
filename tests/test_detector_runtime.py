from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from types import SimpleNamespace

from app.detector_runtime import (
    _prepare_yolox_input,
    decode_yolox_candidates,
    decode_yolox_output,
    detect,
    map_box_to_original,
    normalize_android_source,
)
from app.recognition_pipeline import BBox, Detection


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


def test_diagnostic_decoder_uses_zero_threshold_and_preserves_raw_top_k_without_nms():
    output = np.array(
        [[
            [100.0, 100.0, 80.0, 80.0, 0.2, 0.8],
            [101.0, 101.0, 80.0, 80.0, 0.8, 0.9],  # NMS would suppress the first row.
            [200.0, 200.0, 20.0, 20.0, 0.1, 0.9],
        ]],
        dtype=np.float32,
    )

    candidates = decode_yolox_candidates(
        output,
        scale=1.0,
        source_width=416,
        source_height=416,
        min_confidence=0.0,
        top_k=2,
    )

    assert [item.rank for item in candidates] == [1, 2]
    assert [item.confidence for item in candidates] == pytest.approx([0.72, 0.16])
    assert candidates[0].objectness == pytest.approx(0.8)
    assert candidates[0].fish_probability == pytest.approx(0.9)
    assert candidates[1].area_ratio == pytest.approx((80.0 / 416.0) ** 2)


@pytest.mark.parametrize(
    ("orientation", "expected"),
    [
        ("CW90", BBox(0.2, 0.5, 0.8, 0.9)),
        ("CCW90", BBox(0.2, 0.1, 0.8, 0.5)),
    ],
)
def test_rotation_box_mapping_returns_original_normalized_coordinates(orientation, expected):
    rotated = BBox(0.1, 0.2, 0.5, 0.8)

    actual = map_box_to_original(rotated, orientation)

    assert (actual.x1, actual.y1, actual.x2, actual.y2) == pytest.approx(
        (expected.x1, expected.y1, expected.x2, expected.y2)
    )
    assert actual.area_ratio == pytest.approx(rotated.area_ratio)


def _row_for_box(box, width, height, scale, confidence=0.9):
    x1, y1, x2, y2 = box
    return np.array(
        [[[
            (x1 + x2) * 0.5 * width * scale,
            (y1 + y2) * 0.5 * height * scale,
            (x2 - x1) * width * scale,
            (y2 - y1) * height * scale,
            confidence,
            1.0,
        ]]],
        dtype=np.float32,
    )


def test_detector_retries_only_no_fish_and_maps_rotated_detection_to_original(monkeypatch):
    original = Image.new("RGB", (100, 200), (10, 20, 30))
    empty = np.zeros((1, 1, 6), dtype=np.float32)
    rotated_box = (0.1, 0.2, 0.5, 0.8)
    rotated_row = _row_for_box(rotated_box, width=200, height=100, scale=2.08)
    outputs = [empty, empty, rotated_row]
    seen_shapes = []

    class FakeSession:
        def run(self, _outputs, feeds):
            tensor = feeds["images"]
            seen_shapes.append(tuple(tensor.shape))
            return [outputs[len(seen_shapes) - 1]]

    model = SimpleNamespace(
        model_version="DET_FISH_v0.1",
        onnx_sha256="a" * 64,
        input_size=416,
        input_name="images",
        session=FakeSession(),
    )
    monkeypatch.setattr("app.detector_runtime.load_detector", lambda: model)

    run = detect(original)

    assert len(seen_shapes) == 3
    assert [trace.orientation_attempt for trace in run.attempt_trace] == ["ORIGINAL", "CW90", "CCW90"]
    assert run.selected_attempt == "CCW90"
    assert run.original_width == 100 and run.original_height == 200
    assert run.onnx_sha256 == "a" * 64
    assert len(run.detections) == 1
    box = run.detections[0].box
    assert (box.x1, box.y1, box.x2, box.y2) == pytest.approx((0.2, 0.1, 0.8, 0.5))
    assert box.area_ratio == pytest.approx(0.24)
    assert run.detections[0].area_ratio >= 0.08


@pytest.mark.parametrize("confidence", [0.9, 0.25])
def test_detector_does_not_rotate_after_any_non_no_fish_assessment(monkeypatch, confidence):
    original = Image.new("RGB", (100, 200), (10, 20, 30))
    scale = 416 / 200
    candidate = _row_for_box((0.1, 0.1, 0.8, 0.8), 100, 200, scale, confidence)
    calls = 0

    class FakeSession:
        def run(self, _outputs, _feeds):
            nonlocal calls
            calls += 1
            return [candidate]

    model = SimpleNamespace(
        model_version="DET_FISH_v0.1",
        onnx_sha256="b" * 64,
        input_size=416,
        input_name="images",
        session=FakeSession(),
    )
    monkeypatch.setattr("app.detector_runtime.load_detector", lambda: model)

    run = detect(original)

    assert calls == 1
    assert run.selected_attempt == "ORIGINAL"
    assert [trace.orientation_attempt for trace in run.attempt_trace] == ["ORIGINAL"]
    box = run.detections[0].box
    assert (box.x1, box.y1, box.x2, box.y2) == pytest.approx((0.1, 0.1, 0.8, 0.8))


def test_detector_exhaustion_reports_none_without_changing_thresholds(monkeypatch):
    original = Image.new("RGB", (100, 200), (10, 20, 30))
    calls = 0

    class FakeSession:
        def run(self, _outputs, _feeds):
            nonlocal calls
            calls += 1
            return [np.zeros((1, 1, 6), dtype=np.float32)]

    model = SimpleNamespace(
        model_version="DET_FISH_v0.1",
        onnx_sha256="c" * 64,
        input_size=416,
        input_name="images",
        session=FakeSession(),
    )
    monkeypatch.setattr("app.detector_runtime.load_detector", lambda: model)

    run = detect(original)

    assert calls == 3
    assert run.selected_attempt == "NONE"
    assert run.orientation_attempt == "CCW90"
    assert run.detections == ()
    assert len(run.attempt_trace) == 3
