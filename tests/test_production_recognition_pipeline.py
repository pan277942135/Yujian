from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
from PIL import Image

from app import inference_api
from app.detector_runtime import DetectorRun
from app.recognition_pipeline import BBox, Detection


def _jpeg() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (100, 200), (50, 70, 90)).save(output, format="JPEG")
    return output.getvalue()


def _run(detections: tuple[Detection, ...]) -> DetectorRun:
    return DetectorRun(
        model_version="DET_FISH_v0.1",
        onnx_sha256="a" * 64,
        input_size=416,
        input_scale=2.08,
        input_draw_width=208,
        input_draw_height=416,
        latency_ms=3.2,
        detections=detections,
    )


def _det(confidence: float, x1: float, y1: float, x2: float, y2: float) -> Detection:
    return Detection(confidence=confidence, box=BBox(x1, y1, x2, y2))


@pytest.mark.parametrize(
    ("detections", "expected_status"),
    [
        ((), "NO_FISH"),
        ((_det(0.25, 0.2, 0.2, 0.8, 0.8),), "UNCERTAIN"),
        ((_det(0.9, 0.1, 0.2, 0.45, 0.7), _det(0.8, 0.55, 0.2, 0.9, 0.7)), "MULTIPLE_FISH"),
        ((_det(0.9, 0.4, 0.4, 0.58, 0.58),), "FISH_TOO_SMALL"),
    ],
)
def test_non_ready_status_never_invokes_classifier(monkeypatch, detections, expected_status):
    monkeypatch.setattr(inference_api, "_run_production_detector", lambda _image: _run(detections))
    monkeypatch.setattr(inference_api, "_classifier_prediction", lambda *_args: (_ for _ in ()).throw(AssertionError("classifier must not run")))

    result = inference_api._predict_bytes(object(), "MODEL_M1_v0.2", _jpeg())

    assert result["status"] == expected_status
    assert result["ready"] is False
    assert result["classification_ran"] is False
    assert "top1" not in result
    assert result["detector"]["model_version"] == "DET_FISH_v0.1"


def test_edge_adjacent_detection_reaches_classifier_and_emits_boundary_warning(monkeypatch):
    classifier_called = False
    monkeypatch.setattr(
        inference_api,
        "_run_production_detector",
        lambda _image: _run((_det(0.92, 0.0, 0.02, 0.75, 0.82),)),
    )

    def classifier(_row, crop):
        nonlocal classifier_called
        classifier_called = True
        assert crop.size[0] > 0 and crop.size[1] > 0
        return {
            "model_status": "PRODUCTION",
            "image_size": 224,
            "top1": {"species": "草鱼", "confidence": 0.82},
            "top3": [
                {"species": "草鱼", "confidence": 0.82},
                {"species": "鲤鱼", "confidence": 0.10},
                {"species": "鲫鱼", "confidence": 0.08},
            ],
            "low_confidence": False,
            "low_confidence_threshold": 0.55,
            "classifier_latency_ms": 2.1,
        }

    monkeypatch.setattr(inference_api, "_classifier_prediction", classifier)
    db = SimpleNamespace(
        get=lambda _model, version: SimpleNamespace(
            model_version=version,
            artifact_uri="gs://model.pt",
            pipeline_type="CROP_CLASSIFIER_V1",
        )
    )

    result = inference_api._predict_bytes(db, "MODEL_CROP_M1_v0.1", _jpeg())

    assert classifier_called is True
    assert result["status"] == "READY"
    assert result["ready"] is True
    assert result["classification_ran"] is True
    assert result["classifier_input"] == "crop"
    assert result["quality_gate"]["quality_status"] == "WARNING"
    assert result["quality_gate"]["hard_block"] is False
    assert result["quality_gate"]["classifier_allowed"] is True
    assert result["boundary_check"]["source_edge_near"] is True
    assert result["boundary_check"]["crop_edge_near"] is True
    assert result["boundary_check"]["hard_block"] is False
    assert result["boundary_check"]["reason"] == "SOURCE_EDGE_NEAR"
    assert result["reason"] != "primary_fish_bbox_touches_image_edge"


def test_invalid_bbox_blocks_classifier(monkeypatch):
    monkeypatch.setattr(
        inference_api,
        "_run_production_detector",
        lambda _image: _run((_det(0.92, 0.7, 0.2, 0.7, 0.82),)),
    )
    monkeypatch.setattr(
        inference_api,
        "_classifier_prediction",
        lambda *_args: (_ for _ in ()).throw(AssertionError("classifier must not run")),
    )

    result = inference_api._predict_bytes(object(), "MODEL_CROP_M1_v0.1", _jpeg())

    assert result["status"] == "INVALID_BBOX"
    assert result["ready"] is False
    assert result["classification_ran"] is False
    assert result["quality_gate"]["quality_status"] == "INVALID"
    assert result["quality_gate"]["hard_block"] is True
    assert result["quality_gate"]["classifier_allowed"] is False


def test_empty_crop_blocks_classifier(monkeypatch):
    monkeypatch.setattr(
        inference_api,
        "_run_production_detector",
        lambda _image: _run((_det(0.92, 0.2, 0.25, 0.8, 0.75),)),
    )
    monkeypatch.setattr(inference_api, "crop_box_pixels", lambda *_args: (0, 0, 0, 0))
    monkeypatch.setattr(
        inference_api,
        "_classifier_prediction",
        lambda *_args: (_ for _ in ()).throw(AssertionError("classifier must not run")),
    )

    result = inference_api._predict_bytes(object(), "MODEL_CROP_M1_v0.1", _jpeg())

    assert result["status"] == "EMPTY_CROP"
    assert result["ready"] is False
    assert result["classification_ran"] is False
    assert result["quality_gate"]["quality_status"] == "INVALID"
    assert result["quality_gate"]["hard_block"] is True
    assert result["quality_gate"]["classifier_allowed"] is False
    assert result["reason"] == "empty_crop"


def test_ready_detection_uses_expanded_floor_ceil_crop_before_classifier(monkeypatch):
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        inference_api,
        "_run_production_detector",
        lambda _image: _run((_det(0.92, 0.2, 0.25, 0.8, 0.75),)),
    )

    def classifier(row, crop):
        seen["row"] = row
        seen["size"] = crop.size
        return {
            "model_status": "PRODUCTION",
            "image_size": 224,
            "top1": {"species": "草鱼", "confidence": 0.9},
            "top3": [{"species": "草鱼", "confidence": 0.9}],
            "low_confidence": False,
            "low_confidence_threshold": 0.55,
            "classifier_latency_ms": 2.1,
        }

    monkeypatch.setattr(inference_api, "_classifier_prediction", classifier)
    db = SimpleNamespace(get=lambda _model, version: SimpleNamespace(model_version=version, artifact_uri="gs://model.pt", pipeline_type="CROP_CLASSIFIER_V1"))

    result = inference_api._predict_bytes(db, "MODEL_M1_v0.2", _jpeg())

    assert result["status"] == "READY"
    assert result["ready"] is True
    assert result["classification_ran"] is True
    assert result["crop"]["pixels"] == {"left": 11, "top": 35, "right": 89, "bottom": 165, "width": 78, "height": 130}
    assert seen["size"] == (78, 130)
    assert result["classifier_input"] == "crop"


def test_legacy_whole_image_model_keeps_original_classifier_input(monkeypatch):
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        inference_api,
        "_run_production_detector",
        lambda _image: _run((_det(0.92, 0.2, 0.25, 0.8, 0.75),)),
    )

    def classifier(_row, source):
        seen["size"] = source.size
        return {
            "model_status": "PRODUCTION",
            "image_size": 224,
            "top1": {"species": "草鱼", "confidence": 0.9},
            "top3": [{"species": "草鱼", "confidence": 0.9}],
            "low_confidence": False,
            "low_confidence_threshold": 0.55,
            "classifier_latency_ms": 2.1,
        }

    monkeypatch.setattr(inference_api, "_classifier_prediction", classifier)
    db = SimpleNamespace(
        get=lambda _model, version: SimpleNamespace(
            model_version=version,
            artifact_uri="gs://model.pt",
            pipeline_type="WHOLE_IMAGE_V1",
        )
    )

    result = inference_api._predict_bytes(db, "MODEL_M1_v0.2", _jpeg())

    assert result["pipeline_type"] == "WHOLE_IMAGE_V1"
    assert result["classifier_input"] == "original"
    assert seen["size"] == (100, 200)
