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
        ((_det(0.9, 0.0, 0.2, 0.75, 0.8),), "INCOMPLETE_FISH"),
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
    db = SimpleNamespace(get=lambda _model, version: SimpleNamespace(model_version=version, artifact_uri="gs://model.pt"))

    result = inference_api._predict_bytes(db, "MODEL_M1_v0.2", _jpeg())

    assert result["status"] == "READY"
    assert result["ready"] is True
    assert result["classification_ran"] is True
    assert result["crop"]["pixels"] == {"left": 11, "top": 35, "right": 89, "bottom": 165, "width": 78, "height": 130}
    assert seen["size"] == (78, 130)
