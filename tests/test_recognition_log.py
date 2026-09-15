from __future__ import annotations

import io
from types import SimpleNamespace

from PIL import Image

from app import inference_api
from app.detector_runtime import DetectorRun
from app.recognition_pipeline import BBox, Detection


def _jpeg() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (100, 200), (50, 70, 90)).save(output, format="JPEG")
    return output.getvalue()


def test_recognition_log_contains_full_pipeline_and_intermediates(monkeypatch):
    monkeypatch.setattr(
        inference_api,
        "_run_production_detector",
        lambda _image: DetectorRun(
            model_version="DET_FISH_v0.1",
            onnx_sha256="a" * 64,
            input_size=416,
            input_scale=2.08,
            input_draw_width=208,
            input_draw_height=416,
            latency_ms=3.2,
            detections=(Detection(0.92, BBox(0.0, 0.02, 0.75, 0.82)),),
        ),
    )
    monkeypatch.setattr(
        inference_api,
        "_classifier_prediction",
        lambda _row, _crop: {
            "model_status": "CANDIDATE",
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
        },
    )
    db = SimpleNamespace(
        get=lambda _model, version: SimpleNamespace(
            model_version=version,
            artifact_uri="gs://model.pt",
            pipeline_type="CROP_CLASSIFIER_V1",
        )
    )

    result = inference_api._predict_bytes(db, "MODEL_CROP_M1_v0.1", _jpeg())
    log = result["recognition_log"]

    assert log["schema_version"] == "recognition-log-v1"
    assert [stage["name"] for stage in log["stages"]] == [
        "input_validation",
        "image_load",
        "detector",
        "boundary_gate",
        "crop",
        "classifier",
    ]
    assert log["decision"]["classifier_allowed"] is True
    assert log["classifier"]["invoked"] is True
    assert log["classifier"]["top3"][0]["species"] == "草鱼"
    assert log["artifacts"]["detector_overlay"]["data_url"].startswith("data:image/png;base64,")
    assert log["artifacts"]["classifier_crop"]["data_url"].startswith("data:image/png;base64,")
    assert result["boundary_check"]["source_edge_near"] is True


def test_recognition_log_for_gate_block_records_skipped_stages(monkeypatch):
    monkeypatch.setattr(
        inference_api,
        "_run_production_detector",
        lambda _image: DetectorRun(
            model_version="DET_FISH_v0.1",
            onnx_sha256="a" * 64,
            input_size=416,
            input_scale=2.08,
            input_draw_width=208,
            input_draw_height=416,
            latency_ms=3.2,
            detections=(),
        ),
    )

    result = inference_api._predict_bytes(object(), "MODEL_CROP_M1_v0.1", _jpeg())
    stages = {stage["name"]: stage for stage in result["recognition_log"]["stages"]}

    assert result["status"] == "NO_FISH"
    assert stages["classifier"]["status"] == "SKIPPED"
    assert result["recognition_log"]["decision"]["hard_block"] is True


def test_model_testing_page_exposes_log_download():
    template = open("app/templates/inference.html", encoding="utf-8").read()
    for marker in ("下载识别日志", "downloadRecognitionLog", "new Blob", "detector_overlay", "classifier_crop"):
        assert marker in template
