from io import BytesIO
import asyncio

from fastapi import UploadFile
from PIL import Image

from app import detector_parity_api
from app.detector_runtime import DetectorRun
from app.recognition_pipeline import BBox, Detection


def test_detector_parity_returns_real_pipeline_metadata_and_overlay(monkeypatch):
    image = Image.new("RGB", (100, 50), (20, 30, 40))
    run = DetectorRun(
        model_version="DET_FISH_v0.1", onnx_sha256="a" * 64, input_size=416,
        input_scale=4.16, input_draw_width=416, input_draw_height=208,
        latency_ms=1.2, detections=(Detection(0.876543, BBox(.1, .2, .8, .9)),),
    )
    monkeypatch.setattr(detector_parity_api, "detect", lambda _image: run)
    payload = asyncio.run(
        detector_parity_api.detector_parity(
            UploadFile(filename="fish.png", file=BytesIO(_png(image)), headers={"content-type": "image/png"})
        )
    )
    assert payload["model_version"] == "DET_FISH_v0.1"
    assert payload["image"]["width"] == 100
    assert payload["detector"]["confidence"] == 0.876543
    assert payload["detector"]["bbox_pixel"] == [10, 10, 80, 45]
    assert payload["detector"]["bbox_normalized"] == [0.1, 0.2, 0.7, 0.7]
    assert payload["preprocess"]["letterbox"]["fill"] == 114
    assert payload["overlay"]["data_url"].startswith("data:image/png;base64,")


def test_detector_parity_upload_field_is_image():
    from app.entry import app

    request_body = app.openapi()["paths"]["/api/debug/detector-parity"]["post"]["requestBody"]
    schema_ref = request_body["content"]["multipart/form-data"]["schema"]["$ref"]
    schema = app.openapi()["components"]["schemas"][schema_ref.rsplit("/", 1)[-1]]
    assert "image" in schema["properties"]


def _png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()
