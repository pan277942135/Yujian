from io import BytesIO
import asyncio
import base64

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
    assert payload["classifier_crop"]["data_url"].startswith("data:image/png;base64,")
    assert payload["classifier_crop"]["pixels"] == [0, 4, 91, 50]


def test_detector_parity_upload_field_is_image():
    from app.entry import app

    request_body = app.openapi()["paths"]["/api/debug/detector-parity"]["post"]["requestBody"]
    schema_ref = request_body["content"]["multipart/form-data"]["schema"]["$ref"]
    schema = app.openapi()["components"]["schemas"][schema_ref.rsplit("/", 1)[-1]]
    assert "image" in schema["properties"]


def test_all_detector_boxes_are_red_and_one_step_thicker():
    red = (220, 38, 38)
    for size in ((1000, 600), (2500, 600)):
        source = Image.new("RGB", size, (20, 30, 40))
        detections = (
            Detection(0.91, BBox(0.1, 0.2, 0.4, 0.8)),
            Detection(0.76, BBox(0.55, 0.2, 0.9, 0.8)),
        )
        data_url = detector_parity_api._overlay(source, detections, detections[0])
        result = Image.open(BytesIO(base64.b64decode(data_url.split(",", 1)[1])))
        expected_width = max(4, source.width // 500 + 1)

        for detection in detections:
            x1, y1, _, _ = detector_parity_api._pixel_box(detection.box, source.width, source.height)
            for offset in range(expected_width):
                assert result.getpixel((x1 + 80, y1 + offset)) == red
            assert result.getpixel((x1 + 80, y1 + expected_width)) == (20, 30, 40)

        assert (245, 158, 11) not in {color for _, color in result.getcolors(result.width * result.height) or []}


def test_det_fish_test_previews_preserve_image_ratio_and_collapse_on_narrow_screens():
    from pathlib import Path

    template = Path("app/templates/platform/det_fish_test.html").read_text(encoding="utf-8")

    assert 'class="form-grid det-fish-test-grid"' in template
    assert template.count("bbox-preview det-fish-image-preview") == 2
    assert "overlay.src=data.overlay_data_url;" in template
    assert ".det-fish-image-preview { aspect-ratio: auto;" in template
    assert "max-width: 100%;" in template
    assert "max-height: 70vh;" in template
    assert "object-fit: contain;" in template
    assert "width: auto;" in template
    assert "height: auto;" in template
    assert ".det-fish-image-preview img[hidden] { display: none; }" in template
    assert "@media (max-width: 780px)" in template
    assert "grid-template-columns: minmax(0, 1fr);" in template


def _png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()
