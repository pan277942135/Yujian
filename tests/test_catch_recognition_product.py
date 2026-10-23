from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from PIL import Image
from starlette.datastructures import UploadFile

from app import catch_recognition_api as product_api
from app.segmentation import product_adapter
from app.segmentation.quality_gate import SegmentationQuality
from app.segmentation.service import FishSegmentationResult


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (4, 3), color=(35, 90, 70)).save(output, format="PNG")
    return output.getvalue()


def _pipeline_result() -> dict:
    return {
        "pipeline_version": "CROP_CLASSIFIER_V1",
        "input": {"width": 4, "height": 3, "format": "PNG"},
        "status": "SUCCESS",
        "status_wire": "success",
        "message": "识别完成",
        "guidance": None,
        "detector": {
            "model_version": "DET_FISH_v0.3",
            "onnx_sha256": "detector-sha",
            "latency_ms": 20,
            "detections": [{"bbox": {"x1": 0.1, "y1": 0.2, "x2": 0.8, "y2": 0.9}, "confidence": 0.91}],
        },
        "quality_gate": {"status": "PASS"},
        "crop": {"applied": True, "bbox": {"x1": 0.1, "y1": 0.2, "x2": 0.8, "y2": 0.9}},
        "classification_ran": True,
        "model_version": "CLASSIFIER_v1",
        "model_status": "PRODUCTION",
        "pipeline_type": "CROP_CLASSIFIER_V1",
        "classifier_input": {"width": 224, "height": 224},
        "top1": {"species": "草鱼", "species_key": "grass_carp", "confidence": 0.82},
        "top3": [
            {"species": "草鱼", "species_key": "grass_carp", "confidence": 0.82},
            {"species": "鲤鱼", "species_key": "common_carp", "confidence": 0.31},
            {"species": "鲫鱼", "species_key": "crucian_carp", "confidence": 0.12},
        ],
        "classifier_latency_ms": 30,
        "latency_ms": 54,
        "recognition_log": {
            "stages": [
                {"name": "detector", "status": "SUCCESS", "duration_ms": 20, "debug": "not public"},
                {"name": "crop", "status": "SUCCESS", "duration_ms": 4},
                {"name": "classifier", "status": "SUCCESS", "duration_ms": 30},
            ]
        },
    }


def _upload(name: str, data: bytes, content_type: str) -> UploadFile:
    return UploadFile(filename=name, file=io.BytesIO(data), headers={"content-type": content_type})


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"classification_ran": False, "status_wire": "no_fish"}, "ERROR_NO_FISH"),
        ({"classification_ran": False, "status_wire": "quality_rejected"}, "ERROR_IMAGE_QUALITY"),
        ({"classification_ran": True, "top1": {"confidence": 0.44}}, "RESULT_LOW"),
        (
            {"classification_ran": True, "top1": {"confidence": 0.6}, "top3": [{}, {"confidence": 0.51}]},
            "RESULT_MEDIUM",
        ),
        (
            {"classification_ran": True, "top1": {"confidence": 0.82}, "top3": [{}, {"confidence": 0.31}]},
            "RESULT_HIGH",
        ),
    ],
)
def test_product_result_state_adapts_existing_3plus2_contract(result, expected):
    assert product_api.result_state_for_prediction(result) == expected


def test_product_inference_returns_real_pipeline_data_without_debug_artifacts_or_storage(monkeypatch):
    model = SimpleNamespace(model_version="CLASSIFIER_v1", artifact_uri="gs://models/classifier.pt")
    monkeypatch.setattr(product_api, "_active_classifier", lambda _db: model)
    calls = []

    def predict(db, model_version, data, *, include_intermediates):
        calls.append((db, model_version, data, include_intermediates))
        return _pipeline_result()

    monkeypatch.setattr(product_api, "_predict_bytes", predict)

    response = product_api.infer_catch_photo(_upload("catch.png", _png_bytes(), "image/png"), db=object())

    assert response.status_code == 200
    body = json.loads(response.body)
    assert body["result_state"] == "RESULT_HIGH"
    assert body["detector"]["model_version"] == "DET_FISH_v0.3"
    assert body["classifier"]["top3"][0]["species"] == "草鱼"
    assert body["timing"]["total_ms"] == 54
    assert body["segmentation"]["status"] in {"TRUSTED_BBOX_ONLY", "CANDIDATE_REVIEW_ONLY"}
    assert body["segmentation"]["trust_state"] in {"BBOX_ONLY", "REVIEW_ONLY"}
    assert body["segmentation"]["production_qualified"] is False
    assert body["segmentation"]["source_binding"]["source_sha256"]
    assert body["segmentation"]["source_binding"]["bbox"] == {
        "x1": 0.1,
        "y1": 0.2,
        "x2": 0.8,
        "y2": 0.9,
    }
    assert "image_gcs_uri" not in body
    assert "debug" not in response.body.decode()
    assert "base64" not in response.body.decode()
    assert len(calls) == 1
    assert calls[0][1:] == ("CLASSIFIER_v1", _png_bytes(), False)


def test_product_segmentation_binds_exact_source_and_detector_bbox_without_trusting_sam(monkeypatch):
    data = _png_bytes()
    result = _pipeline_result()
    result["quality_gate"] = {"classifier_allowed": True, "primary_bbox": result["detector"]["detections"][0]["bbox"]}
    observed = {}

    def fake_cutout(image, bbox):
        observed["size"] = image.size
        observed["bbox"] = bbox
        return FishSegmentationResult(
            quality=SegmentationQuality.GOOD,
            reason="mask_passed_basic_checks",
            width=image.width,
            height=image.height,
            mask_area_ratio=0.3,
            edge_ratio=0.1,
            connected_components=1,
            processing_ms=12.0,
            cutout_png=b"private-artifact-not-serialized",
            mask=object(),
        )

    monkeypatch.setattr(product_adapter, "generate_fish_cutout", fake_cutout)
    body = product_adapter.build_product_segmentation(data=data, result=result, session_id="REC_test")

    import hashlib

    assert observed["size"] == (4, 3)
    assert observed["bbox"].x1 == 0.1
    assert body["status"] == "CANDIDATE_REVIEW_ONLY"
    assert body["trust_state"] == "REVIEW_ONLY"
    assert body["production_qualified"] is False
    assert body["mask"] is None
    assert body["source_binding"]["session_id"] == "REC_test"
    assert body["source_binding"]["source_sha256"] == hashlib.sha256(data).hexdigest()
    assert "private-artifact" not in json.dumps(body)


def test_product_inference_rejects_unapproved_image_format(monkeypatch):
    monkeypatch.setattr(product_api, "_active_classifier", lambda _db: SimpleNamespace(model_version="CLASSIFIER_v1"))
    with pytest.raises(HTTPException) as unsupported:
        product_api.infer_catch_photo(
            _upload("catch.gif", b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;", "image/gif"),
            db=object(),
        )
    assert unsupported.value.status_code == 415
    assert unsupported.value.detail["code"] == "IMAGE_FORMAT_UNSUPPORTED"


def test_product_model_selection_requires_exactly_one_production_row():
    class FakeResult:
        def __init__(self, rows):
            self.rows = rows

        def all(self):
            return self.rows

    class FakeDb:
        def __init__(self, rows):
            self.rows = rows

        def scalars(self, _statement):
            return FakeResult(self.rows)

    with pytest.raises(HTTPException) as none:
        product_api._active_classifier(FakeDb([]))
    assert none.value.status_code == 503
    with pytest.raises(HTTPException) as multiple:
        product_api._active_classifier(FakeDb([SimpleNamespace(), SimpleNamespace()]))
    assert multiple.value.status_code == 503
