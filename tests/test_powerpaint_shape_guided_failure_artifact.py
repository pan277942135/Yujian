import asyncio
import io
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

import app.powerpaint_shape_guided_lab as lab


def _image_bytes():
    out = io.BytesIO()
    Image.new("RGB", (32, 24), (20, 80, 120)).save(out, format="PNG")
    return out.getvalue()


class _Request:
    headers = {"content-type": "application/x-www-form-urlencoded"}

    async def form(self):
        return {"dataset_version": "DS_TEST", "dataset_item_id": "1", "fitting_degrees": "0.95"}


class _DB:
    def __init__(self, item, dataset):
        self.item = item
        self.dataset = dataset

    def scalar(self, _statement):
        return self.item

    def get(self, _model, _key):
        return self.dataset


def _patch_runtime(monkeypatch, mask, worker_health=None, worker_result=None):
    item = SimpleNamespace(id=1, dataset_version="DS_TEST", image_id="IMG_TEST", gcs_uri="gs://input/test.png")
    dataset = SimpleNamespace(status="FROZEN")
    source = Image.new("RGB", (32, 24), (20, 80, 120))
    box = SimpleNamespace(x1=0.1, y1=0.1, x2=0.9, y2=0.9, normalized=lambda: box)
    detection = SimpleNamespace(box=box, confidence=0.95)
    assessment = SimpleNamespace(primary=detection, status=SimpleNamespace(value="GOOD"))
    detector_run = SimpleNamespace(model_version="DET_TEST", detections=[detection])
    segmentation = SimpleNamespace(mask=mask, quality=SimpleNamespace(value="GOOD"))

    monkeypatch.setattr(lab, "_read_dataset_image", lambda _item: _image_bytes())
    monkeypatch.setattr(lab, "normalize_android_source", lambda _uploaded: source.copy())
    monkeypatch.setattr(lab, "detect", lambda _source: detector_run)
    monkeypatch.setattr(lab, "assess_detections", lambda _detections: assessment)
    monkeypatch.setattr(lab, "generate_fish_cutout", lambda _source, _box: segmentation)
    monkeypatch.setattr(lab, "_persist", lambda test_id, name, data, content_type: f"gs://test/{test_id}/{name}")
    if worker_health is not None:
        monkeypatch.setattr(lab, "_check_worker_health", worker_health)
    if worker_result is not None:
        monkeypatch.setattr(lab, "_invoke_shape_guided", worker_result)
    return _Request(), _DB(item, dataset)


def _response_status_and_body(response):
    if isinstance(response, dict):
        return 200, json.dumps(response)
    return response.status_code, response.body.decode()


def test_worker_timeout_persists_all_failure_artifacts(monkeypatch):
    visible = np.ones((24, 32), dtype=bool)
    visible[10:14, 14:18] = False
    request, db = _patch_runtime(monkeypatch, visible, worker_health=lambda: (_ for _ in ()).throw(TimeoutError("timeout")))
    response = asyncio.run(lab.run(request, db))
    status, body = _response_status_and_body(response)
    assert status == 503
    for name in ("original", "detector_crop", "sam_visible", "sam_mask", "completion_mask", "detector_report", "sam_report", "completion_report", "shape_guided_request", "shape_guided_response", "report", "error"):
        assert name in body
    assert "FAILED_WORKER" in body
    assert "experiment_stage" in body


def test_complete_fish_is_not_required_and_does_not_call_worker(monkeypatch):
    visible = np.ones((24, 32), dtype=bool)
    called = {"count": 0}

    def fail_worker():
        called["count"] += 1
        raise AssertionError("complete fish must not call Worker")

    request, db = _patch_runtime(monkeypatch, visible, worker_health=fail_worker)
    response = asyncio.run(lab.run(request, db))
    status, body = _response_status_and_body(response)
    assert status == 200
    assert "SUCCESS_NOT_REQUIRED" in body
    assert "COMPLETE_FISH" in body
    assert called["count"] == 0


def test_powerpaint_exception_is_classified_and_report_is_persisted(monkeypatch):
    visible = np.ones((24, 32), dtype=bool)
    visible[10:14, 14:18] = False
    request, db = _patch_runtime(
        monkeypatch,
        visible,
        worker_health=lambda: {"status_code": 200, "status": "ok"},
        worker_result=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("inference failed")),
    )
    response = asyncio.run(lab.run(request, db))
    status, body = _response_status_and_body(response)
    assert status == 502
    assert "FAILED_POWERPAINT" in body
    assert "shape_guided_response" in body
    assert "report" in body


def test_successful_generation_persists_report_and_final(monkeypatch):
    visible = np.ones((24, 32), dtype=bool)
    visible[10:14, 14:18] = False
    result_bytes = _image_bytes()
    result_uri = "data:image/png;base64," + __import__("base64").b64encode(result_bytes).decode()
    request, db = _patch_runtime(
        monkeypatch,
        visible,
        worker_health=lambda: {"status_code": 200, "status": "ok"},
        worker_result=lambda **_kwargs: {"http_status": 200, "result_uri": result_uri, "inference_time_ms": 12},
    )
    response = asyncio.run(lab.run(request, db))
    status, body = _response_status_and_body(response)
    assert status == 200
    assert "SUCCESS_COMPLETED" in body
    assert "shape_guided_report" in body
    assert "final_result" in body
