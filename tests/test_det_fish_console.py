import asyncio
import io
from types import SimpleNamespace

import pytest
import app.det_fish_console as console
from app.det_fish_console import (
    ALLOWED_MODEL_VERSIONS,
    _dataset_gate,
    _execution_status,
    _read_test_image,
    _validate_detector_job,
)
from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image
from starlette.datastructures import Headers
from app.recognition_pipeline import BBox, Detection

def test_dataset_gate_fails_closed_without_frozen_manifest_and_archive():
    result = _dataset_gate(None, None, None)
    assert result["ready"] is False
    assert result["counts_verified"] is False
    assert any("bootstrap_report.json" in item for item in result["blockers"])


def test_execution_status_distinguishes_running_and_failed():
    assert _execution_status({"runningCount": 1}) == "RUNNING"
    assert _execution_status({"completionTime": "2026-10-20T00:00:00Z", "failedCount": 1}) == "FAILED"
    assert _execution_status({"completionTime": "2026-10-20T00:00:00Z", "succeededCount": 1}) == "SUCCEEDED"


def test_only_supported_detector_versions_can_be_selected():
    assert ALLOWED_MODEL_VERSIONS == {"DET_FISH_v0.1", "DET_FISH_v0.2", "DET_FISH_v0.3"}


def test_v03_is_available_when_its_artifacts_load_even_if_v02_is_missing(monkeypatch):
    sha = "a" * 64
    metadata = {
        "model_version": "DET_FISH_v0.3",
        "dataset_version": "DET_DS_v0.3",
        "model_family": "YOLOX_NANO",
        "onnx_sha256": sha,
        "onnx_bytes": 123,
        "training": {"epochs_completed": 30, "ap50": 0.966956, "ap50_95": 0.683351},
    }

    class Blob:
        def __init__(self, name):
            self.name = name

        def exists(self, timeout=10):
            return self.name in {
                "models/DET_FISH_v0.3/detector_metadata.json",
                "models/DET_FISH_v0.3/fish_detector_yolox_nano_v0_1.onnx",
            }

    class Bucket:
        @staticmethod
        def blob(name):
            return Blob(name)

    model = SimpleNamespace(
        onnx_sha256=sha,
        onnx_bytes=123,
        session=SimpleNamespace(get_outputs=lambda: [SimpleNamespace(shape=[1, 10, 6])]),
    )
    monkeypatch.setattr(
        console,
        "_read_json_blob",
        lambda _bucket, name: metadata if name == "models/DET_FISH_v0.3/detector_metadata.json" else None,
    )
    monkeypatch.setattr(console, "load_detector_for_version", lambda version: model)

    v02 = console._model_summary(Bucket(), "DET_FISH_v0.2")
    v03 = console._model_summary(Bucket(), "DET_FISH_v0.3")

    assert v02["available"] is False
    assert v02["reason"] == "缺少 detector_metadata.json"
    assert v03["available"] is True
    assert v03["model_version"] == "DET_FISH_v0.3"
    assert v03["dataset_version"] == "DET_DS_v0.3"
    assert v03["onnx_sha256"] == sha


def test_models_api_lists_v03_available_without_v02_artifacts(monkeypatch):
    sha = "a" * 64
    metadata = {
        "model_version": "DET_FISH_v0.3",
        "dataset_version": "DET_DS_v0.3",
        "model_family": "YOLOX_NANO",
        "onnx_sha256": sha,
        "onnx_bytes": 123,
    }

    class Blob:
        def __init__(self, name):
            self.name = name

        def exists(self, timeout=10):
            return self.name in {
                "models/DET_FISH_v0.3/detector_metadata.json",
                "models/DET_FISH_v0.3/fish_detector_yolox_nano_v0_1.onnx",
            }

    class Bucket:
        @staticmethod
        def blob(name):
            return Blob(name)

    model = SimpleNamespace(
        onnx_sha256=sha,
        onnx_bytes=123,
        session=SimpleNamespace(get_outputs=lambda: [SimpleNamespace(shape=[1, 10, 6])]),
    )
    monkeypatch.setattr(console, "_bucket", Bucket)
    monkeypatch.setattr(
        console,
        "_read_json_blob",
        lambda _bucket, name: metadata if name == "models/DET_FISH_v0.3/detector_metadata.json" else None,
    )
    monkeypatch.setattr(console, "load_detector_for_version", lambda _version: model)

    app = FastAPI()
    app.include_router(console.router)
    status_code, payload = _asgi_request(app, "GET", "/api/det-fish/models")

    assert status_code == 200
    models = {row["model_version"]: row for row in payload["models"]}
    assert models["DET_FISH_v0.2"]["available"] is False
    assert models["DET_FISH_v0.2"]["reason"] == "缺少 detector_metadata.json"
    assert models["DET_FISH_v0.3"]["available"] is True
    assert models["DET_FISH_v0.3"]["dataset_version"] == "DET_DS_v0.3"
    assert models["DET_FISH_v0.3"]["onnx_sha256"] == sha


@pytest.mark.parametrize(
    "metadata,loader,expected_available",
    [
        (
            {"model_version": "DET_FISH_v0.2", "dataset_version": "DET_DS_v0.3", "model_family": "YOLOX_NANO", "onnx_sha256": "a" * 64, "onnx_bytes": 123},
            lambda: SimpleNamespace(onnx_sha256="a" * 64, onnx_bytes=123, session=SimpleNamespace(get_outputs=lambda: [SimpleNamespace(shape=[1, 10, 6])])),
            False,
        ),
        (
            {"model_version": "DET_FISH_v0.3", "dataset_version": "DET_DS_v0.3", "model_family": "YOLOX_NANO", "onnx_sha256": "a" * 64, "onnx_bytes": 123},
            lambda: SimpleNamespace(onnx_sha256="b" * 64, onnx_bytes=123, session=SimpleNamespace(get_outputs=lambda: [SimpleNamespace(shape=[1, 10, 6])])),
            False,
        ),
        (
            {"model_version": "DET_FISH_v0.3", "dataset_version": "DET_DS_v0.3", "model_family": "YOLOX_NANO", "onnx_sha256": "a" * 64, "onnx_bytes": 123},
            lambda: (_ for _ in ()).throw(RuntimeError("ONNX Runtime load failed")),
            False,
        ),
    ],
)
def test_v03_model_summary_fails_closed_on_metadata_sha_or_runtime_error(monkeypatch, metadata, loader, expected_available):
    class Blob:
        def __init__(self, name):
            self.name = name

        def exists(self, timeout=10):
            return self.name.endswith(("detector_metadata.json", "fish_detector_yolox_nano_v0_1.onnx"))

    class Bucket:
        @staticmethod
        def blob(name):
            return Blob(name)

    monkeypatch.setattr(console, "_read_json_blob", lambda _bucket, _name: metadata)
    monkeypatch.setattr(console, "load_detector_for_version", lambda _version: loader())
    result = console._model_summary(Bucket(), "DET_FISH_v0.3")
    assert result["available"] is expected_available


def test_det_fish_test_endpoint_passes_selected_v03_and_returns_version_and_sha(monkeypatch):
    requested_versions = []
    expected_sha = "c" * 64
    detection = Detection(confidence=0.91, box=BBox(0.1, 0.15, 0.8, 0.85))
    run = SimpleNamespace(
        model_version="DET_FISH_v0.3",
        onnx_sha256=expected_sha,
        input_size=416,
        input_draw_width=333,
        input_draw_height=416,
        latency_ms=12.5,
        detections=(detection,),
        selected_attempt="ORIGINAL",
        selection_reason="ORIGINAL_ASSESSMENT_PRESERVED",
        attempt_trace=(),
    )

    def fake_detect(_image, model_version=None):
        requested_versions.append(model_version)
        return run

    monkeypatch.setattr(console, "detect", fake_detect)
    app = FastAPI()
    app.include_router(console.router)
    boundary = "det-fish-test-boundary"
    png = _png_bytes()
    body = b"".join(
        [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"model_version\"\r\n\r\nDET_FISH_v0.3\r\n".encode(),
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"test.png\"\r\nContent-Type: image/png\r\n\r\n".encode(),
            png,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    status_code, payload = _asgi_request(
        app,
        "POST",
        "/api/det-fish/test",
        body=body,
        headers=[
            (b"content-type", f"multipart/form-data; boundary={boundary}".encode()),
            (b"content-length", str(len(body)).encode()),
        ],
    )

    assert requested_versions == ["DET_FISH_v0.3"]
    assert status_code == 200
    assert payload["model_version"] == "DET_FISH_v0.3"
    assert payload["onnx_sha256"] == expected_sha
    assert payload["model_input"]["shape"] == [1, 3, 416, 416]
    assert payload["detector"]["detections"][0]["confidence"] == pytest.approx(0.91)
    assert payload["overlay_data_url"].startswith("data:image/png;base64,")


def test_cloud_run_detector_job_must_match_l4_worker_contract():
    job = {
        "template": {
            "template": {
                "containers": [{
                    "image": "asia-southeast1-docker.pkg.dev/project/repo/detector-trainer@sha256:" + "a" * 64,
                    "env": [
                        {"name": "DETECTOR_DATASET_VERSION", "value": "DET_DS_v0.2"},
                        {"name": "DETECTOR_MODEL_VERSION", "value": "DET_FISH_v0.2"},
                        {"name": "APP_GIT_COMMIT", "value": "b" * 40},
                    ],
                    "resources": {"limits": {"cpu": "4", "memory": "16Gi", "nvidia.com/gpu": "1"}},
                }],
                "serviceAccount": "detector@project.iam.gserviceaccount.com",
                "nodeSelector": {"run.googleapis.com/accelerator": "nvidia-l4"},
                "maxRetries": 0,
                "gpuZonalRedundancyDisabled": True,
            }
        }
    }
    assert _validate_detector_job(job) == []
    job["template"]["template"]["containers"][0]["resources"] = {"limits": {"cpu": "2", "memory": "4Gi"}}
    assert any("1 张 NVIDIA GPU" in item for item in _validate_detector_job(job))


def test_cloud_run_detector_job_rejects_v01_dataset_or_mutable_image_tag():
    job = {
        "template": {
            "template": {
                "containers": [{
                    "image": "asia-southeast1-docker.pkg.dev/project/repo/detector-trainer:deadbeef",
                    "env": [
                        {"name": "DETECTOR_DATASET_VERSION", "value": "DET_DS_v0.1"},
                        {"name": "DETECTOR_MODEL_VERSION", "value": "DET_FISH_v0.1"},
                        {"name": "APP_GIT_COMMIT", "value": "b" * 40},
                    ],
                    "resources": {"limits": {"cpu": "4", "memory": "16Gi", "nvidia.com/gpu": "1"}},
                }],
                "serviceAccount": "detector@project.iam.gserviceaccount.com",
                "nodeSelector": {"run.googleapis.com/accelerator": "nvidia-l4"},
                "maxRetries": 0,
                "gpuZonalRedundancyDisabled": True,
            }
        }
    }
    blockers = _validate_detector_job(job)
    assert any("DET_DS_v0.2" in item for item in blockers)
    assert any("DET_FISH_v0.2" in item for item in blockers)
    assert any("digest" in item for item in blockers)


def _upload(data: bytes, content_type: str | None):
    headers = Headers({"content-type": content_type}) if content_type else Headers()
    return UploadFile(filename="upload", file=io.BytesIO(data), headers=headers)


def _asgi_request(app, method, path, *, body=b"", headers=()):
    messages = []
    received = False

    async def receive():
        nonlocal received
        if received:
            return {"type": "http.disconnect"}
        received = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        messages.append(message)

    encoded_path = path.encode("ascii")
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": encoded_path,
        "root_path": "",
        "query_string": b"",
        "headers": list(headers),
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    asyncio.run(app(scope, receive, send))
    start = next(item for item in messages if item["type"] == "http.response.start")
    response_body = b"".join(item.get("body", b"") for item in messages if item["type"] == "http.response.body")
    import json

    return start["status"], json.loads(response_body or b"{}")


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (2, 3), (10, 20, 30)).save(output, format="PNG")
    return output.getvalue()


def test_upload_accepts_verified_png_content():
    data = _png_bytes()
    assert asyncio.run(_read_test_image(_upload(data, "image/png"))) == data


def test_upload_rejects_nonimage_bytes_and_mime_mismatch():
    with pytest.raises(HTTPException) as invalid:
        asyncio.run(_read_test_image(_upload(b"not an image", "image/png")))
    assert invalid.value.status_code == 400

    with pytest.raises(HTTPException) as mismatch:
        asyncio.run(_read_test_image(_upload(_png_bytes(), "image/jpeg")))
    assert "MIME" in mismatch.value.detail
