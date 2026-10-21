import asyncio
import io

import pytest
import app.det_fish_console as console
from app.det_fish_console import (
    ALLOWED_MODEL_VERSIONS,
    _dataset_gate,
    _execution_status,
    _read_test_image,
    _validate_detector_job,
)
from fastapi import HTTPException, UploadFile
from PIL import Image
from starlette.datastructures import Headers

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


def test_cloud_run_detector_job_accepts_exact_v03_pair(monkeypatch):
    monkeypatch.setattr(console, "DATASET_VERSION", "DET_DS_v0.3")
    monkeypatch.setattr(console, "MODEL_VERSION", "DET_FISH_v0.3")
    job = {
        "template": {
            "template": {
                "containers": [{
                    "image": "asia-southeast1-docker.pkg.dev/project/repo/detector-trainer@sha256:" + "c" * 64,
                    "env": [
                        {"name": "DETECTOR_DATASET_VERSION", "value": "DET_DS_v0.3"},
                        {"name": "DETECTOR_MODEL_VERSION", "value": "DET_FISH_v0.3"},
                        {"name": "APP_GIT_COMMIT", "value": "d" * 40},
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
    assert console._validate_detector_job(job) == []
