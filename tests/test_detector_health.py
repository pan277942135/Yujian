from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app import entry


def test_detector_health_reports_verified_runtime_metadata(monkeypatch):
    monkeypatch.setattr(
        entry,
        "load_detector",
        lambda: SimpleNamespace(
            model_version="DET_FISH_v0.1",
            onnx_sha256="a" * 64,
            onnx_bytes=123,
            input_size=416,
        ),
    )

    assert entry.detector_health() == {
        "status": "ok",
        "model_version": "DET_FISH_v0.1",
        "onnx_sha256": "a" * 64,
        "onnx_bytes": 123,
        "input_size": 416,
    }


def test_detector_health_is_unavailable_when_artifact_cannot_load(monkeypatch):
    def fail():
        raise RuntimeError("GCS artifact unavailable")

    monkeypatch.setattr(entry, "load_detector", fail)
    with pytest.raises(HTTPException) as raised:
        entry.detector_health()
    assert raised.value.status_code == 503
