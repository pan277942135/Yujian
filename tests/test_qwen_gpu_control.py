import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.entry import app
from app.platform.models import PipelineRun
from app.platform.routes import qwen_gpu as gpu
from app.portrait_worker_client import PortraitWorkerError


def _session(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'qwen_gpu.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)()


def _env(monkeypatch):
    monkeypatch.setenv("QWEN_GPU_PROJECT_ID", "gen-lang-client-0224022273")
    monkeypatch.setenv("QWEN_GPU_ZONE", "us-central1-a")
    monkeypatch.setenv("QWEN_GPU_INSTANCE", "fish-completion-worker")
    monkeypatch.setenv("QWEN_WORKER_BASE_URL", "http://34.69.75.199:8002")


def test_qwen_gpu_routes_are_exposed():
    paths = app.openapi()["paths"]
    assert "/api/qwen-lab/gpu/status" in paths
    assert "/api/qwen-lab/gpu/start" in paths
    assert "/api/qwen-lab/gpu/stop" in paths


def test_status_maps_terminated_to_stopped(tmp_path, monkeypatch):
    _env(monkeypatch)
    db = _session(tmp_path)
    try:
        monkeypatch.setattr(gpu, "_get_instance", lambda config: {"status": "TERMINATED"})
        payload = gpu.qwen_gpu_status(db)
        assert payload["vm_status"] == "TERMINATED"
        assert payload["display_status"] == "STOPPED"
        assert payload["model_loaded"] is False
    finally:
        db.close()


def test_status_maps_running_and_ready_worker_to_ready(tmp_path, monkeypatch):
    _env(monkeypatch)
    db = _session(tmp_path)
    try:
        monkeypatch.setattr(
            gpu,
            "_get_instance",
            lambda config: {
                "status": "RUNNING",
                "lastStartTimestamp": "2026-09-21T06:00:00.000-07:00",
            },
        )
        monkeypatch.setattr(
            gpu,
            "check_qwen_refine_worker",
            lambda: {"health": {"status": "ready", "model_loaded": True}},
        )
        payload = gpu.qwen_gpu_status(db)
        assert payload["vm_status"] == "RUNNING"
        assert payload["worker_status"] == "ready"
        assert payload["model_loaded"] is True
        assert payload["display_status"] == "READY"
        assert payload["run_seconds"] is not None
    finally:
        db.close()


def test_running_vm_without_ready_health_maps_to_loading(tmp_path, monkeypatch):
    _env(monkeypatch)
    db = _session(tmp_path)
    try:
        monkeypatch.setattr(gpu, "_get_instance", lambda config: {"status": "RUNNING"})
        monkeypatch.setattr(
            gpu,
            "check_qwen_refine_worker",
            lambda: (_ for _ in ()).throw(
                PortraitWorkerError("QWEN_WORKER_HEALTH_UNREACHABLE", "connection refused")
            ),
        )
        payload = gpu.qwen_gpu_status(db)
        assert payload["display_status"] == "LOADING"
        assert payload["model_loaded"] is False
        assert payload["worker_status"] is None
    finally:
        db.close()


def test_start_only_calls_compute_start_for_stopped_vm(tmp_path, monkeypatch):
    _env(monkeypatch)
    db = _session(tmp_path)
    calls = []
    try:
        monkeypatch.setattr(gpu, "_get_instance", lambda config: {"status": "TERMINATED"})
        monkeypatch.setattr(
            gpu,
            "_start_instance",
            lambda config: calls.append(config.instance) or {"name": "operation-1"},
        )
        payload = gpu.qwen_gpu_start(db)
        assert payload == {
            "accepted": True,
            "display_status": "STARTING",
            "vm_status": "TERMINATED",
            "operation": "operation-1",
        }
        assert calls == ["fish-completion-worker"]
    finally:
        db.close()


def test_stop_rejects_active_qwen_run(tmp_path, monkeypatch):
    _env(monkeypatch)
    db = _session(tmp_path)
    try:
        db.add(
            PipelineRun(
                run_id="QWEN_BUSY_1",
                pipeline_type="QWEN_IMAGE_EDIT_LAB",
                status="RUNNING",
                stage_json=json.dumps({"type": "QWEN_IMAGE_EDIT_LAB"}),
            )
        )
        db.commit()
        monkeypatch.setattr(gpu, "_get_instance", lambda config: {"status": "RUNNING"})
        monkeypatch.setattr(
            gpu,
            "check_qwen_refine_worker",
            lambda: {"health": {"status": "ready", "model_loaded": True}},
        )
        with pytest.raises(HTTPException) as error:
            gpu.qwen_gpu_stop(db)
        assert error.value.status_code == 409
        assert error.value.detail["error_code"] == "QWEN_GPU_BUSY"
    finally:
        db.close()


def test_stop_calls_compute_stop_only_when_ready_and_idle(tmp_path, monkeypatch):
    _env(monkeypatch)
    db = _session(tmp_path)
    calls = []
    try:
        monkeypatch.setattr(gpu, "_get_instance", lambda config: {"status": "RUNNING"})
        monkeypatch.setattr(
            gpu,
            "check_qwen_refine_worker",
            lambda: {"health": {"status": "ready", "model_loaded": True}},
        )
        monkeypatch.setattr(
            gpu,
            "_stop_instance",
            lambda config: calls.append(config.instance) or {"name": "operation-2"},
        )
        payload = gpu.qwen_gpu_stop(db)
        assert payload["accepted"] is True
        assert payload["display_status"] == "STOPPING"
        assert payload["operation"] == "operation-2"
        assert calls == ["fish-completion-worker"]
    finally:
        db.close()


def test_status_exposes_permission_error_without_hiding_it(tmp_path, monkeypatch):
    _env(monkeypatch)
    db = _session(tmp_path)
    try:
        monkeypatch.setattr(
            gpu,
            "_get_instance",
            lambda config: (_ for _ in ()).throw(
                gpu.QwenGpuControlError(
                    "QWEN_GPU_PERMISSION_DENIED",
                    "missing compute.instances.get",
                    permission="compute.instances.get",
                    service_account="runtime@example.iam.gserviceaccount.com",
                )
            ),
        )
        payload = gpu.qwen_gpu_status(db)
        assert payload["display_status"] == "ERROR"
        assert payload["error"]["permission"] == "compute.instances.get"
        assert payload["error"]["service_account"] == "runtime@example.iam.gserviceaccount.com"
    finally:
        db.close()


def test_qwen_gpu_template_contains_manual_control_and_polling():
    template = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "templates"
        / "platform"
        / "lab"
        / "qwen_image_edit.html"
    ).read_text(encoding="utf-8")
    assert 'id="qwenGpuStatusTag"' in template
    assert 'id="qwenGpuStartButton"' in template
    assert 'id="qwenGpuStopButton"' in template
    assert "STOPPED: '已关闭'" in template
    assert "STARTING: 'GPU 启动中'" in template
    assert "LOADING: 'Qwen 模型加载中'" in template
    assert "READY: 'Ready'" in template
    assert "BUSY: '正在生成'" in template
    assert "STOPPING: 'GPU 关闭中'" in template
    assert "ERROR: '状态异常'" in template
    assert "fast ? 5000 : 15000" in template
    assert "gpuDisplayStatus !== 'READY'" in template
    assert "确认关闭" in template
