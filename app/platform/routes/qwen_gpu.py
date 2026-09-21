"""Manual GPU VM control for the independent Qwen Image Edit Lab.

This router only controls the existing Compute Engine VM. Starting the VM is
deliberately non-blocking: the VM boot process and the existing systemd unit
start fish-qwen-refine-worker and load Qwen. The Lab polls this router until
the worker health response reports ready and model_loaded=true.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.platform.models import PipelineRun
from app.qwen_refine_worker_client import check_qwen_refine_worker
from app.portrait_worker_client import PortraitWorkerError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/qwen-lab/gpu", tags=["qwen-lab-gpu"])

GPU_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
DEFAULT_GPU_NAME = "NVIDIA L4"
DEFAULT_INSTANCE = "fish-completion-worker"
DEFAULT_ZONE = "us-central1-a"
DEFAULT_WORKER = "fish-qwen-refine-worker"
DEFAULT_MODEL = "Qwen Image Edit 2511"
PIPELINE_TYPE = "QWEN_IMAGE_EDIT_LAB"
TRANSITIONAL_VM_STATES = {"STAGING", "PROVISIONING"}
STOPPING_VM_STATES = {"STOPPING", "SUSPENDING"}
STOPPED_VM_STATES = {"TERMINATED", "SUSPENDED"}


class QwenGpuControlError(RuntimeError):
    """Safe, structured error returned by the GPU control API."""

    def __init__(
        self,
        error_code: str,
        message: str,
        *,
        status_code: int = 503,
        permission: str | None = None,
        service_account: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        self.permission = permission
        self.service_account = service_account


@dataclass(frozen=True)
class QwenGpuConfig:
    project_id: str
    zone: str
    instance: str
    worker_base_url: str
    gpu: str
    worker: str
    model: str


def _env_required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise QwenGpuControlError(
            "QWEN_GPU_CONFIG_MISSING",
            f"Backend 环境变量 {name} 未配置",
            status_code=503,
        )
    return value


def _config() -> QwenGpuConfig:
    worker_base_url = (
        os.getenv("QWEN_WORKER_BASE_URL", "").strip().rstrip("/")
        or os.getenv("FISH_QWEN_REFINE_WORKER_URL", "").strip().rstrip("/")
    )
    return QwenGpuConfig(
        project_id=_env_required("QWEN_GPU_PROJECT_ID"),
        zone=os.getenv("QWEN_GPU_ZONE", DEFAULT_ZONE).strip() or DEFAULT_ZONE,
        instance=os.getenv("QWEN_GPU_INSTANCE", DEFAULT_INSTANCE).strip() or DEFAULT_INSTANCE,
        worker_base_url=worker_base_url,
        gpu=os.getenv("QWEN_GPU_NAME", DEFAULT_GPU_NAME).strip() or DEFAULT_GPU_NAME,
        worker=os.getenv("QWEN_GPU_WORKER", DEFAULT_WORKER).strip() or DEFAULT_WORKER,
        model=os.getenv("QWEN_GPU_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
    )


def _identity_config() -> dict[str, str]:
    return {
        "gpu": os.getenv("QWEN_GPU_NAME", DEFAULT_GPU_NAME).strip() or DEFAULT_GPU_NAME,
        "instance": os.getenv("QWEN_GPU_INSTANCE", DEFAULT_INSTANCE).strip() or DEFAULT_INSTANCE,
        "zone": os.getenv("QWEN_GPU_ZONE", DEFAULT_ZONE).strip() or DEFAULT_ZONE,
        "worker": os.getenv("QWEN_GPU_WORKER", DEFAULT_WORKER).strip() or DEFAULT_WORKER,
        "model": os.getenv("QWEN_GPU_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
    }


def _authorized_session() -> tuple[Any, str]:
    try:
        import google.auth
        from google.auth.transport.requests import AuthorizedSession

        credentials, _ = google.auth.default(scopes=[GPU_SCOPE])
        return AuthorizedSession(credentials), str(
            getattr(credentials, "service_account_email", "") or "unknown"
        )
    except Exception as exc:
        raise QwenGpuControlError(
            "QWEN_GPU_AUTH_FAILED",
            f"无法使用 Cloud Run Runtime SA 访问 Compute Engine: {exc}",
            status_code=503,
        ) from exc


def _instance_url(config: QwenGpuConfig) -> str:
    project = quote(config.project_id, safe="")
    zone = quote(config.zone, safe="")
    instance = quote(config.instance, safe="")
    return (
        "https://compute.googleapis.com/compute/v1/"
        f"projects/{project}/zones/{zone}/instances/{instance}"
    )


def _request_json(
    method: str,
    url: str,
    *,
    permission: str,
    session: Any | None = None,
    service_account: str | None = None,
) -> dict[str, Any]:
    if session is None or service_account is None:
        session, service_account = _authorized_session()
    try:
        response = session.request(method, url, timeout=20)
    except Exception as exc:
        raise QwenGpuControlError(
            "QWEN_GPU_API_UNREACHABLE",
            f"Compute Engine API 请求失败: {exc}",
            status_code=503,
            permission=permission,
            service_account=service_account,
        ) from exc

    if response.status_code in {401, 403}:
        detail = str(getattr(response, "text", "") or "")[:1000]
        raise QwenGpuControlError(
            "QWEN_GPU_PERMISSION_DENIED",
            (
                f"Runtime SA {service_account} 缺少或未生效的权限 {permission}"
                + (f": {detail}" if detail else "")
            ),
            status_code=503,
            permission=permission,
            service_account=service_account,
        )
    if response.status_code < 200 or response.status_code >= 300:
        detail = str(getattr(response, "text", "") or "")[:1000]
        raise QwenGpuControlError(
            "QWEN_GPU_API_ERROR",
            f"Compute Engine API 返回 HTTP {response.status_code}"
            + (f": {detail}" if detail else ""),
            status_code=503,
            permission=permission,
            service_account=service_account,
        )
    try:
        payload = response.json()
    except Exception:
        payload = {}
    return payload if isinstance(payload, dict) else {}


def _get_instance(config: QwenGpuConfig) -> dict[str, Any]:
    session, service_account = _authorized_session()
    return _request_json(
        "GET",
        _instance_url(config),
        permission="compute.instances.get",
        session=session,
        service_account=service_account,
    )


def _start_instance(config: QwenGpuConfig) -> dict[str, Any]:
    session, service_account = _authorized_session()
    return _request_json(
        "POST",
        _instance_url(config) + "/start",
        permission="compute.instances.start",
        session=session,
        service_account=service_account,
    )


def _stop_instance(config: QwenGpuConfig) -> dict[str, Any]:
    session, service_account = _authorized_session()
    return _request_json(
        "POST",
        _instance_url(config) + "/stop",
        permission="compute.instances.stop",
        session=session,
        service_account=service_account,
    )


def _active_jobs(db: Session) -> int:
    value = db.scalar(
        select(func.count())
        .select_from(PipelineRun)
        .where(
            PipelineRun.pipeline_type == PIPELINE_TYPE,
            PipelineRun.status.in_(["QUEUED", "RUNNING"]),
        )
    )
    return int(value or 0)


def _worker_snapshot() -> dict[str, Any]:
    try:
        result = check_qwen_refine_worker()
    except PortraitWorkerError as exc:
        return {
            "worker_status": None,
            "model_loaded": False,
            "ready": False,
            "error": {"error_code": exc.error_code, "message": str(exc)[:1000]},
        }
    except Exception as exc:
        logger.exception("qwen_gpu_worker_health_failed")
        return {
            "worker_status": None,
            "model_loaded": False,
            "ready": False,
            "error": {"error_code": "QWEN_WORKER_HEALTH_FAILED", "message": str(exc)[:1000]},
        }

    health = result.get("health") if isinstance(result.get("health"), dict) else {}
    worker_status = str(health.get("status") or "").strip().lower() or None
    model_loaded = health.get("model_loaded") is True
    ready = worker_status == "ready" and model_loaded
    explicit_error = worker_status in {"error", "failed"}
    error = None
    if explicit_error:
        error = {
            "error_code": str(
                health.get("error_code")
                or health.get("code")
                or "QWEN_WORKER_ERROR"
            ),
            "message": str(
                health.get("error")
                or health.get("message")
                or "Qwen Worker 报告了明确错误"
            )[:1000],
        }
    elif not ready:
        error = {
            "error_code": "QWEN_WORKER_NOT_READY",
            "message": "Qwen Worker 尚未报告 status=ready 且 model_loaded=true",
        }
    return {
        "worker_status": worker_status,
        "model_loaded": model_loaded,
        "ready": ready,
        "explicit_error": explicit_error,
        "error": error,
        "health": health,
    }


def _run_time_fields(instance: dict[str, Any], vm_status: str) -> dict[str, Any]:
    started_at = instance.get("lastStartTimestamp")
    run_seconds = None
    if vm_status == "RUNNING" and started_at:
        try:
            value = str(started_at).replace("Z", "+00:00")
            started = datetime.fromisoformat(value)
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            run_seconds = max(0, int((datetime.now(timezone.utc) - started).total_seconds()))
        except (TypeError, ValueError, OverflowError):
            run_seconds = None
    return {"started_at": started_at, "run_seconds": run_seconds}


def _base_payload(config: QwenGpuConfig | None = None) -> dict[str, Any]:
    identity = _identity_config() if config is None else {
        "gpu": config.gpu,
        "instance": config.instance,
        "zone": config.zone,
        "worker": config.worker,
        "model": config.model,
    }
    return {
        **identity,
        "service": identity["worker"],
        "vm_status": None,
        "worker_status": None,
        "model_loaded": False,
        "active_jobs": 0,
        "display_status": "ERROR",
        "started_at": None,
        "run_seconds": None,
    }


def _status_from_instance(
    config: QwenGpuConfig,
    instance: dict[str, Any],
    db: Session,
) -> dict[str, Any]:
    vm_status = str(instance.get("status") or "UNKNOWN").upper()
    payload = _base_payload(config)
    payload["vm_status"] = vm_status
    payload.update(_run_time_fields(instance, vm_status))

    if vm_status in STOPPED_VM_STATES:
        payload["display_status"] = "STOPPED"
        return payload
    if vm_status in TRANSITIONAL_VM_STATES:
        payload["display_status"] = "STARTING"
        return payload
    if vm_status in STOPPING_VM_STATES:
        payload["display_status"] = "STOPPING"
        return payload
    if vm_status != "RUNNING":
        payload["display_status"] = "ERROR"
        payload["error"] = {
            "error_code": "QWEN_GPU_UNKNOWN_VM_STATUS",
            "message": f"不支持的 VM 状态: {vm_status}",
        }
        return payload

    worker = _worker_snapshot()
    active_jobs = _active_jobs(db)
    payload["worker_status"] = worker.get("worker_status")
    payload["model_loaded"] = bool(worker.get("model_loaded"))
    payload["active_jobs"] = active_jobs
    if worker.get("explicit_error"):
        payload["display_status"] = "ERROR"
        payload["error"] = worker["error"]
        payload["worker_error"] = worker["error"]
        payload["error_code"] = worker["error"]["error_code"]
        payload["message"] = worker["error"]["message"]
    elif active_jobs > 0:
        payload["display_status"] = "BUSY"
    elif worker.get("ready"):
        payload["display_status"] = "READY"
    else:
        # A just-started VM can expose RUNNING before the Worker HTTP
        # listener exists. Keep this transient transport gap in LOADING.
        payload["display_status"] = "LOADING"
    if worker.get("error") and not worker.get("explicit_error"):
        payload["worker_error"] = worker["error"]
    return payload


def _error_payload(exc: QwenGpuControlError) -> dict[str, Any]:
    payload = _base_payload()
    payload["error"] = {
        "error_code": exc.error_code,
        "message": exc.message,
        **({"permission": exc.permission} if exc.permission else {}),
        **({"service_account": exc.service_account} if exc.service_account else {}),
    }
    return payload


def _http_error(exc: QwenGpuControlError) -> HTTPException:
    detail = {
        "error_code": exc.error_code,
        "message": exc.message,
    }
    if exc.permission:
        detail["permission"] = exc.permission
    if exc.service_account:
        detail["service_account"] = exc.service_account
    return HTTPException(status_code=exc.status_code, detail=detail)


@router.get("/status")
def qwen_gpu_status(db: Session = Depends(get_db)) -> dict[str, Any]:
    try:
        config = _config()
        instance = _get_instance(config)
        return _status_from_instance(config, instance, db)
    except QwenGpuControlError as exc:
        raise _http_error(exc) from exc
    except Exception as exc:
        logger.exception("qwen_gpu_status_failed")
        raise _http_error(
            QwenGpuControlError(
                "QWEN_GPU_STATUS_FAILED",
                str(exc)[:1000],
                status_code=503,
            )
        ) from exc


@router.post("/start")
def qwen_gpu_start(db: Session = Depends(get_db)) -> dict[str, Any]:
    try:
        config = _config()
        instance = _get_instance(config)
        vm_status = str(instance.get("status") or "UNKNOWN").upper()
        if vm_status in STOPPED_VM_STATES:
            operation = _start_instance(config)
            return {
                "accepted": True,
                "display_status": "STARTING",
                "vm_status": vm_status,
                "operation": operation.get("name"),
            }
        if vm_status in TRANSITIONAL_VM_STATES:
            return {"accepted": True, "display_status": "STARTING", "vm_status": vm_status}
        if vm_status == "RUNNING":
            state = _status_from_instance(config, instance, db)
            return {
                "accepted": False,
                "display_status": state["display_status"],
                "vm_status": vm_status,
            }
        if vm_status in STOPPING_VM_STATES:
            return {"accepted": False, "display_status": "STOPPING", "vm_status": vm_status}
        raise QwenGpuControlError(
            "QWEN_GPU_START_INVALID_STATE",
            f"VM 当前状态 {vm_status} 不允许启动",
            status_code=409,
        )
    except QwenGpuControlError as exc:
        raise _http_error(exc) from exc
    except Exception as exc:
        logger.exception("qwen_gpu_start_failed")
        raise _http_error(
            QwenGpuControlError("QWEN_GPU_START_FAILED", str(exc)[:1000])
        ) from exc


@router.post("/stop")
def qwen_gpu_stop(db: Session = Depends(get_db)) -> dict[str, Any]:
    try:
        config = _config()
        instance = _get_instance(config)
        vm_status = str(instance.get("status") or "UNKNOWN").upper()
        if vm_status in STOPPED_VM_STATES:
            return {"accepted": False, "display_status": "STOPPED", "vm_status": vm_status}
        if vm_status in STOPPING_VM_STATES:
            return {"accepted": True, "display_status": "STOPPING", "vm_status": vm_status}
        if vm_status != "RUNNING":
            raise QwenGpuControlError(
                "QWEN_GPU_STOP_INVALID_STATE",
                f"VM 当前状态 {vm_status} 不允许关闭",
                status_code=409,
            )

        active_jobs = _active_jobs(db)
        if active_jobs > 0:
            raise QwenGpuControlError(
                "QWEN_GPU_BUSY",
                "当前有 Qwen 生成任务正在运行，不能关闭 GPU",
                status_code=409,
            )

        worker = _worker_snapshot()
        if not worker.get("ready"):
            raise QwenGpuControlError(
                "QWEN_GPU_NOT_READY",
                "Qwen Worker 尚未 Ready，不能关闭 GPU",
                status_code=409,
            )

        operation = _stop_instance(config)
        return {
            "accepted": True,
            "display_status": "STOPPING",
            "vm_status": vm_status,
            "operation": operation.get("name"),
        }
    except QwenGpuControlError as exc:
        raise _http_error(exc) from exc
    except Exception as exc:
        logger.exception("qwen_gpu_stop_failed")
        raise _http_error(
            QwenGpuControlError("QWEN_GPU_STOP_FAILED", str(exc)[:1000])
        ) from exc


__all__ = ["QwenGpuControlError", "QwenGpuConfig", "router"]
