from __future__ import annotations

import hashlib
import io
import json
import os
import re
import threading
from typing import Any

import google.auth
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from google.auth.transport.requests import AuthorizedSession
from google.cloud import storage
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db import get_db
from app.detector_parity_api import _normalized_xywh, _overlay, _pixel_box
from app.detector_runtime import (
    DETECTOR_MODEL_FILENAME,
    DETECTOR_MODEL_VERSION,
    detect,
    load_detector_for_version,
    normalize_android_source,
)
from app.factory import get_bucket_name
from app.platform.services.crop_dataset import accepted_pool_snapshot
from app.recognition_pipeline import assess_detections, boundary_debug_payload, load_contract
from trainer.detector_dataset_gate import DATASET_VERSION, validate_v02_dataset


router = APIRouter(tags=["det-fish-console"])

MODEL_VERSION = os.getenv(
    "DETECTOR_TRAINING_MODEL_VERSION",
    {
        "DET_DS_v0.2": "DET_FISH_v0.2",
        "DET_DS_v0.3": "DET_FISH_v0.3",
    }.get(DATASET_VERSION, ""),
).strip()
TRAINING_JOB_NAME = os.getenv("DETECTOR_TRAINING_JOB_NAME", "yujian-detector-trainer").strip()
# The region default mirrors scripts/deploy_detector_training_gcp.sh. The GCP
# project is deliberately environment-only: a historical project fallback could
# direct an otherwise valid request to the wrong account.
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT", "")).strip()
GPU_REGION = os.getenv("DETECTOR_GPU_REGION", "asia-southeast1").strip()
MAX_TEST_IMAGE_BYTES = 25 * 1024 * 1024
MAX_TEST_IMAGE_PIXELS = 40_000_000
ALLOWED_MODEL_VERSIONS = {"DET_FISH_v0.1", "DET_FISH_v0.2", "DET_FISH_v0.3"}
_TRAINING_LOCK_ID = 0x594A4446
_LOCAL_TRAINING_LOCK = threading.Lock()


class DetectorTrainingStart(BaseModel):
    epochs: int = Field(default=30, ge=1, le=100)
    batch_size: int = Field(default=16, ge=1, le=64)
    seed: int = Field(default=20261020, ge=0, le=2147483647)


def _bucket():
    return storage.Client().bucket(get_bucket_name())


def _read_json_blob(bucket, object_name: str) -> dict[str, Any] | None:
    blob = bucket.blob(object_name)
    if not blob.exists(timeout=10):
        return None
    try:
        value = json.loads(blob.download_as_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid JSON artifact: " + object_name) from exc
    return value if isinstance(value, dict) else None


def _blob_sha256(blob) -> str | None:
    """Hash a GCS object as a stream so a frozen archive is not loaded in RAM."""
    digest = hashlib.sha256()
    try:
        with blob.open("rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except Exception:
        return None
    return digest.hexdigest()


def _dataset_artifacts(bucket) -> tuple[dict[str, Any] | None, bytes | None, str | None, bool, bool]:
    prefix = "detector-datasets/" + DATASET_VERSION
    report = _read_json_blob(bucket, prefix + "/bootstrap_report.json")
    manifest_blob = bucket.blob(prefix + "/bootstrap_manifest.json")
    archive_blob = bucket.blob(prefix + "/" + DATASET_VERSION + ".tar.gz")
    try:
        manifest_bytes = manifest_blob.download_as_bytes(timeout=60) if manifest_blob.exists(timeout=10) else None
    except Exception:
        manifest_bytes = None
    archive_exists = bool(archive_blob.exists(timeout=10))
    archive_sha256 = _blob_sha256(archive_blob) if archive_exists else None
    manifest_verified = bool(
        report
        and manifest_bytes is not None
        and hashlib.sha256(manifest_bytes).hexdigest() == str(report.get("manifest_sha256") or "")
    )
    return report, manifest_bytes, archive_sha256, archive_exists, manifest_verified


def _model_summary(bucket, version: str) -> dict[str, Any]:
    prefix = "models/" + version + "/"
    reason = ""
    metadata = None
    actual_sha = None
    onnx_bytes = None
    output_shape = None
    metadata_exists = False
    onnx_exists = False
    try:
        metadata_blob = bucket.blob(prefix + "detector_metadata.json")
        metadata_exists = bool(metadata_blob.exists(timeout=10))
        metadata = _read_json_blob(bucket, prefix + "detector_metadata.json") if metadata_exists else None
        if metadata is None:
            reason = "缺少 detector_metadata.json"
        else:
            onnx_blob = bucket.blob(prefix + DETECTOR_MODEL_FILENAME)
            onnx_exists = bool(onnx_blob.exists(timeout=10))
            if not onnx_exists:
                reason = "缺少 ONNX 文件"
            else:
                # Use the same loader as the real test endpoint. It verifies the
                # artifact SHA, recognition contract, ONNX input contract, and
                # creates an actual ONNX Runtime session before the UI lists a
                # model as selectable.
                runtime_model = load_detector_for_version(version)
                actual_sha = runtime_model.onnx_sha256
                onnx_bytes = runtime_model.onnx_bytes
                output_shape = list(runtime_model.session.get_outputs()[0].shape)
                if metadata.get("model_version") != version:
                    reason = "metadata 模型版本与目录不一致"
                elif metadata.get("model_family") != "YOLOX_NANO":
                    reason = "模型架构不是 YOLOX_NANO"
                elif not re.fullmatch(r"[0-9a-f]{64}", str(metadata.get("onnx_sha256") or "")):
                    reason = "metadata 缺少有效 ONNX SHA-256"
                elif metadata.get("onnx_sha256") != actual_sha:
                    reason = "ONNX 实际 SHA-256 与 metadata 不一致"
                elif int(metadata.get("onnx_bytes") or 0) != onnx_bytes:
                    reason = "ONNX 实际字节数与 metadata 不一致"
                elif len(output_shape) != 3 or output_shape[0] != 1 or not isinstance(output_shape[-1], int) or output_shape[-1] < 6:
                    reason = "ONNX 实际输出形状不符合 [1, 候选数, 至少 6 列] 合同"
    except Exception:
        reason = "模型无法通过生产 ONNX Runtime 加载与工件校验"
    available = bool(metadata and not reason and actual_sha)
    metadata_matches = bool(metadata and metadata.get("model_version") == version)
    return {
        "model_version": version,
        "available": available,
        "artifact_objects_present": metadata_exists or onnx_exists,
        "reason": None if available else (reason or "尚未找到完整模型产物"),
        "is_configured_production_version": version == DETECTOR_MODEL_VERSION,
        "dataset_version": metadata.get("dataset_version") if metadata_matches else None,
        "onnx_sha256": actual_sha if available else None,
        "onnx_bytes": onnx_bytes if available else None,
        "output_shape": output_shape if available else None,
        "epochs_completed": ((metadata.get("training") or {}).get("epochs_completed") if metadata_matches else None),
        "ap50": ((metadata.get("training") or {}).get("ap50") if metadata_matches else None),
        "ap50_95": ((metadata.get("training") or {}).get("ap50_95") if metadata_matches else None),
    }


def _dataset_gate(
    report: dict[str, Any] | None,
    manifest_bytes: bytes | None,
    archive_sha256: str | None,
) -> dict[str, Any]:
    return validate_v02_dataset(report, manifest_bytes, archive_sha256)


def _configured_cloud_run_context() -> tuple[str, str, str]:
    if not GCP_PROJECT_ID:
        raise RuntimeError("GCP_PROJECT_ID / GOOGLE_CLOUD_PROJECT 未配置；未选择 Cloud Run 项目")
    if not GPU_REGION:
        raise RuntimeError("DETECTOR_GPU_REGION 未配置；未选择训练 Job 区域")
    if not TRAINING_JOB_NAME:
        raise RuntimeError("DETECTOR_TRAINING_JOB_NAME 未配置；未选择训练 Job")
    return GCP_PROJECT_ID, GPU_REGION, TRAINING_JOB_NAME


def _cloud_run_job_url() -> str:
    project_id, region, job_name = _configured_cloud_run_context()
    return f"https://run.googleapis.com/v2/projects/{project_id}/locations/{region}/jobs/{job_name}"


def _validate_detector_job(job: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    task = ((job.get("template") or {}).get("template") or {})
    containers = task.get("containers") or []
    if not containers:
        return ["训练 Job 没有可核验的容器配置"]
    container = containers[0]
    image = str(container.get("image") or "")
    if "/detector-trainer:" not in image:
        if "/detector-trainer@sha256:" not in image:
            blockers.append("训练 Job 镜像不是固定 digest 的仓库 detector-trainer 镜像")
    env = {
        str(item.get("name") or ""): str(item.get("value") or "")
        for item in container.get("env") or []
        if isinstance(item, dict)
    }
    if env.get("DETECTOR_DATASET_VERSION") != DATASET_VERSION:
        blockers.append(
            f"训练 Job 未显式配置 DETECTOR_DATASET_VERSION={DATASET_VERSION}；禁止使用其他数据集版本启动"
        )
    if not MODEL_VERSION:
        blockers.append("控制台未配置受支持的数据集/模型版本组合")
    elif env.get("DETECTOR_MODEL_VERSION") != MODEL_VERSION:
        blockers.append(
            f"训练 Job 未显式配置 DETECTOR_MODEL_VERSION={MODEL_VERSION}"
        )
    build_sha = env.get("APP_GIT_COMMIT", "").strip()
    if not build_sha:
        blockers.append("训练 Job 缺少构建代码提交 SHA，无法核对训练镜像来源")
    elif not re.fullmatch(r"[0-9a-fA-F]{40}", build_sha):
        blockers.append("训练 Job 的 APP_GIT_COMMIT 不是完整 Git SHA")
    if "@sha256:" not in image:
        blockers.append("训练 Job 镜像未固定为 Artifact Registry digest；禁止使用可变标签启动训练")
    elif not re.search(r"@sha256:[0-9a-f]{64}$", image):
        blockers.append("训练 Job 镜像 digest 格式无效")
    command = " ".join(str(item) for item in (container.get("command") or []) + (container.get("args") or []))
    if command and "trainer.train_detector" not in command:
        blockers.append("训练 Job 覆盖了非 DET_FISH 训练入口")
    if not task.get("serviceAccount"):
        blockers.append("训练 Job 未配置运行服务账号")

    limits = container.get("resources", {}).get("limits", {})
    try:
        cpu = str(limits.get("cpu", "0"))
        cpu_value = float(cpu[:-1]) / 1000.0 if cpu.endswith("m") else float(cpu)
        cpu_ok = cpu_value >= 4
    except (TypeError, ValueError):
        cpu_ok = False
    memory = str(limits.get("memory") or "")
    memory_ok = memory.lower() in {"16gi", "17g", "16384mi"} or bool(re.fullmatch(r"(?:1[6-9]|[2-9][0-9])Gi", memory, re.I))
    node_selector = task.get("nodeSelector") or {}
    accelerator = str(node_selector.get("run.googleapis.com/accelerator") or "").lower()
    gpu_count = str(limits.get("nvidia.com/gpu") or "")
    if gpu_count != "1":
        blockers.append("训练 Job 未配置 1 张 NVIDIA GPU")
    if accelerator != "nvidia-l4":
        blockers.append("训练 Job GPU 型号不是 NVIDIA L4")
    if not cpu_ok or not memory_ok:
        blockers.append("训练 Job 资源低于 L4 所需的 4 CPU / 16 GiB 内存")
    if int(task.get("maxRetries") or 0) != 0:
        blockers.append("训练 Job maxRetries 不是 0")
    if not task.get("gpuZonalRedundancyDisabled"):
        blockers.append("训练 Job 未关闭 GPU 区域冗余")
    return blockers


def _job_configuration() -> dict[str, Any]:
    try:
        url = _cloud_run_job_url()
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        response = AuthorizedSession(credentials).get(url, timeout=20)
        if response.status_code == 404:
            return {"status": "BLOCKED", "verified": False, "blockers": ["Cloud Run DET_FISH 训练 Job 不存在"]}
        if response.status_code >= 400:
            return {"status": "NOT_VERIFIED", "verified": False, "blockers": [f"无法读取训练 Job 配置（HTTP {response.status_code}）"]}
        payload = response.json()
        blockers = _validate_detector_job(payload)
        task = ((payload.get("template") or {}).get("template") or {})
        container = next(iter(task.get("containers") or []), {})
        env = {
            str(item.get("name") or ""): str(item.get("value") or "")
            for item in container.get("env") or []
            if isinstance(item, dict)
        }
        return {
            "status": "READY" if not blockers else "BLOCKED",
            "verified": not blockers,
            "blockers": blockers,
            "project_id": GCP_PROJECT_ID,
            "region": GPU_REGION,
            "job_name": TRAINING_JOB_NAME,
            "image": container.get("image"),
            "image_digest": str(container.get("image") or "").rsplit("@", 1)[-1] if "@sha256:" in str(container.get("image") or "") else None,
            "runtime_service_account": task.get("serviceAccount"),
            "dataset_version": env.get("DETECTOR_DATASET_VERSION"),
            "model_version": env.get("DETECTOR_MODEL_VERSION"),
            "build_commit": env.get("APP_GIT_COMMIT"),
            "cpu": (container.get("resources") or {}).get("limits", {}).get("cpu"),
            "memory": (container.get("resources") or {}).get("limits", {}).get("memory"),
            "gpu_count": (container.get("resources") or {}).get("limits", {}).get("nvidia.com/gpu"),
            "gpu_type": (task.get("nodeSelector") or {}).get("run.googleapis.com/accelerator"),
        }
    except Exception as exc:
        message = str(exc) if isinstance(exc, RuntimeError) else "当前运行身份无法核验 Cloud Run Job 配置"
        return {"status": "NOT_VERIFIED", "verified": False, "blockers": [message]}


def _acquire_training_lock(db: Session) -> str:
    """Serialize launch requests across app instances using Cloud SQL advisory lock."""
    dialect = db.get_bind().dialect.name
    if dialect == "postgresql":
        acquired = bool(db.execute(text("SELECT pg_try_advisory_lock(:lock_id)"), {"lock_id": _TRAINING_LOCK_ID}).scalar())
        return "postgresql" if acquired else "busy"
    if dialect == "sqlite":
        return "sqlite" if _LOCAL_TRAINING_LOCK.acquire(blocking=False) else "busy"
    raise RuntimeError("数据库不支持安全的 DET_FISH 训练启动锁")


def _release_training_lock(db: Session, lock_kind: str) -> None:
    if lock_kind == "postgresql":
        db.execute(text("SELECT pg_advisory_unlock(:lock_id)"), {"lock_id": _TRAINING_LOCK_ID})
        db.commit()
    elif lock_kind == "sqlite":
        _LOCAL_TRAINING_LOCK.release()


def _execution_status(execution: dict[str, Any]) -> str:
    if execution.get("completionTime"):
        return "FAILED" if int(execution.get("failedCount") or 0) > 0 else "SUCCEEDED"
    if int(execution.get("runningCount") or 0) > 0 or execution.get("startTime"):
        return "RUNNING"
    return "QUEUED"


def _list_job_executions(limit: int | None = 8) -> list[dict[str, Any]]:
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    session = AuthorizedSession(credentials)
    url = _cloud_run_job_url() + "/executions"
    executions = []
    page_token = None
    while True:
        page_size = 1000 if limit is None else min(max(limit - len(executions), 1), 1000)
        params = {"pageSize": page_size}
        if page_token:
            params["pageToken"] = page_token
        response = session.get(url, params=params, timeout=20)
        if response.status_code >= 400:
            raise RuntimeError("无法读取 Cloud Run 训练任务历史（HTTP " + str(response.status_code) + ")")
        payload = response.json()
        for item in payload.get("executions") or []:
            executions.append(
                {
                    "execution_id": str(item.get("name") or "").rsplit("/", 1)[-1],
                    "status": _execution_status(item),
                    "created_at": item.get("createTime"),
                    "started_at": item.get("startTime"),
                    "completed_at": item.get("completionTime"),
                    "running_count": int(item.get("runningCount") or 0),
                    "succeeded_count": int(item.get("succeededCount") or 0),
                    "failed_count": int(item.get("failedCount") or 0),
                }
            )
        page_token = payload.get("nextPageToken")
        if not page_token or (limit is not None and len(executions) >= limit):
            break
    if limit is not None:
        executions = executions[:limit]
    return executions


@router.get("/api/det-fish/models")
def det_fish_models() -> dict[str, Any]:
    try:
        bucket = _bucket()
        return {
            "models": [_model_summary(bucket, version) for version in sorted(ALLOWED_MODEL_VERSIONS)],
            "default_model_version": DETECTOR_MODEL_VERSION,
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail="无法读取 DET_FISH 模型仓库") from exc


@router.get("/api/det-fish/training/status")
def det_fish_training_status(db: Session = Depends(get_db)) -> dict[str, Any]:
    try:
        bucket = _bucket()
        report, manifest_bytes, archive_sha256, archive_exists, manifest_verified = _dataset_artifacts(bucket)
        gate = _dataset_gate(report, manifest_bytes, archive_sha256)
        model = _model_summary(bucket, MODEL_VERSION)
        pool = accepted_pool_snapshot(db)
        accepted_pool_reconciliation = {
            "status": "NOT_VERIFIED",
            "verified": False,
            "blockers": ["冻结报告尚未声明与当前 Accepted Pool 的记录数对照"],
        }
        declared_source_records = (report or {}).get("source_accepted_bbox_record_count") if isinstance(report, dict) else None
        if isinstance(declared_source_records, int):
            live_source_records = int(pool.get("accepted_bbox_pool_count") or 0)
            matches = declared_source_records == live_source_records
            accepted_pool_reconciliation = {
                "status": "PASS" if matches else "BLOCKED",
                "verified": matches,
                "manifest_source_records": declared_source_records,
                "live_accepted_bbox_records": live_source_records,
                "blockers": [] if matches else ["冻结构建源记录数与当前已审核 Accepted Pool 数不一致"],
            }
            if not matches:
                gate["ready"] = False
                gate["blockers"] = list(dict.fromkeys(gate["blockers"] + accepted_pool_reconciliation["blockers"]))
        try:
            executions = _list_job_executions()
            execution_history_error = None
        except Exception:
            executions = []
            execution_history_error = "当前运行身份无法读取 Cloud Run 训练执行历史"
        job_configuration = _job_configuration()
        training_blockers = list(gate["blockers"])
        training_blockers.extend(accepted_pool_reconciliation["blockers"] if not accepted_pool_reconciliation["verified"] else [])
        training_blockers.extend(job_configuration["blockers"])
        if model.get("artifact_objects_present"):
            training_blockers.append(
                f"{MODEL_VERSION} 目标产物路径已存在；为避免覆盖不可变模型工件，已阻止重复训练"
            )
        if execution_history_error:
            training_blockers.append(execution_history_error)
        active = [item for item in executions if item["status"] in {"RUNNING", "QUEUED"}]
        if active:
            training_blockers.append("已有 DET_FISH 训练任务正在运行或排队")
        training_gate = {
            "ready": not training_blockers,
            "blockers": list(dict.fromkeys(training_blockers)),
        }
        return {
            "accepted_pool": {
                "accepted_images": int(pool.get("count") or 0),
                "accepted_bbox": int(pool.get("accepted_bbox_pool_count") or 0),
                "source": pool.get("source"),
                "note": f"Accepted Pool 记录数不代表已进入 {DATASET_VERSION} 冻结训练集",
            },
            "accepted_pool_reconciliation": accepted_pool_reconciliation,
            "dataset": {
                "version": DATASET_VERSION,
                "status": (report or {}).get("status", "NOT_BUILT"),
                "archive_available": archive_exists,
                "manifest_available": manifest_bytes is not None,
                "manifest_sha256_verified": manifest_verified,
                "archive_sha256_verified": bool(
                    report and report.get("archive_sha256") and archive_sha256 == report.get("archive_sha256")
                ),
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest() if manifest_bytes is not None else None,
                "archive_sha256": archive_sha256,
                "source": (report or {}).get("source"),
                "gate": gate,
            },
            "model": model,
            "execution_history": executions,
            "execution_history_error": execution_history_error,
            "execution_history_gate": {
                "status": "PASS" if execution_history_error is None else "NOT_VERIFIED",
                "verified": execution_history_error is None,
                "blockers": [] if execution_history_error is None else [execution_history_error],
            },
            "job_configuration": job_configuration,
            "training_gate": training_gate,
            "training_config": {
                "dataset_version": DATASET_VERSION,
                "model_version": MODEL_VERSION,
                "model_family": "YOLOX_NANO",
                "input_size": 416,
                "default_epochs": 30,
                "default_batch_size": 16,
                "gpu": "NVIDIA L4",
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail="无法读取 DET_FISH 训练状态") from exc


@router.post("/api/det-fish/training/start")
def det_fish_training_start(payload: DetectorTrainingStart, db: Session = Depends(get_db)) -> dict[str, Any]:
    try:
        lock_kind = _acquire_training_lock(db)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if lock_kind == "busy":
        raise HTTPException(status_code=409, detail="已有 DET_FISH 训练启动请求正在处理；为避免重复任务，本次已停止。")
    try:
        try:
            bucket = _bucket()
            report, manifest_bytes, archive_sha256, _archive_exists, _manifest_verified = _dataset_artifacts(bucket)
            gate = _dataset_gate(report, manifest_bytes, archive_sha256)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"无法验证 {DATASET_VERSION} 冻结数据集") from exc
        if not gate["ready"]:
            raise HTTPException(
                status_code=409,
                detail={"message": f"{DATASET_VERSION} 尚未通过逐图数据与哈希门禁，已阻止启动 GPU 任务。", "blockers": gate["blockers"]},
            )

        existing_model = _model_summary(bucket, MODEL_VERSION)
        if existing_model.get("artifact_objects_present"):
            raise HTTPException(
                status_code=409,
                detail=f"{MODEL_VERSION} 目标产物路径已存在；为避免覆盖不可变模型工件，训练已停止。",
            )

        job_configuration = _job_configuration()
        if not job_configuration["verified"]:
            raise HTTPException(
                status_code=409,
                detail={"message": "Cloud Run DET_FISH Job 配置未通过核验，已阻止启动。", "blockers": job_configuration["blockers"]},
            )
        try:
            executions = _list_job_executions(limit=None)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="无法确认现有训练任务状态；为避免重复启动，本次已停止。") from exc
        active = [item for item in executions if item["status"] in {"RUNNING", "QUEUED"}]
        if active:
            raise HTTPException(
                status_code=409,
                detail={"message": "已有 DET_FISH 训练任务正在运行或排队。", "execution_id": active[0]["execution_id"]},
            )

        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        session = AuthorizedSession(credentials)
        url = _cloud_run_job_url() + ":run"
        env = [
            {"name": "GCS_BUCKET", "value": get_bucket_name()},
            {"name": "DETECTOR_DATASET_VERSION", "value": DATASET_VERSION},
            {"name": "DETECTOR_MODEL_VERSION", "value": MODEL_VERSION},
            {"name": "DETECTOR_EPOCHS", "value": str(payload.epochs)},
            {"name": "DETECTOR_BATCH_SIZE", "value": str(payload.batch_size)},
            {"name": "DETECTOR_SEED", "value": str(payload.seed)},
            {"name": "DETECTOR_EVAL_INTERVAL", "value": "2"},
        ]
        response = session.post(
            url,
            json={"overrides": {"containerOverrides": [{"env": env}], "taskCount": 1, "timeout": "3600s"}},
            timeout=30,
        )
        if response.status_code >= 400:
            raise RuntimeError("Cloud Run Job 启动失败（HTTP " + str(response.status_code) + "）")
        operation = response.json()
        return {
            "status": "QUEUED",
            "dataset_version": DATASET_VERSION,
            "model_version": MODEL_VERSION,
            "epochs": payload.epochs,
            "batch_size": payload.batch_size,
            "operation_name": operation.get("name"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc) if isinstance(exc, RuntimeError) else "启动 Cloud Run 训练任务失败") from exc
    finally:
        try:
            _release_training_lock(db, lock_kind)
        except Exception:
            db.rollback()


async def _read_test_image(file: UploadFile) -> bytes:
    if file.content_type and file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="仅支持 JPG、PNG、WEBP 图片")
    data = await file.read(MAX_TEST_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="请选择一张照片")
    if len(data) > MAX_TEST_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="图片不能超过 25 MiB")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            if probe.format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(status_code=400, detail="文件内容不是受支持的 JPG、PNG 或 WEBP 图片")
            expected_content_type = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}[probe.format]
            if file.content_type and file.content_type != expected_content_type:
                raise HTTPException(status_code=400, detail="图片 MIME 类型与实际文件内容不一致")
            width, height = probe.size
            if width <= 0 or height <= 0 or width * height > MAX_TEST_IMAGE_PIXELS:
                raise HTTPException(status_code=400, detail="图片尺寸无效或超过 40 百万像素")
            probe.verify()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="无法读取图片，请使用完整且有效的 JPG / PNG / WEBP 文件") from exc
    return data


@router.post("/api/det-fish/test")
async def det_fish_test(
    image: UploadFile = File(...),
    model_version: str = Form(default="DET_FISH_v0.1"),
):
    if model_version not in ALLOWED_MODEL_VERSIONS:
        raise HTTPException(status_code=400, detail="不支持的 DET_FISH 模型版本")
    data = await _read_test_image(image)
    try:
        with Image.open(io.BytesIO(data)) as uploaded:
            original_width, original_height = uploaded.size
            oriented = ImageOps.exif_transpose(uploaded)
            oriented_width, oriented_height = oriented.size
            oriented.close()
            image_format = uploaded.format or "unknown"
            source = normalize_android_source(uploaded)
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="无法读取图片，请使用有效的 JPG / PNG / WEBP 文件") from exc

    try:
        detector_run = detect(source, model_version=model_version)
        assessment = assess_detections(detector_run.detections)
        primary = assessment.primary
        contract = load_contract()
        detections = [
            {
                "rank": index + 1,
                "confidence": round(float(item.confidence), 6),
                "bbox_pixel": _pixel_box(item.box, oriented_width, oriented_height),
                "bbox_pixel_input": _pixel_box(item.box, source.width, source.height),
                "bbox_normalized_xywh": _normalized_xywh(item.box),
                "bbox_area_ratio": round(float(item.area_ratio), 6),
                "class_name": item.class_name,
            }
            for index, item in enumerate(detector_run.detections)
        ]
        primary_payload = None
        if primary is not None:
            primary_payload = {
                "confidence": round(float(primary.confidence), 6),
                "bbox_pixel": _pixel_box(primary.box, oriented_width, oriented_height),
                "bbox_pixel_input": _pixel_box(primary.box, source.width, source.height),
                "bbox_normalized_xywh": _normalized_xywh(primary.box),
                "bbox_area_ratio": round(float(primary.area_ratio), 6),
            }
        return {
            "model_version": detector_run.model_version,
            "onnx_sha256": detector_run.onnx_sha256,
            "input": {
                "filename": image.filename or "image",
                "format": image_format,
                "original_width": original_width,
                "original_height": original_height,
                "oriented_width": oriented_width,
                "oriented_height": oriented_height,
                "width": source.width,
                "height": source.height,
            },
            "model_input": {
                "shape": [1, 3, detector_run.input_size, detector_run.input_size],
                "dtype": "float32",
                "layout": "NCHW",
                "width": detector_run.input_size,
                "height": detector_run.input_size,
                "pixel_range": [0, 255],
                "resized_content_width": detector_run.input_draw_width,
                "resized_content_height": detector_run.input_draw_height,
                "letterbox_fill": 114,
            },
            "detector": {
                **(primary_payload or {"confidence": None, "bbox_pixel": None, "bbox_normalized_xywh": None, "bbox_area_ratio": 0.0}),
                "detections": detections,
                "assessment": assessment.status.value,
                "reason": assessment.reason,
                "selected_orientation": detector_run.selected_attempt,
                "selection_reason": detector_run.selection_reason,
                "latency_ms": detector_run.latency_ms,
                "attempts": [
                    {
                        "orientation_attempt": attempt.orientation_attempt,
                        "detection_count": attempt.detection_count,
                        "top_confidence": attempt.top_confidence,
                        "quality_status": attempt.quality_status,
                        "quality_level": attempt.quality_level,
                    }
                    for attempt in detector_run.attempt_trace
                ],
            },
            "boundary_check": boundary_debug_payload(assessment, source.width, source.height, contract),
            "overlay_data_url": _overlay(source, detector_run.detections, primary),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail="所选 DET_FISH 模型不可用或推理失败") from exc
    finally:
        source.close()
