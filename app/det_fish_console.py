from __future__ import annotations

import io
import json
import os
import re
from typing import Any

import google.auth
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from google.auth.transport.requests import AuthorizedSession
from google.cloud import storage
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db import get_db
from app.detector_parity_api import _normalized_xywh, _overlay, _pixel_box
from app.detector_runtime import DETECTOR_MODEL_FILENAME, DETECTOR_MODEL_VERSION, detect, normalize_android_source
from app.factory import get_bucket_name
from app.platform.services.crop_dataset import accepted_pool_snapshot
from app.recognition_pipeline import assess_detections, boundary_debug_payload, load_contract


router = APIRouter(tags=["det-fish-console"])

DATASET_VERSION = "DET_DS_v0.2"
MODEL_VERSION = "DET_FISH_v0.2"
TRAINING_JOB_NAME = os.getenv("DETECTOR_TRAINING_JOB_NAME", "yujian-detector-trainer").strip()
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT", "gemini-api-project-503706")).strip()
GPU_REGION = os.getenv("DETECTOR_GPU_REGION", "asia-southeast1").strip()
MAX_TEST_IMAGE_BYTES = 25 * 1024 * 1024
ALLOWED_MODEL_VERSIONS = {"DET_FISH_v0.1", "DET_FISH_v0.2"}


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


def _model_summary(bucket, version: str) -> dict[str, Any]:
    prefix = "models/" + version + "/"
    try:
        metadata = _read_json_blob(bucket, prefix + "detector_metadata.json")
        onnx_exists = bucket.blob(prefix + DETECTOR_MODEL_FILENAME).exists(timeout=10)
    except Exception:
        metadata = None
        onnx_exists = False
    metadata_matches = bool(metadata and metadata.get("model_version") == version)
    available = bool(
        metadata_matches
        and onnx_exists
        and re.fullmatch(r"[0-9a-f]{64}", str(metadata.get("onnx_sha256") or ""))
    )
    return {
        "model_version": version,
        "available": available,
        "reason": None if available else "尚未找到完整且元数据匹配的 ONNX 模型产物",
        "is_configured_production_version": version == DETECTOR_MODEL_VERSION,
        "dataset_version": metadata.get("dataset_version") if metadata_matches else None,
        "onnx_sha256": metadata.get("onnx_sha256") if metadata_matches else None,
        "epochs_completed": ((metadata.get("training") or {}).get("epochs_completed") if metadata_matches else None),
        "ap50": ((metadata.get("training") or {}).get("ap50") if metadata_matches else None),
        "ap50_95": ((metadata.get("training") or {}).get("ap50_95") if metadata_matches else None),
    }


def _gate_flag(report: dict[str, Any], name: str) -> bool:
    nested = report.get("training_gate") or report.get("gate") or {}
    return report.get(name) is True or (isinstance(nested, dict) and nested.get(name) is True)


def _dataset_gate(report: dict[str, Any] | None, archive_exists: bool) -> dict[str, Any]:
    blockers: list[str] = []
    if not report:
        return {
            "ready": False,
            "blockers": ["尚未找到 DET_DS_v0.2/bootstrap_report.json"],
            "counts": {},
        }

    if report.get("dataset_version") != DATASET_VERSION:
        blockers.append("数据集版本不是 DET_DS_v0.2")
    if report.get("status") != "FROZEN_READY_FOR_TRAINING":
        blockers.append("数据集尚未标记为 FROZEN_READY_FOR_TRAINING")
    source = str(report.get("source") or report.get("label_source") or "").strip().lower()
    if source not in {"accepted_bbox_human_review", "human_reviewed_accepted_bbox"}:
        blockers.append("标签来源没有明确声明为人工 accepted_bbox")
    if not _gate_flag(report, "human_review_verified"):
        blockers.append("缺少人工标签核验通过证据")
    if not _gate_flag(report, "held_out_exclusions_verified"):
        blockers.append("缺少 Case A / Case B / 04-181200 留出集排除证据")
    split_integrity = report.get("split_integrity")
    nested_gate = report.get("training_gate") or report.get("gate") or {}
    if split_integrity != "PASS" and not (isinstance(nested_gate, dict) and nested_gate.get("split_integrity") == "PASS"):
        blockers.append("缺少按来源分组的数据泄漏检查 PASS")
    if not _gate_flag(report, "negative_coverage_verified"):
        blockers.append("缺少人工确认无鱼负样本覆盖通过证据")
    manifest_sha = str(report.get("manifest_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest_sha):
        blockers.append("缺少有效的 manifest_sha256")
    if not archive_exists:
        blockers.append("缺少 DET_DS_v0.2.tar.gz 冻结数据包")

    splits = report.get("splits") or {}
    train = splits.get("train") or {}
    val = splits.get("val") or {}
    test = splits.get("test") or {}
    counts = {
        "train_images": int(train.get("images") or 0),
        "train_boxes": int(train.get("annotations") or 0),
        "val_images": int(val.get("images") or 0),
        "val_boxes": int(val.get("annotations") or 0),
        "test_images": int(test.get("images") or 0),
        "test_boxes": int(test.get("annotations") or 0),
        "human_reviewed_negatives": int(report.get("negatives") or 0),
    }
    if counts["train_images"] < 40:
        blockers.append("训练集完整场景图片不足 40 张")
    if counts["train_boxes"] < 30:
        blockers.append("训练集人工鱼体框不足 30 个")
    if counts["val_boxes"] < 3:
        blockers.append("验证集人工鱼体框不足 3 个")
    if counts["test_images"] < 1:
        blockers.append("独立测试集为空")
    if counts["human_reviewed_negatives"] < 1:
        blockers.append("缺少人工确认的无鱼负样本")

    return {"ready": not blockers, "blockers": blockers, "counts": counts}


def _execution_status(execution: dict[str, Any]) -> str:
    if execution.get("completionTime"):
        return "FAILED" if int(execution.get("failedCount") or 0) > 0 else "SUCCEEDED"
    if int(execution.get("runningCount") or 0) > 0 or execution.get("startTime"):
        return "RUNNING"
    return "QUEUED"


def _list_job_executions(limit: int = 8) -> list[dict[str, Any]]:
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    session = AuthorizedSession(credentials)
    url = (
        "https://run.googleapis.com/v2/projects/"
        + GCP_PROJECT_ID
        + "/locations/"
        + GPU_REGION
        + "/jobs/"
        + TRAINING_JOB_NAME
        + "/executions"
    )
    response = session.get(url, params={"pageSize": limit}, timeout=20)
    if response.status_code >= 400:
        raise RuntimeError("无法读取 Cloud Run 训练任务历史（HTTP " + str(response.status_code) + "）")
    payload = response.json()
    executions = []
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
    return executions


@router.get("/api/det-fish/models")
def det_fish_models() -> dict[str, Any]:
    try:
        bucket = _bucket()
        return {
            "models": [_model_summary(bucket, version) for version in ("DET_FISH_v0.1", "DET_FISH_v0.2")],
            "default_model_version": DETECTOR_MODEL_VERSION,
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail="无法读取 DET_FISH 模型仓库") from exc


@router.get("/api/det-fish/training/status")
def det_fish_training_status(db: Session = Depends(get_db)) -> dict[str, Any]:
    try:
        bucket = _bucket()
        report = _read_json_blob(bucket, "detector-datasets/" + DATASET_VERSION + "/bootstrap_report.json")
        archive_exists = bucket.blob("detector-datasets/" + DATASET_VERSION + "/" + DATASET_VERSION + ".tar.gz").exists(timeout=10)
        gate = _dataset_gate(report, archive_exists)
        model = _model_summary(bucket, MODEL_VERSION)
        pool = accepted_pool_snapshot(db)
        try:
            executions = _list_job_executions()
            execution_history_error = None
        except Exception:
            executions = []
            execution_history_error = "当前运行身份无法读取 Cloud Run 训练执行历史"
        return {
            "accepted_pool": {
                "accepted_images": int(pool.get("count") or 0),
                "accepted_bbox": int(pool.get("accepted_bbox_pool_count") or 0),
            },
            "dataset": {
                "version": DATASET_VERSION,
                "status": (report or {}).get("status", "NOT_BUILT"),
                "archive_available": archive_exists,
                "gate": gate,
            },
            "model": model,
            "execution_history": executions,
            "execution_history_error": execution_history_error,
            "training_config": {
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
def det_fish_training_start(payload: DetectorTrainingStart) -> dict[str, Any]:
    try:
        bucket = _bucket()
        report = _read_json_blob(bucket, "detector-datasets/" + DATASET_VERSION + "/bootstrap_report.json")
        archive_exists = bucket.blob("detector-datasets/" + DATASET_VERSION + "/" + DATASET_VERSION + ".tar.gz").exists(timeout=10)
        gate = _dataset_gate(report, archive_exists)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="无法验证 DET_DS_v0.2 数据集门禁") from exc
    if not gate["ready"]:
        raise HTTPException(
            status_code=409,
            detail={"message": "DET_DS_v0.2 尚未通过训练门禁，已阻止启动 GPU 任务。", "blockers": gate["blockers"]},
        )

    try:
        executions = _list_job_executions()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="无法确认现有训练任务状态；为避免重复启动，本次已停止。") from exc
    active = [item for item in executions if item["status"] in {"RUNNING", "QUEUED"}]
    if active:
        raise HTTPException(
            status_code=409,
            detail={"message": "已有 DET_FISH 训练任务正在运行或排队。", "execution_id": active[0]["execution_id"]},
        )

    try:
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        session = AuthorizedSession(credentials)
        url = (
            "https://run.googleapis.com/v2/projects/"
            + GCP_PROJECT_ID
            + "/locations/"
            + GPU_REGION
            + "/jobs/"
            + TRAINING_JOB_NAME
            + ":run"
        )
        env = [
            {"name": "GCS_BUCKET", "value": get_bucket_name()},
            {"name": "DETECTOR_DATASET_VERSION", "value": DATASET_VERSION},
            {"name": "DETECTOR_MODEL_VERSION", "value": MODEL_VERSION},
            {"name": "DETECTOR_EPOCHS", "value": str(payload.epochs)},
            {"name": "DETECTOR_BATCH_SIZE", "value": str(payload.batch_size)},
            {"name": "DETECTOR_SEED", "value": str(payload.seed)},
            {"name": "DETECTOR_EVAL_INTERVAL", "value": "2"},
            {"name": "APP_GIT_COMMIT", "value": os.getenv("APP_GIT_COMMIT", "unknown").strip() or "unknown"},
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
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc) if isinstance(exc, RuntimeError) else "启动 Cloud Run 训练任务失败") from exc


async def _read_test_image(file: UploadFile) -> bytes:
    if file.content_type and file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="仅支持 JPG、PNG、WEBP 图片")
    data = await file.read(MAX_TEST_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="请选择一张照片")
    if len(data) > MAX_TEST_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="图片不能超过 25 MiB")
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
                "bbox_pixel": _pixel_box(item.box, source.width, source.height),
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
                "bbox_pixel": _pixel_box(primary.box, source.width, source.height),
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
                "width": source.width,
                "height": source.height,
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
