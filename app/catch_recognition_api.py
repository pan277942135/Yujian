"""Product-facing, non-persistent adapter for the registered recognition pipeline.

This module deliberately delegates inference to ``app.inference_api``. It does
not select a detector version, write image objects, or create training data.
"""

from __future__ import annotations

import io
import logging
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.inference_api import MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS, _inspect_image, _predict_bytes
from app.models import ModelVersion


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/product/catch-recognition", tags=["catch-recognition-product"])
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}


def result_state_for_prediction(result: dict) -> str:
    """Adapt actual pipeline output to the existing Recognition Result 3+2 states.

    The HIGH/MEDIUM/LOW decision mirrors the observed V1.1 result contract in
    Yujian_App/main. It does not change the model's logits, gate, or thresholds.
    """
    if not result.get("classification_ran"):
        if str(result.get("status_wire") or "").lower() == "no_fish":
            return "ERROR_NO_FISH"
        return "ERROR_IMAGE_QUALITY"

    top1 = result.get("top1") or {}
    confidence = top1.get("confidence")
    if confidence is None:
        return "TECHNICAL_FAILURE"
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return "TECHNICAL_FAILURE"
    if confidence < 0.45:
        return "RESULT_LOW"
    top3 = result.get("top3") or []
    if len(top3) > 1:
        try:
            margin = confidence - float(top3[1].get("confidence"))
        except (TypeError, ValueError):
            return "TECHNICAL_FAILURE"
        if margin < 0.12:
            return "RESULT_MEDIUM"
    return "RESULT_HIGH"


def _active_classifier(db: Session) -> ModelVersion:
    rows = db.scalars(
        select(ModelVersion)
        .where(ModelVersion.is_production.is_(True), ModelVersion.status == "PRODUCTION")
        .order_by(ModelVersion.published_at.desc(), ModelVersion.created_at.desc())
    ).all()
    if len(rows) != 1:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "PRODUCTION_CLASSIFIER_UNAVAILABLE",
                "message": "当前没有唯一可用的生产分类模型，请稍后重试。",
                "retryable": True,
            },
        )
    if not rows[0].artifact_uri:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "PRODUCTION_CLASSIFIER_ARTIFACT_UNAVAILABLE",
                "message": "生产分类模型产物暂不可用，请稍后重试。",
                "retryable": True,
            },
        )
    return rows[0]


def _response_payload(result: dict, request_id: str) -> dict:
    detector = result.get("detector") or {}
    quality_gate = result.get("quality_gate") or {}
    crop = result.get("crop") or None
    stages = ((result.get("recognition_log") or {}).get("stages") or [])
    top1 = result.get("top1") or None
    top3 = result.get("top3") or []
    return {
        "request_id": request_id,
        "pipeline_version": result.get("pipeline_version"),
        "result_state": result_state_for_prediction(result),
        "input": result.get("input") or {},
        "status": result.get("status"),
        "status_wire": result.get("status_wire"),
        "message": result.get("message"),
        "guidance": result.get("guidance"),
        "detector": {
            "model_version": detector.get("model_version"),
            "onnx_sha256": detector.get("onnx_sha256"),
            "latency_ms": detector.get("latency_ms"),
            "detections": detector.get("detections") or [],
            "quality_gate": quality_gate,
            "coordinate_space": "EXIF_TRANSPOSED_SOURCE_IMAGE_NORMALIZED",
        },
        "crop": crop,
        "classifier": {
            "model_version": result.get("model_version") if result.get("classification_ran") else None,
            "model_status": result.get("model_status") if result.get("classification_ran") else None,
            "pipeline_type": result.get("pipeline_type"),
            "input": result.get("classifier_input"),
            "top1": top1,
            "top3": top3,
            "latency_ms": result.get("classifier_latency_ms"),
        },
        "timing": {
            "total_ms": result.get("latency_ms"),
            "detector_ms": detector.get("latency_ms"),
            "classifier_ms": result.get("classifier_latency_ms"),
            "stages": [
                {key: stage.get(key) for key in ("name", "status", "duration_ms")}
                for stage in stages
            ],
        },
        # SAM remains an explicitly unqualified optional enhancement. No mask
        # or precise fish contour is emitted until its real-image gate passes.
        "segmentation": {
            "status": "UNAVAILABLE",
            "reason_code": "SAM_NOT_QUALIFIED",
            "model_version": None,
            "quality": None,
            "mask": None,
        },
    }


def _read_product_image(file: UploadFile) -> tuple[bytes, dict]:
    data = file.file.read(MAX_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail={"code": "IMAGE_EMPTY", "message": "请先选择照片。", "retryable": False})
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail={"code": "IMAGE_TOO_LARGE", "message": f"照片不能超过 {MAX_IMAGE_BYTES // (1024 * 1024)} MB。", "retryable": False},
        )
    try:
        # Read the format from the source image before EXIF transpose. Pillow
        # may drop ``Image.format`` on the transposed copy; treating that as
        # JPEG would accidentally accept unsupported GIF/TIFF content.
        with Image.open(io.BytesIO(data)) as source:
            source_format = str(source.format or "").upper()
        meta = _inspect_image(data)
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail={"code": "IMAGE_DECODE_FAILED", "message": "无法识别图片格式。", "retryable": False}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"code": "IMAGE_DECODE_FAILED", "message": str(exc), "retryable": False}) from exc
    if source_format not in ALLOWED_IMAGE_FORMATS:
        raise HTTPException(
            status_code=415,
            detail={"code": "IMAGE_FORMAT_UNSUPPORTED", "message": "仅支持 JPEG、PNG、WEBP 图片。", "retryable": False},
        )
    if int(meta.get("width") or 0) * int(meta.get("height") or 0) > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=400,
            detail={"code": "IMAGE_PIXELS_TOO_LARGE", "message": "照片分辨率过高，请压缩后重试。", "retryable": False},
        )
    return data, meta


@router.post("/infer")
def infer_catch_photo(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    request_id = f"REC_{uuid4().hex}"
    try:
        data, _meta = _read_product_image(file)
        model = _active_classifier(db)
        result = _predict_bytes(db, model.model_version, data, include_intermediates=False)
        payload = _response_payload(result, request_id)
        logger.info(
            "product_recognition_complete request_id=%s detector=%s classifier=%s state=%s total_ms=%s",
            request_id,
            payload["detector"].get("model_version"),
            payload["classifier"].get("model_version"),
            payload["result_state"],
            payload["timing"].get("total_ms"),
        )
        return JSONResponse(payload, headers={"Cache-Control": "no-store"})
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"code": "IMAGE_INVALID", "message": str(exc), "retryable": False}) from exc
    except RuntimeError as exc:
        logger.warning("product_recognition_unavailable request_id=%s error_type=%s", request_id, type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail={"code": "MODEL_UNAVAILABLE", "message": "识别服务暂不可用，请稍后重试。", "retryable": True, "request_id": request_id},
        ) from exc
    except Exception as exc:
        logger.exception("product_recognition_failed request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail={"code": "RECOGNITION_INTERNAL_ERROR", "message": "本次识别未完成，请重试。", "retryable": True, "request_id": request_id},
        ) from exc


__all__ = ["_active_classifier", "_response_payload", "_read_product_image", "result_state_for_prediction", "router"]
