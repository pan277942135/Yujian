from __future__ import annotations

import io
import json
import mimetypes
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from google.cloud import storage
from PIL import Image, ImageOps, UnidentifiedImageError
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.requests import Request

from app.db import get_db
from app.detector_runtime import DetectorRun, detect
from app.factory import get_bucket_name
from app.pipeline_contract import CROP_CLASSIFIER_V1, WHOLE_IMAGE_V1, validate_pipeline_type
from app.models import ModelVersion, TrainingRun
from app.recognition_pipeline import BBox, Detection, PipelineStatus, assess_detections, crop_box_pixels, load_contract

router = APIRouter(tags=["model-inference"])
templates = Jinja2Templates(directory="app/templates")

MAX_IMAGE_BYTES = int(os.getenv("INFERENCE_MAX_IMAGE_BYTES", str(15 * 1024 * 1024)))
MAX_BATCH_FILES = int(os.getenv("INFERENCE_MAX_BATCH_FILES", "20"))
MAX_BATCH_BYTES = int(os.getenv("INFERENCE_MAX_BATCH_BYTES", str(24 * 1024 * 1024)))
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("INFERENCE_LOW_CONFIDENCE_THRESHOLD", "0.55"))
MAX_IMAGE_PIXELS = int(os.getenv("INFERENCE_MAX_IMAGE_PIXELS", "40000000"))
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# These are exactly the Android MODEL_M1_v0.2 letterbox padding values.
IMAGENET_MEAN_RGB = (124, 116, 104)

_MODEL_CACHE: dict[str, "LoadedModel"] = {}
_MODEL_CACHE_LOCK = threading.Lock()


@dataclass
class LoadedModel:
    artifact_uri: str
    model: Any
    transform: Any
    classes: list[dict]
    classes_by_index: dict[int, dict]
    image_size: int


class WholeImageLetterbox:
    """MODEL_M1_v0.2 preprocessing: retain the entire detector crop, never center-crop it."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        source = image.convert("RGB")
        contained = ImageOps.contain(source, (self.size, self.size), method=Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), IMAGENET_MEAN_RGB)
        canvas.paste(contained, ((self.size - contained.width) // 2, (self.size - contained.height) // 2))
        return canvas


class CropLetterbox(WholeImageLetterbox):
    """CROP_CLASSIFIER_V1 preprocessing shared with the Android crop input.

    The pixel operation is deliberately the same centered letterbox used by
    the App classifier; the distinct type makes the model contract explicit
    and prevents a future whole-image transform from being reused silently.
    """


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"不是有效的 GCS URI：{uri}")
    body = uri[5:]
    if "/" not in body:
        raise ValueError(f"不是有效的 GCS URI：{uri}")
    return tuple(body.split("/", 1))  # type: ignore[return-value]


def _sibling_uri(artifact_uri: str, filename: str) -> str:
    bucket, object_name = _parse_gs_uri(artifact_uri)
    parent = object_name.rsplit("/", 1)[0]
    return f"gs://{bucket}/{parent}/{filename}"


def _download_bytes(client: storage.Client, uri: str) -> bytes:
    bucket_name, object_name = _parse_gs_uri(uri)
    return client.bucket(bucket_name).blob(object_name).download_as_bytes(timeout=180)


def _download_json(client: storage.Client, uri: str) -> dict:
    return json.loads(_download_bytes(client, uri).decode("utf-8"))


@lru_cache(maxsize=1)
def _torch_stack():
    try:
        import torch
        from torchvision import transforms
    except Exception as exc:  # pragma: no cover - exercised only in deployed runtime
        raise RuntimeError("Console 尚未安装 CPU PyTorch 推理运行时，请重新部署包含 inference runtime 的镜像") from exc
    threads = max(1, int(os.getenv("INFERENCE_TORCH_THREADS", "1")))
    torch.set_num_threads(threads)
    return torch, transforms


def _build_eval_transform(image_size: int, pipeline_type: str = WHOLE_IMAGE_V1):
    _torch, transforms = _torch_stack()
    pipeline_type = validate_pipeline_type(pipeline_type)
    normalize = transforms.Normalize(
        mean=list(IMAGENET_MEAN),
        std=list(IMAGENET_STD),
    )
    letterbox = CropLetterbox(image_size) if pipeline_type == CROP_CLASSIFIER_V1 else WholeImageLetterbox(image_size)
    return transforms.Compose(
        [
            letterbox,
            transforms.ToTensor(),
            normalize,
        ]
    )


def _load_model(row: ModelVersion) -> LoadedModel:
    cache_key = f"{row.model_version}:{row.artifact_uri}"
    cached = _MODEL_CACHE.get(cache_key)
    if cached:
        return cached

    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached:
            return cached

        torch, _transforms = _torch_stack()
        client = storage.Client()
        class_doc = _download_json(client, _sibling_uri(row.artifact_uri, "class_map.json"))
        metrics_doc = _download_json(client, row.metrics_uri) if row.metrics_uri else {}
        classes = sorted(list(class_doc.get("classes") or []), key=lambda item: int(item.get("class_index", 0)))
        if not classes:
            raise RuntimeError("模型 class_map.json 为空")
        image_size = int((metrics_doc.get("params") or {}).get("image_size") or 224)
        pipeline_type = validate_pipeline_type(getattr(row, "pipeline_type", WHOLE_IMAGE_V1))
        model_bytes = _download_bytes(client, row.artifact_uri)
        model = torch.jit.load(io.BytesIO(model_bytes), map_location="cpu")
        model.eval()
        loaded = LoadedModel(
            artifact_uri=row.artifact_uri,
            model=model,
            transform=_build_eval_transform(image_size, pipeline_type),
            classes=classes,
            classes_by_index={int(item["class_index"]): item for item in classes},
            image_size=image_size,
        )
        # Keep only current artifacts; model versions are small but Cloud Run memory is finite.
        _MODEL_CACHE.clear()
        _MODEL_CACHE[cache_key] = loaded
        return loaded


def _inspect_image(data: bytes) -> dict:
    if not data:
        raise ValueError("图片为空")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"单张图片不能超过 {MAX_IMAGE_BYTES // (1024 * 1024)} MB")
    try:
        with Image.open(io.BytesIO(data)) as source:
            source = ImageOps.exif_transpose(source)
            width, height = source.size
            if width <= 0 or height <= 0:
                raise ValueError("图片尺寸无效")
            if width * height > MAX_IMAGE_PIXELS:
                raise ValueError("图片像素过大，请压缩后再测试")
            image_format = (source.format or "JPEG").upper()
            return {"width": width, "height": height, "format": image_format}
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("无法识别图片格式，仅支持常见 JPEG / PNG / WEBP 图片") from exc


def _species_label(row: dict) -> str:
    return str(row.get("common_name_zh") or row.get("species_key") or row.get("class_index"))


def _serialize_box(box: BBox | None) -> dict | None:
    if box is None:
        return None
    normalized = box.normalized()
    return {
        "x1": round(normalized.x1, 6),
        "y1": round(normalized.y1, 6),
        "x2": round(normalized.x2, 6),
        "y2": round(normalized.y2, 6),
        "area_ratio": round(normalized.area_ratio, 6),
    }


def _serialize_detection(detection: Detection) -> dict:
    return {
        "confidence": round(float(detection.confidence), 6),
        "class_name": detection.class_name,
        "bbox": _serialize_box(detection.box),
    }


def _input_message(status: PipelineStatus) -> tuple[str, str]:
    messages = {
        PipelineStatus.NO_FISH: ("没有检测到鱼", "请重新拍摄或选择包含鱼的照片"),
        PipelineStatus.UNCERTAIN: ("鱼体检测结果不够确定", "请重新拍摄或选择更清晰、完整的单条鱼照片"),
        PipelineStatus.MULTIPLE_FISH: ("检测到多条鱼", "请重新拍摄单条鱼，或选择更清晰的照片"),
        PipelineStatus.INCOMPLETE_FISH: ("鱼体没有完整进入画面", "请尽量让鱼头、鱼尾和主要鳍部完整出现在照片中"),
        PipelineStatus.FISH_TOO_SMALL: ("鱼离镜头有点远", "靠近一点再拍，更容易准确识别鱼种"),
        PipelineStatus.READY: ("检测到完整单条鱼", "正在进行鱼种识别"),
    }
    return messages[status]


def _run_production_detector(image: Image.Image) -> DetectorRun:
    """Thin seam for deterministic production-pipeline tests; it always calls the verified runtime."""
    return detect(image)


def _classifier_prediction(model_row: ModelVersion, crop: Image.Image) -> dict:
    """Run MODEL_M1 only after the detector quality gate has declared the crop READY."""
    if not model_row.artifact_uri:
        raise ValueError("模型没有可用产物")
    loaded = _load_model(model_row)
    torch, _transforms = _torch_stack()
    tensor = loaded.transform(crop).unsqueeze(0)

    started = time.perf_counter()
    with torch.inference_mode():
        logits = loaded.model(tensor)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        probabilities = torch.softmax(logits, dim=1)[0]
        k = min(3, int(probabilities.shape[0]))
        values, indices = torch.topk(probabilities, k=k)
    latency_ms = round((time.perf_counter() - started) * 1000.0, 1)

    top3 = []
    for score, class_index in zip(values.tolist(), indices.tolist()):
        class_row = loaded.classes_by_index.get(int(class_index), {"class_index": int(class_index)})
        top3.append(
            {
                "class_index": int(class_index),
                "species": _species_label(class_row),
                "species_key": class_row.get("species_key"),
                "confidence": round(float(score), 6),
            }
        )
    if not top3:
        raise RuntimeError("模型没有返回预测结果")

    return {
        "model_status": model_row.status,
        "image_size": loaded.image_size,
        "top1": top3[0],
        "top3": top3,
        "low_confidence": top3[0]["confidence"] < LOW_CONFIDENCE_THRESHOLD,
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        "classifier_latency_ms": latency_ms,
    }


def _predict_bytes(db: Session, model_version: str, data: bytes) -> dict:
    """RECOGNITION_PIPELINE_v1: YOLOX → NMS → gate → expanded crop → classifier."""
    meta = _inspect_image(data)
    pipeline_started = time.perf_counter()
    try:
        with Image.open(io.BytesIO(data)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("无法读取图片") from exc

    detector_run = _run_production_detector(image)
    assessment = assess_detections(detector_run.detections)
    title, guidance = _input_message(assessment.status)
    detector_payload = {
        "model_version": detector_run.model_version,
        "onnx_sha256": detector_run.onnx_sha256,
        "input_size": detector_run.input_size,
        "input_scale": detector_run.input_scale,
        "input_draw_width": detector_run.input_draw_width,
        "input_draw_height": detector_run.input_draw_height,
        "latency_ms": detector_run.latency_ms,
        "detections": [_serialize_detection(item) for item in detector_run.detections],
    }
    gate_payload = {
        "status": assessment.status.name,
        "status_wire": assessment.status.value,
        "reason": assessment.reason,
        "primary_bbox": _serialize_box(assessment.primary.box if assessment.primary else None),
        "strong_detection_count": len(assessment.strong_detections),
        "weak_detection_count": len(assessment.weak_detections),
    }
    result = {
        "pipeline_version": str(load_contract()["contract_version"]),
        "model_version": model_version,
        "input": meta,
        "status": assessment.status.name,
        "status_wire": assessment.status.value,
        "ready": assessment.status is PipelineStatus.READY,
        "message": title,
        "guidance": guidance,
        "reason": assessment.reason,
        "detector": detector_payload,
        "quality_gate": gate_payload,
        "classification_ran": False,
        "low_confidence": False,
    }
    if assessment.status is not PipelineStatus.READY:
        result["latency_ms"] = round((time.perf_counter() - pipeline_started) * 1000.0, 1)
        return result

    assert assessment.crop_box is not None
    crop_pixels = crop_box_pixels(assessment.crop_box, image.width, image.height)
    crop = image.crop(crop_pixels)
    model_row = db.get(ModelVersion, model_version)
    if not model_row:
        raise ValueError("模型不存在")
    # New models are trained on the exact detector-expanded crop consumed by
    # the Android production pipeline. Explicit legacy models retain their
    # original-image input contract and are never silently relabelled.
    # A registry row created before the additive pipeline metadata migration is
    # a legacy whole-image model.  Keep that safe default so old models never
    # receive a crop input merely because the new column is unavailable.
    pipeline_type = validate_pipeline_type(getattr(model_row, "pipeline_type", WHOLE_IMAGE_V1))
    classifier_input = crop if pipeline_type == CROP_CLASSIFIER_V1 else image
    classifier = _classifier_prediction(model_row, classifier_input)
    result.update(classifier)
    result.update(
        {
            "pipeline_type": pipeline_type,
            "classifier_input": "crop" if pipeline_type == CROP_CLASSIFIER_V1 else "original",
            "classification_ran": True,
            "crop": {
                "bbox": _serialize_box(assessment.crop_box),
                "pixels": {
                    "left": crop_pixels[0],
                    "top": crop_pixels[1],
                    "right": crop_pixels[2],
                    "bottom": crop_pixels[3],
                    "width": crop.width,
                    "height": crop.height,
                },
            },
            "latency_ms": round((time.perf_counter() - pipeline_started) * 1000.0, 1),
        }
    )
    return result


def _safe_model_component(model_version: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_version)[:128] or "model"


def _extension(filename: str | None, content_type: str | None, image_format: str | None = None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    mapping = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "JPEG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
    }
    return mapping.get(content_type or "", mapping.get(image_format or "", ".jpg"))


def _persist_image(*, model_version: str, filename: str | None, content_type: str | None, data: bytes, image_format: str | None) -> str:
    bucket_name = get_bucket_name()
    day = _utcnow().strftime("%Y%m%d")
    suffix = _extension(filename, content_type, image_format)
    object_name = f"inference-tests/{_safe_model_component(model_version)}/{day}/{uuid4().hex}{suffix}"
    media_type = content_type if (content_type or "").startswith("image/") else (mimetypes.guess_type("x" + suffix)[0] or "image/jpeg")
    storage.Client().bucket(bucket_name).blob(object_name).upload_from_string(data, content_type=media_type)
    return f"gs://{bucket_name}/{object_name}"


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"{file.filename or '图片'} 超过单张 {MAX_IMAGE_BYTES // (1024 * 1024)} MB 限制")
    return data


def _result_with_storage(db: Session, *, model_version: str, file: UploadFile, data: bytes) -> dict:
    result = _predict_bytes(db, model_version, data)
    result["inference_id"] = f"INF_{uuid4().hex}"
    result["file_name"] = file.filename or "image"
    result["image_gcs_uri"] = _persist_image(
        model_version=model_version,
        filename=file.filename,
        content_type=file.content_type,
        data=data,
        image_format=(result.get("input") or {}).get("format"),
    )
    return result


@router.get("/inference", response_class=HTMLResponse)
def inference_page(request: Request):
    return templates.TemplateResponse(request=request, name="inference.html", context={})


@router.get("/api/inference/models")
def inference_models(db: Session = Depends(get_db)):
    stmt = (
        select(ModelVersion, TrainingRun)
        .join(TrainingRun, TrainingRun.run_id == ModelVersion.run_id)
        .where(ModelVersion.artifact_uri.is_not(None))
        .order_by(ModelVersion.created_at.desc())
    )
    rows = db.execute(stmt).all()
    return [
        {
            "model_version": model.model_version,
            "status": model.status,
            "run_id": model.run_id,
            "dataset_version": run.dataset_version,
            "created_at": model.created_at.isoformat() if model.created_at else None,
            "artifact_uri": model.artifact_uri,
            "pipeline_type": getattr(model, "pipeline_type", "WHOLE_IMAGE_V1"),
            "detector_version": getattr(model, "detector_version", None),
            "crop_version": getattr(model, "crop_version", None),
            "classifier_version": getattr(model, "classifier_version", None),
        }
        for model, run in rows
    ]


@router.post("/api/inference/predict")
async def inference_predict(
    model_version: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    try:
        data = await _read_upload(file)
        return _result_with_storage(db, model_version=model_version.strip(), file=file, data=data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"模型推理失败：{exc}") from exc


@router.post("/api/inference/batch")
async def inference_batch(
    model_version: str = Form(...),
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    if not files:
        raise HTTPException(status_code=400, detail="请选择图片")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(status_code=400, detail=f"单次 API 批量最多 {MAX_BATCH_FILES} 张；页面大批量模式会自动拆分上传")

    buffered: list[tuple[UploadFile, bytes]] = []
    total_bytes = 0
    try:
        for file in files:
            data = await _read_upload(file)
            total_bytes += len(data)
            if total_bytes > MAX_BATCH_BYTES:
                raise ValueError(f"单次 API 批量总大小不能超过 {MAX_BATCH_BYTES // (1024 * 1024)} MB")
            buffered.append((file, data))

        results = [
            _result_with_storage(db, model_version=model_version.strip(), file=file, data=data)
            for file, data in buffered
        ]
        latencies = [float(item.get("latency_ms") or 0) for item in results]
        return {
            "model_version": model_version.strip(),
            "count": len(results),
            "low_confidence_count": sum(1 for item in results if item.get("low_confidence")),
            "average_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
            "results": results,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"批量模型推理失败：{exc}") from exc

