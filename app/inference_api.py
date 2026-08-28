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
from app.factory import get_bucket_name
from app.models import ModelVersion, TrainingRun

router = APIRouter(tags=["model-inference"])
templates = Jinja2Templates(directory="app/templates")

MAX_IMAGE_BYTES = int(os.getenv("INFERENCE_MAX_IMAGE_BYTES", str(15 * 1024 * 1024)))
MAX_BATCH_FILES = int(os.getenv("INFERENCE_MAX_BATCH_FILES", "20"))
MAX_BATCH_BYTES = int(os.getenv("INFERENCE_MAX_BATCH_BYTES", str(24 * 1024 * 1024)))
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("INFERENCE_LOW_CONFIDENCE_THRESHOLD", "0.55"))
MAX_IMAGE_PIXELS = int(os.getenv("INFERENCE_MAX_IMAGE_PIXELS", "40000000"))

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


def _build_eval_transform(image_size: int):
    _torch, transforms = _torch_stack()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
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
        model_bytes = _download_bytes(client, row.artifact_uri)
        model = torch.jit.load(io.BytesIO(model_bytes), map_location="cpu")
        model.eval()
        loaded = LoadedModel(
            artifact_uri=row.artifact_uri,
            model=model,
            transform=_build_eval_transform(image_size),
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


def _predict_bytes(db: Session, model_version: str, data: bytes) -> dict:
    model_row = db.get(ModelVersion, model_version)
    if not model_row:
        raise ValueError("模型不存在")
    if not model_row.artifact_uri:
        raise ValueError("模型没有可用产物")

    meta = _inspect_image(data)
    loaded = _load_model(model_row)
    torch, _transforms = _torch_stack()
    try:
        with Image.open(io.BytesIO(data)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            tensor = loaded.transform(image).unsqueeze(0)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("无法读取图片") from exc

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
        "model_version": model_version,
        "model_status": model_row.status,
        "image_size": loaded.image_size,
        "input": meta,
        "top1": top3[0],
        "top3": top3,
        "low_confidence": top3[0]["confidence"] < LOW_CONFIDENCE_THRESHOLD,
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        "latency_ms": latency_ms,
    }


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
