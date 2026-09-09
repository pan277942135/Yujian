"""PowerPaint Direct Output Lab.

This is an isolated diagnostic path for testing the real PowerPaint Worker with
an original Dataset Freeze image and the raw SAM fish mask.  It intentionally
does not import or execute the Completion Decision, ROI, or Protected Compose
pipeline.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import secrets
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from google.cloud import storage
from PIL import Image
from sqlalchemy import select

from app.completion_worker_client import CompletionWorkerError, invoke_completion_worker
from app.dataset_models import DatasetItem
from app.db import get_db
from app.detector_runtime import detect, normalize_android_source
from app.fish_completion_lab import _read_dataset_image
from app.models import DatasetVersion
from app.recognition_pipeline import assess_detections
from app.segmentation.service import generate_fish_cutout

router = APIRouter(tags=["powerpaint-direct-lab"])
templates = Jinja2Templates(directory="app/templates")
DIRECT_VERSION = "POWERPAINT_DIRECT_V0.1"
DIRECT_PROMPT = (
    "Extract the fish from the image. Preserve the original fish appearance, "
    "color, texture, body shape and biological details. Complete missing parts "
    "naturally. Generate a complete realistic fish image. Do not change the "
    "fish species. Do not add background objects."
)
DIRECT_PREFIX = "experiments/powerpaint_direct_lab/v0.1"
MAX_BYTES = 25 * 1024 * 1024
logger = logging.getLogger(__name__)


def _bucket():
    name = os.getenv("GCS_BUCKET", "").strip()
    return storage.Client().bucket(name) if name else None


def _persist(test_id: str, name: str, content: bytes, content_type: str) -> str:
    bucket = _bucket()
    object_name = f"{DIRECT_PREFIX}/{test_id}/{name}"
    if bucket is None:
        path = os.path.join("var", "powerpaint_direct_lab", test_id, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(content)
        return f"local://{path}"
    blob = bucket.blob(object_name)
    blob.upload_from_string(content, content_type=content_type)
    return f"gs://{bucket.name}/{object_name}"


def _png(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _mask_png(mask: np.ndarray) -> bytes:
    return _png(Image.fromarray(np.where(mask, 255, 0).astype("uint8"), "L"))


def _data_url(data: bytes, media_type: str) -> str:
    return f"data:{media_type};base64,{base64.b64encode(data).decode()}"


def _decode_data_url(value: str | None) -> bytes | None:
    if not isinstance(value, str) or not value.startswith("data:") or "," not in value:
        return None
    return base64.b64decode(value.split(",", 1)[1])


def _read_uri(uri: str) -> bytes:
    if uri.startswith("gs://"):
        bucket_name, object_name = uri[5:].split("/", 1)
        return storage.Client().bucket(bucket_name).blob(object_name).download_as_bytes(timeout=120)
    if uri.startswith("local://"):
        with open(uri.removeprefix("local://"), "rb") as handle:
            return handle.read()
    raise ValueError(f"unsupported result URI: {uri}")


def _runtime(test_id: str) -> dict[str, Any]:
    return {
        "test_id": test_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": os.getenv("K_SERVICE", "unknown"),
        "revision": os.getenv("K_REVISION", "unknown"),
        "commit": os.getenv("APP_GIT_COMMIT", "unknown"),
    }


def _datasets(db):
    rows = db.scalars(
        select(DatasetVersion)
        .where(DatasetVersion.status == "FROZEN")
        .order_by(DatasetVersion.created_at.desc())
    ).all()
    return [
        {
            "dataset_version": row.dataset_version,
            "status": row.status,
            "pipeline_type": getattr(row, "pipeline_type", "WHOLE_IMAGE_V1"),
            "image_count": row.train_count + row.val_count + row.test_count,
            "train_count": row.train_count,
            "val_count": row.val_count,
            "test_count": row.test_count,
        }
        for row in rows
    ]


@router.get("/debug/powerpaint-direct-lab", response_class=HTMLResponse)
def direct_lab_page(request: Request):
    return templates.TemplateResponse(request=request, name="powerpaint_direct_lab.html", context={})


@router.get("/api/debug/powerpaint-direct-lab/datasets")
def direct_lab_datasets(db=Depends(get_db)):
    return _datasets(db)


@router.get("/api/debug/powerpaint-direct-lab/datasets/{dataset_version}/images")
def direct_lab_dataset_images(
    dataset_version: str,
    split: str | None = None,
    species: str | None = None,
    limit: int = Query(default=60, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db=Depends(get_db),
):
    dataset = db.get(DatasetVersion, dataset_version)
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集版本不存在")
    if dataset.status != "FROZEN":
        raise HTTPException(status_code=409, detail="只能从已冻结 Dataset Freeze 版本选择图片")
    stmt = select(DatasetItem).where(DatasetItem.dataset_version == dataset_version)
    if split:
        stmt = stmt.where(DatasetItem.split == split)
    if species:
        stmt = stmt.where(DatasetItem.species_name == species)
    rows = db.scalars(stmt.order_by(DatasetItem.id).offset(offset).limit(limit)).all()
    return [
        {
            "dataset_version": row.dataset_version,
            "dataset_item_id": row.id,
            "image_id": row.image_id,
            "species": row.species_name,
            "species_key": row.species_key,
            "split": row.split,
            "preview_url": f"/media/{row.batch_id}/{row.image_id}",
        }
        for row in rows
    ]


def _progress(report: dict[str, Any]) -> list[dict[str, Any]]:
    timings = report.get("timings", {})
    worker = report.get("worker", {})
    return [
        {"stage": "input", "label": "Original", "status": "READY", "elapsed_ms": timings.get("input_decode_ms"), "result": report.get("input", {}).get("filename")},
        {"stage": "detector", "label": "Detector / BBox", "status": "READY", "elapsed_ms": timings.get("detector_ms"), "result": report.get("detector")},
        {"stage": "sam", "label": "SAM Mask", "status": "READY", "elapsed_ms": timings.get("sam_ms"), "result": report.get("sam")},
        {"stage": "powerpaint", "label": "PowerPaint Direct", "status": worker.get("status", "PENDING"), "elapsed_ms": timings.get("worker_ms"), "result": worker.get("result_uri") or worker.get("error")},
    ]


@router.post("/api/debug/powerpaint-direct-lab/run")
def run_direct_lab(dataset_version: str, dataset_item_id: int, db=Depends(get_db)):
    started = time.perf_counter()
    test_id = "PPD_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(2)
    item = db.scalar(
        select(DatasetItem).where(
            DatasetItem.dataset_version == dataset_version,
            DatasetItem.id == dataset_item_id,
        )
    )
    if not item:
        raise HTTPException(status_code=404, detail="DATASET_ITEM_NOT_FOUND")
    try:
        data = _read_dataset_image(item)
        if not data or len(data) > MAX_BYTES:
            raise HTTPException(status_code=400, detail="INVALID_DATASET_IMAGE")
        decode_started = time.perf_counter()
        with Image.open(io.BytesIO(data)) as uploaded:
            source = normalize_android_source(uploaded)
        input_decode_ms = round((time.perf_counter() - decode_started) * 1000, 2)

        detector_started = time.perf_counter()
        detector_run = detect(source)
        assessment = assess_detections(detector_run.detections)
        detector_ms = round((time.perf_counter() - detector_started) * 1000, 2)
        primary = assessment.primary
        if primary is None:
            raise HTTPException(status_code=422, detail="NO_RELIABLE_PRIMARY_FISH")

        sam_started = time.perf_counter()
        segmentation = generate_fish_cutout(source, primary.box)
        sam_ms = round((time.perf_counter() - sam_started) * 1000, 2)
        raw_mask = segmentation.mask.astype(bool)
        normalized = primary.box.normalized()
        bbox_pixels = [
            round(normalized.x1 * source.width),
            round(normalized.y1 * source.height),
            round(normalized.x2 * source.width),
            round(normalized.y2 * source.height),
        ]
        detector = {
            "model": detector_run.model_version,
            "assessment": assessment.status.value,
            "confidence": round(float(primary.confidence), 6),
            "bbox_pixels": bbox_pixels,
        }
        original_bytes = _png(source.convert("RGB"))
        sam_mask_bytes = _mask_png(raw_mask)
        original_uri = _persist(test_id, "original.png", original_bytes, "image/png")
        sam_mask_uri = _persist(test_id, "sam_mask.png", sam_mask_bytes, "image/png")
        sam_transparent_uri = _persist(test_id, "sam_transparent.png", segmentation.cutout_png, "image/png")
        _persist(test_id, "detector.json", json.dumps(detector, ensure_ascii=False).encode(), "application/json")

        report: dict[str, Any] = {
            "experiment": DIRECT_VERSION,
            "prompt_id": DIRECT_VERSION,
            "prompt": DIRECT_PROMPT,
            "runtime": _runtime(test_id),
            "input": {
                "filename": item.image_id,
                "image_id": hashlib.sha256(data).hexdigest()[:16],
                "dataset_version": dataset_version,
                "dataset_item_id": item.id,
                "source_type": "dataset_freeze",
                "width": source.width,
                "height": source.height,
                "size_bytes": len(data),
            },
            "detector": detector,
            "sam": {
                "used": True,
                "model": "SAM_VIT_B",
                "quality": segmentation.quality.value,
                "mask_area_pixels": int(raw_mask.sum()),
            },
            "preview_original": _data_url(original_bytes, "image/png"),
            "preview_sam": _data_url(segmentation.cutout_png, "image/png"),
            "worker": {"status": "PENDING", "result_uri": None, "worker_ms": None},
            "assets": {"original": original_uri, "sam_mask": sam_mask_uri, "sam_transparent": sam_transparent_uri},
            "timings": {"input_decode_ms": input_decode_ms, "detector_ms": detector_ms, "sam_ms": sam_ms, "worker_ms": None, "total_ms": None},
        }

        worker_started = time.perf_counter()
        try:
            worker_result = invoke_completion_worker(
                image_uri=original_uri,
                mask_uri=sam_mask_uri,
                prompt=DIRECT_PROMPT,
            )
            worker_ms = round((time.perf_counter() - worker_started) * 1000, 2)
            result_uri = worker_result.get("result_uri")
            generated_uri = None
            generated_preview = worker_result.get("generated_roi")
            generated_bytes = _decode_data_url(result_uri) or _decode_data_url(generated_preview)
            if generated_bytes is not None:
                generated_uri = _persist(test_id, "powerpaint_result.png", generated_bytes, "image/png")
                generated_preview = _data_url(generated_bytes, "image/png")
            elif result_uri:
                generated_bytes = _read_uri(result_uri)
                generated_uri = _persist(test_id, "powerpaint_result.png", generated_bytes, "image/png")
                generated_preview = _data_url(generated_bytes, "image/png")
            report["worker"] = {
                "status": "WORKER_EXECUTED",
                "result_uri": result_uri,
                "model_version": worker_result.get("model_version"),
                "inference_time_ms": worker_result.get("inference_time_ms"),
                "worker_ms": worker_ms,
            }
            report["assets"]["powerpaint_result"] = generated_uri or result_uri
            report["result_preview"] = generated_preview
            report["timings"]["worker_ms"] = worker_ms
        except CompletionWorkerError as exc:
            worker_ms = round((time.perf_counter() - worker_started) * 1000, 2)
            report["worker"] = {"status": "WORKER_FAILED", "error_code": exc.error_code, "error": str(exc), "status_code": exc.status_code, "worker_ms": worker_ms}
            report["timings"]["worker_ms"] = worker_ms
            report["timings"]["total_ms"] = round((time.perf_counter() - started) * 1000, 2)
            report["progress"] = _progress(report)
            report_uri = _persist(test_id, "report.json", json.dumps(report, ensure_ascii=False, indent=2).encode(), "application/json")
            report["assets"]["report"] = report_uri
            return JSONResponse(status_code=502, content={"status": "error", "stage": "powerpaint", "error": report["worker"], "report": report})
        finally:
            source.close()

        report["timings"]["total_ms"] = round((time.perf_counter() - started) * 1000, 2)
        report["progress"] = _progress(report)
        report_uri = _persist(test_id, "report.json", json.dumps(report, ensure_ascii=False, indent=2).encode(), "application/json")
        report["assets"]["report"] = report_uri
        return {"status": "ok", "test_id": test_id, "report": report}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("PowerPaint Direct Lab failed; test_id=%s", test_id)
        return JSONResponse(status_code=500, content={"status": "error", "stage": "pipeline", "error": {"error_code": "POWERPAINT_DIRECT_FAILED", "message": f"{exc.__class__.__name__}: {exc}"}})
