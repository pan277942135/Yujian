"""PowerPaint Direct Lab V0.2.

This is an intentionally isolated diagnostic path:

    Dataset Freeze -> Detector -> SAM -> Worker Edit Mask -> PowerPaint

It does not import or execute the fish-completion decision, ROI, completion
mask, AUTO_COMPLETION, MANUAL_DEBUG, or Protected Compose pipelines.
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
import urllib.request
from datetime import datetime, timezone
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from google.cloud import storage
from PIL import Image
from sqlalchemy import select

from app.completion_worker_client import (
    CompletionWorkerError,
    check_completion_worker,
    invoke_completion_worker,
)
from app.dataset_models import DatasetItem
from app.db import get_db
from app.detector_runtime import detect, normalize_android_source
from app.models import DatasetVersion
from app.recognition_pipeline import assess_detections
from app.segmentation.service import generate_fish_cutout

router = APIRouter(tags=["powerpaint-direct-lab"])
templates = Jinja2Templates(directory="app/templates")

DIRECT_VERSION = "POWERPAINT_DIRECT_V0.2"
PROMPT_ID = DIRECT_VERSION
DIRECT_PROMPT = """Reconstruct one complete realistic fish from the provided image.

Use the visible fish as the primary identity reference.

Preserve as much as possible:
- fish species characteristics
- head shape
- body proportions
- body color
- scale pattern
- fin shape
- tail structure
- visible texture
- distinctive biological features

Naturally reconstruct fish parts that are missing or occluded.

Remove or replace non-fish objects only inside the editable region when they interfere with the fish.

Do not change the fish species.
Do not create a second fish.
Do not introduce unrelated objects.
Keep the completed anatomy coherent and biologically plausible."""

MASK_STRATEGIES = {
    "SAM_PROTECT_VISIBLE": "可见鱼体保护 · 周边允许生成",
    "SAM_REPAINT_VISIBLE": "实验重绘模式 · 可见鱼体允许重绘",
}
DIRECT_PREFIX = "experiments/powerpaint_direct_lab/v0.2"
MAX_BYTES = 25 * 1024 * 1024
logger = logging.getLogger(__name__)


def _bucket():
    name = os.getenv("GCS_BUCKET", "").strip()
    return storage.Client().bucket(name) if name else None


def _local_path(test_id: str, name: str) -> str:
    return os.path.join("var", "powerpaint_direct_lab", test_id, name)


def _persist(test_id: str, name: str, content: bytes, content_type: str) -> str:
    bucket = _bucket()
    object_name = f"{DIRECT_PREFIX}/{test_id}/{name}"
    if bucket is None:
        path = _local_path(test_id, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(content)
        return f"local://{path}"
    blob = bucket.blob(object_name)
    blob.upload_from_string(content, content_type=content_type)
    return f"gs://{bucket.name}/{object_name}"


def _state_uri(test_id: str) -> str:
    bucket = _bucket()
    if bucket is None:
        return f"local://{_local_path(test_id, 'prepare.json')}"
    return f"gs://{bucket.name}/{DIRECT_PREFIX}/{test_id}/prepare.json"


def _read_uri(uri: str) -> bytes:
    if uri.startswith("data:"):
        return _decode_data_url(uri) or b""
    if uri.startswith("gs://"):
        bucket_name, object_name = uri[5:].split("/", 1)
        return storage.Client().bucket(bucket_name).blob(object_name).download_as_bytes(timeout=120)
    if uri.startswith("local://"):
        with open(uri.removeprefix("local://"), "rb") as handle:
            return handle.read()
    if uri.startswith(("http://", "https://")):
        with urllib.request.urlopen(uri, timeout=120) as response:
            return response.read()
    raise ValueError(f"unsupported result URI: {uri}")


def _read_json(uri: str) -> dict[str, Any]:
    payload = json.loads(_read_uri(uri).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("persisted prepare state is not an object")
    return payload


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
    try:
        return base64.b64decode(value.split(",", 1)[1])
    except (ValueError, TypeError):
        return None


def _runtime(test_id: str) -> dict[str, Any]:
    return {
        "test_id": test_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": os.getenv("K_SERVICE", "unknown"),
        "revision": os.getenv("K_REVISION", "unknown"),
        "commit": os.getenv("APP_GIT_COMMIT", "unknown"),
    }


def _new_test_id() -> str:
    return "PPD_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(2)


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


def _read_dataset_image(item: DatasetItem) -> bytes:
    uri = (item.gcs_uri or "").strip()
    if not uri.startswith("gs://") or "/" not in uri[5:]:
        raise HTTPException(status_code=422, detail="DATASET_IMAGE_URI_INVALID")
    bucket_name, object_name = uri[5:].split("/", 1)
    try:
        return storage.Client().bucket(bucket_name).blob(object_name).download_as_bytes(timeout=120)
    except Exception as exc:
        logger.exception("Direct Lab dataset image read failed item=%s", item.id)
        raise HTTPException(status_code=502, detail="DATASET_IMAGE_READ_FAILED") from exc


def _bbox_mask(box, width: int, height: int, *, expand_ratio: float = 0.0) -> np.ndarray:
    expanded = box.expand(expand_ratio).normalized()
    left = max(0, min(width - 1, int(np.floor(expanded.x1 * width))))
    top = max(0, min(height - 1, int(np.floor(expanded.y1 * height))))
    right = max(left + 1, min(width, int(np.ceil(expanded.x2 * width))))
    bottom = max(top + 1, min(height, int(np.ceil(expanded.y2 * height))))
    result = np.zeros((height, width), dtype=bool)
    result[top:bottom, left:right] = True
    return result


def build_worker_edit_mask(raw_sam_mask: np.ndarray, bbox, strategy: str) -> np.ndarray:
    """Build a full-frame Worker edit mask without making a semantic guess."""
    if strategy not in MASK_STRATEGIES:
        raise ValueError(f"unsupported mask strategy: {strategy}")
    visible = np.asarray(raw_sam_mask, dtype=bool)
    if visible.ndim != 2:
        raise ValueError("SAM mask must be a 2D array")
    if strategy == "SAM_REPAINT_VISIBLE":
        return visible.copy()
    envelope = _bbox_mask(bbox, visible.shape[1], visible.shape[0], expand_ratio=0.30)
    # White means editable. The raw visible fish remains protected/black.
    return envelope & ~visible


def _detector_summary(detector_run, assessment, primary, source: Image.Image) -> dict[str, Any]:
    normalized = primary.box.normalized()
    bbox_pixels = [
        round(normalized.x1 * source.width),
        round(normalized.y1 * source.height),
        round(normalized.x2 * source.width),
        round(normalized.y2 * source.height),
    ]
    return {
        "model": detector_run.model_version,
        "assessment": assessment.status.value,
        "confidence": round(float(primary.confidence), 6),
        "bbox_pixels": bbox_pixels,
        "bbox_normalized": [normalized.x1, normalized.y1, normalized.x2, normalized.y2],
    }


def _base_report(test_id: str, item: DatasetItem, data: bytes, source: Image.Image, detector: dict[str, Any], sam: dict[str, Any], strategy: str) -> dict[str, Any]:
    return {
        "experiment": DIRECT_VERSION,
        "prompt_id": PROMPT_ID,
        "prompt": DIRECT_PROMPT,
        "runtime": _runtime(test_id),
        "input": {
            "filename": item.image_id,
            "image_id": hashlib.sha256(data).hexdigest()[:16],
            "dataset_version": item.dataset_version,
            "dataset_item_id": item.id,
            "source_type": "dataset_freeze",
            "width": source.width,
            "height": source.height,
            "size_bytes": len(data),
        },
        "detector": detector,
        "sam": sam,
        "direct_generation": {
            "mask_strategy": strategy,
            "edit_mask_pixels": 0,
            "edit_mask_ratio": 0.0,
            "prompt_id": PROMPT_ID,
        },
        "worker": {
            "health": None,
            "status": "PENDING",
            "worker_ms": None,
            "inference_time_ms": None,
            "result_uri": None,
        },
        "timings": {
            "detector_ms": None,
            "sam_ms": None,
            "prepare_total_ms": None,
            "worker_ms": None,
            "total_ms": None,
        },
    }


def _progress(report: dict[str, Any]) -> list[dict[str, Any]]:
    timings = report.get("timings", {})
    worker = report.get("worker", {})
    direct = report.get("direct_generation", {})
    return [
        {"stage": "input", "label": "Input", "status": "READY", "elapsed_ms": timings.get("input_decode_ms"), "result": report.get("input", {}).get("filename")},
        {"stage": "detector", "label": "Detector", "status": "READY", "elapsed_ms": timings.get("detector_ms"), "result": report.get("detector")},
        {"stage": "sam", "label": "SAM", "status": "READY", "elapsed_ms": timings.get("sam_ms"), "result": report.get("sam")},
        {"stage": "worker_edit_mask", "label": "Worker Edit Mask", "status": "READY", "elapsed_ms": None, "result": direct.get("mask_strategy")},
        {"stage": "powerpaint", "label": "PowerPaint Direct", "status": worker.get("status", "PENDING"), "elapsed_ms": timings.get("worker_ms"), "result": worker.get("result_uri") or worker.get("error")},
    ]


def _error_response(status_code: int, *, stage: str, test_id: str, error_code: str, message: str, report: dict[str, Any] | None = None, http_status: int | None = None) -> JSONResponse:
    error = {"error_code": error_code, "message": message}
    if http_status is not None:
        error["http_status"] = http_status
    body: dict[str, Any] = {"status": "error", "stage": stage, "test_id": test_id, "error": error}
    if report is not None:
        body["report"] = report
    return JSONResponse(status_code=status_code, content=body)


async def _request_payload(request: Request) -> dict[str, Any]:
    content_type = (request.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        payload = await request.json()
    else:
        form = await request.form()
        payload = dict(form)
    if not isinstance(payload, dict):
        raise ValueError("request payload must be an object")
    return payload


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


@router.post("/api/debug/powerpaint-direct-lab/prepare")
async def prepare_direct_lab(request: Request, db=Depends(get_db)):
    started = time.perf_counter()
    test_id = _new_test_id()
    source = None
    try:
        payload = await _request_payload(request)
        dataset_version = str(payload.get("dataset_version") or "").strip()
        dataset_item_id = int(payload.get("dataset_item_id"))
        strategy = str(payload.get("mask_strategy") or "SAM_PROTECT_VISIBLE").strip().upper()
        if strategy not in MASK_STRATEGIES:
            return _error_response(400, stage="prepare", test_id=test_id, error_code="INVALID_MASK_STRATEGY", message=strategy)
        item = db.scalar(select(DatasetItem).where(DatasetItem.dataset_version == dataset_version, DatasetItem.id == dataset_item_id))
        dataset = db.get(DatasetVersion, dataset_version)
        if not dataset or dataset.status != "FROZEN":
            return _error_response(409, stage="prepare", test_id=test_id, error_code="DATASET_VERSION_NOT_FROZEN", message=dataset_version)
        if not item:
            return _error_response(404, stage="prepare", test_id=test_id, error_code="DATASET_ITEM_NOT_FOUND", message=str(dataset_item_id))
        data = _read_dataset_image(item)
        if not data or len(data) > MAX_BYTES:
            return _error_response(400, stage="prepare", test_id=test_id, error_code="INVALID_DATASET_IMAGE", message="image is empty or larger than 25 MiB")

        decode_started = time.perf_counter()
        with Image.open(io.BytesIO(data)) as uploaded:
            source = normalize_android_source(uploaded)
        input_decode_ms = round((time.perf_counter() - decode_started) * 1000, 2)

        detector_started = time.perf_counter()
        detector_run = detect(source)
        assessment = assess_detections(detector_run.detections)
        primary = assessment.primary
        detector_ms = round((time.perf_counter() - detector_started) * 1000, 2)
        if primary is None:
            return _error_response(422, stage="detector", test_id=test_id, error_code="NO_RELIABLE_PRIMARY_FISH", message=assessment.status.value)

        sam_started = time.perf_counter()
        segmentation = generate_fish_cutout(source, primary.box)
        sam_ms = round((time.perf_counter() - sam_started) * 1000, 2)
        raw_mask = np.asarray(segmentation.mask, dtype=bool)
        worker_edit_mask = build_worker_edit_mask(raw_mask, primary.box, strategy)
        if int(worker_edit_mask.sum()) <= 0:
            return _error_response(422, stage="worker_edit_mask", test_id=test_id, error_code="WORKER_EDIT_MASK_EMPTY", message="selected strategy produced an empty edit region")

        original_bytes = _png(source.convert("RGB"))
        sam_mask_bytes = _mask_png(raw_mask)
        worker_mask_bytes = _mask_png(worker_edit_mask)
        original_uri = _persist(test_id, "original.png", original_bytes, "image/png")
        sam_mask_uri = _persist(test_id, "sam_mask.png", sam_mask_bytes, "image/png")
        sam_transparent_uri = _persist(test_id, "sam_transparent.png", segmentation.cutout_png, "image/png")
        worker_edit_mask_uri = _persist(test_id, "worker_edit_mask.png", worker_mask_bytes, "image/png")
        detector = _detector_summary(detector_run, assessment, primary, source)
        sam = {"used": True, "model": "SAM_VIT_B", "quality": segmentation.quality.value, "mask_area_pixels": int(raw_mask.sum())}
        report = _base_report(test_id, item, data, source, detector, sam, strategy)
        report["assets"] = {"original": original_uri, "sam_mask": sam_mask_uri, "sam_transparent": sam_transparent_uri, "worker_edit_mask": worker_edit_mask_uri}
        report["preview_original"] = _data_url(original_bytes, "image/png")
        report["preview_sam"] = _data_url(segmentation.cutout_png, "image/png")
        report["preview_worker_mask"] = _data_url(worker_mask_bytes, "image/png")
        report["direct_generation"]["edit_mask_pixels"] = int(worker_edit_mask.sum())
        report["direct_generation"]["edit_mask_ratio"] = round(float(worker_edit_mask.mean()), 6)
        report["timings"].update({"input_decode_ms": input_decode_ms, "detector_ms": detector_ms, "sam_ms": sam_ms})
        report["timings"]["prepare_total_ms"] = round((time.perf_counter() - started) * 1000, 2)
        report["progress"] = _progress(report)
        state = {"test_id": test_id, "dataset_version": dataset_version, "dataset_item_id": item.id, "image_id": item.image_id, "original_uri": original_uri, "sam_mask_uri": sam_mask_uri, "sam_transparent_uri": sam_transparent_uri, "worker_edit_mask_uri": worker_edit_mask_uri, "mask_strategy": strategy, "prompt_id": PROMPT_ID, "report": report}
        state_uri = _persist(test_id, "prepare.json", json.dumps(state, ensure_ascii=False, indent=2).encode(), "application/json")
        report["assets"]["prepare"] = state_uri
        report_uri = _persist(test_id, "prepare_report.json", json.dumps(report, ensure_ascii=False, indent=2).encode(), "application/json")
        report["assets"]["prepare_report"] = report_uri
        logger.info("direct_lab_prepare test_id=%s status=prepared strategy=%s elapsed_ms=%s", test_id, strategy, report["timings"]["prepare_total_ms"])
        return {"status": "ok", "stage": "prepared", "test_id": test_id, "report": report}
    except (TypeError, ValueError) as exc:
        logger.exception("direct_lab_prepare invalid request test_id=%s", test_id)
        return _error_response(400, stage="prepare", test_id=test_id, error_code="INVALID_PREPARE_REQUEST", message=str(exc))
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail, ensure_ascii=False)
        return _error_response(exc.status_code, stage="prepare", test_id=test_id, error_code=str(exc.detail), message=detail, http_status=exc.status_code)
    except Exception as exc:
        logger.exception("direct_lab_prepare failed test_id=%s", test_id)
        return _error_response(500, stage="prepare", test_id=test_id, error_code="DIRECT_PREPARE_FAILED", message=f"{exc.__class__.__name__}: {exc}")
    finally:
        if source is not None:
            source.close()


@router.post("/api/debug/powerpaint-direct-lab/run")
async def run_direct_lab(request: Request):
    started = time.perf_counter()
    test_id = "unknown"
    try:
        payload = await _request_payload(request)
        test_id = str(payload.get("test_id") or "").strip()
        if not test_id or len(test_id) > 80:
            return _error_response(400, stage="run", test_id=test_id or "unknown", error_code="TEST_ID_REQUIRED", message="test_id is required")
        state = _read_json(_state_uri(test_id))
        report = dict(state.get("report") or {})
        report["runtime"] = _runtime(test_id)
        try:
            health = check_completion_worker()
        except CompletionWorkerError as exc:
            worker_ms = round((time.perf_counter() - started) * 1000, 2)
            report["worker"] = {
                "health": {
                    "endpoint_configured": bool(os.getenv("FISH_COMPLETION_WORKER_URL", "").strip()),
                    "status": "UNREACHABLE",
                    "error_code": exc.error_code,
                    "message": str(exc),
                    "http_status": exc.status_code,
                },
                "status": "WORKER_FAILED",
                "error_code": exc.error_code,
                "error": str(exc),
                "status_code": exc.status_code,
                "worker_ms": worker_ms,
            }
            report["timings"]["worker_ms"] = worker_ms
            report["timings"]["total_ms"] = worker_ms
            report["progress"] = _progress(report)
            report_uri = _persist(test_id, "report.json", json.dumps(report, ensure_ascii=False, indent=2).encode(), "application/json")
            report["assets"]["report"] = report_uri
            logger.exception("direct_lab_worker_health_failed test_id=%s error_code=%s", test_id, exc.error_code)
            return _error_response(503, stage="powerpaint", test_id=test_id, error_code=exc.error_code, message=str(exc), report=report, http_status=exc.status_code)
        report["worker"]["health"] = health
        if health.get("status") != "READY":
            return _error_response(503, stage="powerpaint", test_id=test_id, error_code="COMPLETION_WORKER_NOT_READY", message=json.dumps(health, ensure_ascii=False), report=report)
        worker_started = time.perf_counter()
        try:
            worker_result = invoke_completion_worker(image_uri=state["original_uri"], mask_uri=state["worker_edit_mask_uri"], prompt=DIRECT_PROMPT, task="fish_completion")
        except CompletionWorkerError as exc:
            worker_ms = round((time.perf_counter() - worker_started) * 1000, 2)
            report["worker"] = {"health": health, "status": "WORKER_FAILED", "error_code": exc.error_code, "error": str(exc), "status_code": exc.status_code, "worker_ms": worker_ms}
            report["timings"]["worker_ms"] = worker_ms
            report["timings"]["total_ms"] = round((time.perf_counter() - started) * 1000, 2)
            report["progress"] = _progress(report)
            report_uri = _persist(test_id, "report.json", json.dumps(report, ensure_ascii=False, indent=2).encode(), "application/json")
            report["assets"]["report"] = report_uri
            logger.exception("direct_lab_worker_failed test_id=%s error_code=%s", test_id, exc.error_code)
            return _error_response(502, stage="powerpaint", test_id=test_id, error_code=exc.error_code, message=str(exc), report=report, http_status=exc.status_code)

        worker_ms = round((time.perf_counter() - worker_started) * 1000, 2)
        result_uri = worker_result.get("result_uri")
        generated_bytes = _decode_data_url(result_uri) or _decode_data_url(worker_result.get("generated_roi"))
        if generated_bytes is None and result_uri:
            generated_bytes = _read_uri(result_uri)
        if not generated_bytes:
            raise ValueError("worker returned no readable generated image")
        generated_uri = _persist(test_id, "powerpaint_result.png", generated_bytes, "image/png")
        report["worker"] = {"health": health, "status": "WORKER_EXECUTED", "worker_ms": worker_ms, "inference_time_ms": worker_result.get("inference_time_ms"), "model_version": worker_result.get("model_version"), "result_uri": result_uri}
        report["assets"]["powerpaint_result"] = generated_uri
        report["result_preview"] = _data_url(generated_bytes, "image/png")
        report["timings"]["worker_ms"] = worker_ms
        report["timings"]["total_ms"] = round((time.perf_counter() - started) * 1000, 2)
        report["progress"] = _progress(report)
        report_uri = _persist(test_id, "report.json", json.dumps(report, ensure_ascii=False, indent=2).encode(), "application/json")
        report["assets"]["report"] = report_uri
        logger.info("direct_lab_run test_id=%s status=WORKER_EXECUTED worker_ms=%s", test_id, worker_ms)
        return {"status": "ok", "stage": "powerpaint_complete", "test_id": test_id, "report": report}
    except FileNotFoundError as exc:
        logger.exception("direct_lab_run state not found test_id=%s", test_id)
        return _error_response(404, stage="run", test_id=test_id, error_code="PREPARE_STATE_NOT_FOUND", message=str(exc))
    except CompletionWorkerError as exc:
        logger.exception("direct_lab_worker_health_failed test_id=%s error_code=%s", test_id, exc.error_code)
        return _error_response(503, stage="powerpaint", test_id=test_id, error_code=exc.error_code, message=str(exc), http_status=exc.status_code)
    except Exception as exc:
        logger.exception("direct_lab_run failed test_id=%s", test_id)
        return _error_response(500, stage="run", test_id=test_id, error_code="DIRECT_RUN_FAILED", message=f"{exc.__class__.__name__}: {exc}")
