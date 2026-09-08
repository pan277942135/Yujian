"""Isolated Fish Completion Lab v0.2 automatic pipeline."""
from __future__ import annotations

import io
import logging
import os
import secrets
from datetime import datetime, timezone
from math import ceil, floor
from typing import Any

import numpy as np
from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageFilter

from app.completion_worker_client import CompletionWorkerError, check_completion_worker, invoke_completion_worker
from app.detector_runtime import detect, normalize_android_source
from app.fish_completion_lab import MAX_BYTES, _data_url, _mask_bytes, _persist, _png, _read_persist, _runtime, _save_state
from app.recognition_pipeline import assess_detections
from app.segmentation.service import generate_fish_cutout

router = APIRouter(tags=["fish-completion-lab-v02"])
templates = Jinja2Templates(directory="app/templates")
V02_VERSION = "YUJIAN_FISH_COMPLETION_LAB_V0.2"
logger = logging.getLogger(__name__)


def _error(status: int, code: str, message: str, stage: str, detail: Any = None):
    return JSONResponse(status_code=status, content={"status": "error", "stage": stage, "error": {"error_code": code, "message": message, "detail": detail or {}}})


def _bbox_to_pixels(box: Any, width: int, height: int) -> tuple[int, int, int, int]:
    """Convert a normalized detector box to a clamped pixel box."""
    normalized = box.normalized()
    x1 = max(0, min(width, floor(float(normalized.x1) * width)))
    y1 = max(0, min(height, floor(float(normalized.y1) * height)))
    x2 = max(0, min(width, ceil(float(normalized.x2) * width)))
    y2 = max(0, min(height, ceil(float(normalized.y2) * height)))
    return x1, y1, x2, y2


def _component_count(mask: np.ndarray) -> int:
    seen = np.zeros(mask.shape, dtype=bool)
    count = 0
    for y, x in zip(*np.where(mask)):
        if seen[y, x]:
            continue
        count += 1
        stack = [(int(y), int(x))]
        seen[y, x] = True
        while stack:
            cy, cx = stack.pop()
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
    return count


def analyze_completion(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[dict[str, Any], np.ndarray]:
    h, w = mask.shape
    x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(w, bbox[2]), min(h, bbox[3])
    box_mask = np.zeros_like(mask)
    if x2 <= x1 or y2 <= y1:
        return {"completion_required": False, "severity": "NOT_ELIGIBLE", "completion_ratio": 1.0, "reason": ["invalid_bbox"], "region_count": 0}, box_mask
    box_mask[y1:y2, x1:x2] = True
    closed = Image.fromarray(np.where(mask & box_mask, 255, 0).astype("uint8"), "L").filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.MinFilter(5))
    candidate = (np.asarray(closed) > 127) & ~mask & box_mask
    missing = int(candidate.sum())
    ratio = missing / max(1, int(mask.sum()) + missing)
    regions = _component_count(candidate)
    severity, required, reasons = "COMPLETION_NOT_REQUIRED", False, []
    if missing:
        reasons.append("contour_gap_detected")
        if ratio <= .05 and regions <= 2: severity, required = "LIGHT", True
        elif ratio <= .10 and regions <= 2: severity, required = "MEDIUM", True
        elif ratio <= .15 and regions <= 2: severity, required = "HEAVY", True
        elif ratio <= .20 and regions <= 2: severity, required = "EXTREME_EXPERIMENTAL", True
        else: severity, reasons = "NOT_ELIGIBLE", reasons + ["completion_safety_limit"]
    return {"completion_required": required, "severity": severity, "completion_ratio": round(ratio, 6), "reason": reasons or ["visible_mask_sufficient"], "region_count": regions}, candidate


def _build_completion_roi(original: Image.Image, completion_mask: np.ndarray, test_id: str) -> dict[str, Any]:
    ys, xs = np.where(completion_mask)
    if not len(xs):
        raise ValueError("COMPLETION_MASK_EMPTY")
    h, w = completion_mask.shape
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    pad_x, pad_y = max(8, int((x2 - x1) * .20)), max(8, int((y2 - y1) * .20))
    x1, y1, x2, y2 = max(0, x1 - pad_x), max(0, y1 - pad_y), min(w, x2 + pad_x), min(h, y2 + pad_y)
    roi = original.crop((x1, y1, x2, y2))
    roi_mask = Image.fromarray(np.where(completion_mask[y1:y2, x1:x2], 255, 0).astype("uint8"), "L")
    original_size = roi.size
    longest = max(original_size)
    if longest > 512:
        scale = 512 / longest
        worker_size = (max(8, int(roi.width * scale)), max(8, int(roi.height * scale)))
        roi = roi.resize(worker_size, Image.Resampling.LANCZOS)
        roi_mask = roi_mask.resize(worker_size, Image.Resampling.NEAREST)
    else:
        worker_size = original_size
    image_uri = _persist(test_id, "05_completion_roi.png", _png(roi), "image/png")
    mask_uri = _persist(test_id, "05_completion_roi_mask.png", _png(roi_mask), "image/png")
    return {"image_uri": image_uri, "mask_uri": mask_uri, "box": [x1, y1, x2, y2], "original_size": list(original_size), "worker_size": list(worker_size)}


def _protected_compose(original: Image.Image, visible_mask: np.ndarray, completion_mask: np.ndarray, generated: Image.Image, roi: dict[str, Any]) -> tuple[bytes, dict[str, Any], np.ndarray]:
    x1, y1, x2, y2 = roi["box"]
    generated = generated.convert("RGB").resize(tuple(roi["original_size"]), Image.Resampling.LANCZOS)
    canvas = np.asarray(original.convert("RGB")).copy()
    generated_pixels = np.asarray(generated)
    target = completion_mask[y1:y2, x1:x2]
    canvas[y1:y2, x1:x2][target] = generated_pixels[target]
    visible_changed = int(np.any(canvas[visible_mask] != np.asarray(original.convert("RGB"))[visible_mask], axis=1).sum())
    visible_total = int(visible_mask.sum())
    final_mask = visible_mask | completion_mask
    rgba = np.dstack([canvas, np.where(final_mask, 255, 0).astype("uint8")])
    return _png(Image.fromarray(rgba, "RGBA")), {"generated_pixels": int(target.sum()), "visible_changed_pixels": visible_changed, "visible_pixel_change_ratio": round(visible_changed / max(1, visible_total), 6)}, final_mask


def _edge_refine(original: Image.Image, rgba_bytes: bytes, fallback_mask: np.ndarray) -> tuple[bytes, np.ndarray]:
    image = Image.open(io.BytesIO(rgba_bytes)).convert("RGBA") if rgba_bytes else Image.fromarray(np.dstack([np.asarray(original.convert("RGB")), np.where(fallback_mask, 255, 0).astype("uint8")]), "RGBA")
    alpha = np.asarray(image.getchannel("A").filter(ImageFilter.MedianFilter(3))) > 127
    refined = Image.fromarray(np.dstack([np.asarray(image.convert("RGB")), np.where(alpha, 255, 0).astype("uint8")]), "RGBA")
    return _png(refined), alpha


def _outline_assets(rgba_bytes: bytes, alpha: np.ndarray) -> dict[str, bytes]:
    image = Image.open(io.BytesIO(rgba_bytes)).convert("RGBA")
    rgb = np.asarray(image.convert("RGB"))
    ring = (np.asarray(Image.fromarray(np.where(alpha, 255, 0).astype("uint8"), "L").filter(ImageFilter.MaxFilter(9))) > 0) & ~alpha
    clean = _png(Image.fromarray(np.dstack([rgb, np.where(alpha, 255, 0).astype("uint8")]), "RGBA"))
    def outlined(color: tuple[int, int, int]) -> bytes:
        out = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
        out[ring] = (*color, 235)
        out[alpha, :3] = rgb[alpha]
        out[alpha, 3] = 255
        return _png(Image.fromarray(out, "RGBA"))
    return {"fish_clean.png": clean, "fish_gold_outline.png": outlined((245, 180, 40)), "fish_black_outline.png": outlined((20, 24, 22))}


@router.get("/debug/fish-completion-lab-v02", response_class=HTMLResponse)
def fish_completion_lab_v02_page(request: Request):
    return templates.TemplateResponse(request=request, name="fish_completion_auto.html", context={})


@router.post("/api/debug/fish-completion-lab-v02/auto-run")
async def auto_run(file: UploadFile = File(...), case_label: str = ""):
    data = await file.read(MAX_BYTES + 1)
    if not data or len(data) > MAX_BYTES:
        return _error(400, "INVALID_IMAGE_UPLOAD", "图片为空或超过 25 MiB", "upload")
    test_id = "FCL2_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(2)
    source = None
    try:
        with Image.open(io.BytesIO(data)) as uploaded:
            source = normalize_android_source(uploaded)
        original = source.convert("RGB")
        detector_run = detect(source)
        assessment = assess_detections(detector_run.detections)
        primary = assessment.primary
        if primary is None:
            return _error(422, "NO_RELIABLE_PRIMARY_FISH", "未找到可靠主鱼体", "detector")
        bbox_normalized = primary.box.normalized()
        bbox_pixel = _bbox_to_pixels(primary.box, source.width, source.height)
        if bbox_pixel[2] <= bbox_pixel[0] or bbox_pixel[3] <= bbox_pixel[1]:
            return _error(422, "INVALID_PRIMARY_BBOX", "主鱼体检测框无效", "detector", {"bbox_normalized": [bbox_normalized.x1, bbox_normalized.y1, bbox_normalized.x2, bbox_normalized.y2], "bbox_pixel": list(bbox_pixel)})
        result = generate_fish_cutout(source, primary.box)
        raw_mask = result.mask.astype(bool)
        analysis, completion_mask = analyze_completion(raw_mask, bbox_pixel)
        worker = {"status": "NOT_REQUIRED", "skip_reason": "completion_not_required", "endpoint_configured": bool(os.getenv("FISH_COMPLETION_WORKER_URL", "").strip())}
        generated_bytes = None
        compose = {"generated_pixels": 0, "visible_changed_pixels": 0, "visible_pixel_change_ratio": 0.0}
        final_bytes = _png(Image.fromarray(np.dstack([np.asarray(original), np.where(raw_mask, 255, 0).astype("uint8")]), "RGBA"))
        roi = None
        original_uri = _persist(test_id, "01_original_image.png", _png(original), "image/png")
        completion_uri = _persist(test_id, "03_auto_completion_mask.png", _mask_bytes(completion_mask), "image/png")
        if analysis["severity"] == "NOT_ELIGIBLE":
            worker = {"status": "NOT_ELIGIBLE", "skip_reason": analysis["reason"][0], "endpoint_configured": bool(os.getenv("FISH_COMPLETION_WORKER_URL", "").strip())}
        elif analysis["completion_required"]:
            worker = {"status": "WORKER_UNAVAILABLE", "skip_reason": "worker_endpoint_not_configured", "endpoint_configured": bool(os.getenv("FISH_COMPLETION_WORKER_URL", "").strip())}
            try:
                roi = _build_completion_roi(original, completion_mask, test_id)
                worker["roi"] = {k: roi[k] for k in ("box", "original_size", "worker_size")}
                if not worker["endpoint_configured"]:
                    raise CompletionWorkerError("COMPLETION_WORKER_NOT_CONFIGURED", "FISH_COMPLETION_WORKER_URL is not configured")
                health = check_completion_worker()
                worker["health_status"] = health.get("health_status")
                if health.get("status") != "READY":
                    raise CompletionWorkerError("COMPLETION_WORKER_HEALTH_NOT_READY", "Worker health check did not return READY")
                worker["status"] = "WORKER_RUNNING"
                worker_result = invoke_completion_worker(image_uri=roi["image_uri"], mask_uri=roi["mask_uri"], prompt="FIXED_FISH_COMPLETION_V0.2")
                generated_bytes = _read_persist(worker_result["result_uri"])
                generated_image = Image.open(io.BytesIO(generated_bytes)).convert("RGB")
                final_bytes, compose, final_mask = _protected_compose(original, raw_mask, completion_mask, generated_image, roi)
                worker = {"status": "WORKER_EXECUTED", "endpoint_configured": True, "skip_reason": None, "roi": {k: roi[k] for k in ("box", "original_size", "worker_size")}, "result": {"model_version": worker_result.get("model_version"), "inference_time_ms": worker_result.get("inference_time_ms")}}
            except CompletionWorkerError as exc:
                worker = {**worker, "status": "WORKER_TIMEOUT" if "TIMEOUT" in exc.error_code else "WORKER_FAILED", "error": {"error_code": exc.error_code, "message": str(exc)}}
            except Exception as exc:
                worker = {**worker, "status": "WORKER_FAILED", "error": {"error_code": "COMPLETION_WORKER_FAILED", "message": f"{exc.__class__.__name__}: {exc}"}}
        edge_bytes, final_mask = _edge_refine(original, final_bytes, raw_mask | completion_mask)
        asset_data = {"original_image": _data_url(_png(original), "image/png"), "sam_transparent": _data_url(result.cutout_png, "image/png"), "auto_completion_mask": _data_url(_mask_bytes(completion_mask), "image/png"), "edge_refined": _data_url(edge_bytes, "image/png")}
        asset_uris = {"original_image": original_uri, "completion_mask": completion_uri, "sam_transparent": _persist(test_id, "02_sam_transparent.png", result.cutout_png, "image/png"), "edge_refined": _persist(test_id, "07_edge_refined.png", edge_bytes, "image/png")}
        if roi:
            asset_uris.update({"completion_roi": roi["image_uri"], "completion_roi_mask": roi["mask_uri"]})
        if generated_bytes:
            asset_uris["generated_roi"] = _persist(test_id, "06_generated_roi.png", generated_bytes, "image/png")
            asset_data["generated_roi"] = _data_url(generated_bytes, "image/png")
            asset_uris["final_asset"] = _persist(test_id, "08_final_asset.png", final_bytes, "image/png")
            asset_data["final_asset"] = _data_url(final_bytes, "image/png")
        for name, content in _outline_assets(edge_bytes, final_mask).items():
            asset_uris[name] = _persist(test_id, name, content, "image/png")
            asset_data[name] = _data_url(content, "image/png")
        state = {"report_version": V02_VERSION, "runtime": {**_runtime(test_id), "demo_version": V02_VERSION}, "input": {"filename": file.filename or "uploaded", "case_label": case_label[:200], "width": source.width, "height": source.height, "size_bytes": len(data)}, "detector": {"model": detector_run.model_version, "assessment": assessment.status.value, "confidence": round(float(primary.confidence), 6), "bbox_normalized": [bbox_normalized.x1, bbox_normalized.y1, bbox_normalized.x2, bbox_normalized.y2], "bbox_pixel": list(bbox_pixel)}, "segmentation": {"model": "SAM_VIT_B", "quality": result.quality.value, "mask_area_pixels": int(raw_mask.sum()), "mask_area_ratio": round(result.mask_area_ratio, 6)}, "auto_completion": analysis, "completion_mask": {"area_pixels": int(completion_mask.sum()), "protected": True}, "worker": worker, "composition": compose, "assets": asset_uris}
        _save_state(test_id, state)
        return {"status": "ok", "stage": "outline", "test_id": test_id, "data": {**state, "assets": asset_data}, "error": None}
    except Exception as exc:
        logger.exception("Fish Completion Lab v0.2 failed; test_id=%s", test_id)
        return _error(500, "AUTO_PIPELINE_FAILED", f"{exc.__class__.__name__}: {exc}", "pipeline")
    finally:
        if source is not None:
            source.close()
