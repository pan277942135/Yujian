"""Isolated automatic Fish Completion Lab v0.2 pipeline."""
from __future__ import annotations

import io
import logging
import os
import secrets
from datetime import datetime, timezone
from typing import Any

import numpy as np
from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageFilter

from app.detector_runtime import detect, normalize_android_source
from app.completion_worker_client import invoke_completion_worker
from app.fish_completion_lab import MAX_BYTES, _data_url, _mask_bytes, _persist, _png, _runtime, _save_state
from app.recognition_pipeline import assess_detections
from app.segmentation.service import generate_fish_cutout

router = APIRouter(tags=["fish-completion-lab-v02"])
templates = Jinja2Templates(directory="app/templates")
V02_VERSION = "YUJIAN_FISH_COMPLETION_LAB_V0.2"
logger = logging.getLogger(__name__)


def _error(status: int, code: str, message: str, stage: str, detail: Any = None):
    return JSONResponse(status_code=status, content={"status": "error", "stage": stage, "error": {"error_code": code, "message": message, "detail": detail or {}}})


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
    """Conservative contour-gap heuristic; never proposes pixels outside the primary box."""
    h, w = mask.shape
    x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(w, bbox[2]), min(h, bbox[3])
    box = np.zeros_like(mask)
    if x2 <= x1 or y2 <= y1:
        return {"completion_required": False, "severity": "NOT_ELIGIBLE", "completion_ratio": 1.0, "reason": ["invalid_bbox"], "region_count": 0}, box
    box[y1:y2, x1:x2] = True
    closed = Image.fromarray(np.where(mask & box, 255, 0).astype("uint8"), "L").filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.MinFilter(5))
    candidate = (np.asarray(closed) > 127) & ~mask & box
    area = int(mask.sum())
    missing = int(candidate.sum())
    ratio = missing / max(1, area + missing)
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


def _edge_refine(original: Image.Image, mask: np.ndarray) -> tuple[bytes, np.ndarray]:
    refined = np.asarray(Image.fromarray(np.where(mask, 255, 0).astype("uint8"), "L").filter(ImageFilter.MedianFilter(3))) > 127
    rgba = np.dstack([np.asarray(original.convert("RGB")), np.where(refined, 255, 0).astype("uint8")])
    return _png(Image.fromarray(rgba, "RGBA")), refined


def _output_assets(original: Image.Image, alpha: np.ndarray) -> dict[str, bytes]:
    rgb = np.asarray(original.convert("RGB"))
    a = np.where(alpha, 255, 0).astype("uint8")
    ring = (np.asarray(Image.fromarray(a, "L").filter(ImageFilter.MaxFilter(9))) > 0) & ~alpha
    clean = _png(Image.fromarray(np.dstack([rgb, a]), "RGBA"))
    def outlined(color: tuple[int, int, int]) -> bytes:
        out = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
        out[ring] = (*color, 235)
        out[alpha, :3] = rgb[alpha]
        out[alpha, 3] = 255
        return _png(Image.fromarray(out, "RGBA"))
    return {"fish_clean.png": clean, "fish_gold_outline.png": outlined((245, 180, 40)), "fish_black_outline.png": outlined((20, 24, 22))}


def _store_data(test_id: str, name: str, data: bytes) -> str:
    _persist(test_id, name, data, "image/png")
    return _data_url(data, "image/png")


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
        result = generate_fish_cutout(source, primary.box)
        raw_mask = result.mask.astype(bool)
        bbox = (int(primary.box.x1), int(primary.box.y1), int(primary.box.x2), int(primary.box.y2))
        analysis, completion_mask = analyze_completion(raw_mask, bbox)
        refined_png, refined = _edge_refine(original, raw_mask)
        original_bytes = _png(original)
        original_uri = _persist(test_id, "01_original_image.png", original_bytes, "image/png")
        completion_bytes = _mask_bytes(completion_mask)
        completion_uri = _persist(test_id, "03_auto_completion_mask.png", completion_bytes, "image/png")
        assets = {
            "original_image": _data_url(original_bytes, "image/png"),
            "sam_transparent": _store_data(test_id, "02_sam_transparent.png", result.cutout_png),
            "auto_completion_mask": _data_url(completion_bytes, "image/png"),
            "edge_refined": _store_data(test_id, "04_edge_refined.png", refined_png),
        }
        for name, content in _output_assets(original, refined).items():
            assets[name] = _store_data(test_id, name, content)
        worker_status = "SKIPPED" if not analysis["completion_required"] else ("WORKER_READY" if os.getenv("FISH_COMPLETION_WORKER_URL", "").strip() else "WORKER_UNAVAILABLE")
        worker_result = None
        worker_error = None
        if worker_status == "WORKER_READY":
            try:
                worker_result = invoke_completion_worker(image_uri=original_uri, mask_uri=completion_uri, prompt="FIXED_FISH_COMPLETION_V0.2")
            except Exception as exc:
                worker_status = "WORKER_FAILED"
                worker_error = {"error_code": getattr(exc, "error_code", "WORKER_FAILED"), "message": str(exc)}
        state = {"report_version": V02_VERSION, "runtime": {**_runtime(test_id), "demo_version": V02_VERSION}, "input": {"filename": file.filename or "uploaded", "case_label": case_label[:200], "width": source.width, "height": source.height, "size_bytes": len(data)}, "detector": {"model": detector_run.model_version, "assessment": assessment.status.value, "confidence": round(float(primary.confidence), 6), "bbox": bbox}, "segmentation": {"model": "SAM_VIT_B", "quality": result.quality.value, "mask_area_ratio": round(result.mask_area_ratio, 6)}, "auto_completion": analysis, "completion_mask": {"area_pixels": int(completion_mask.sum()), "protected": True}, "worker": {"status": worker_status, "endpoint_configured": bool(os.getenv("FISH_COMPLETION_WORKER_URL", "").strip())}, "assets": assets}
        state["worker"].update({"result": worker_result, "error": worker_error})
        _save_state(test_id, state)
        return {"status": "ok", "stage": "outline", "test_id": test_id, "data": state, "error": None}
    except Exception as exc:
        logger.exception("Fish Completion Lab v0.2 failed; test_id=%s", test_id)
        return _error(500, "AUTO_PIPELINE_FAILED", f"{exc.__class__.__name__}: {exc}", "pipeline")
    finally:
        if source is not None:
            source.close()

