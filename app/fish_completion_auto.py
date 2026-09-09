"""Isolated Fish Completion Lab v0.2 automatic pipeline."""
from __future__ import annotations

import base64
import io
import logging
import os
import secrets
from collections import deque
from datetime import datetime, timezone
from math import ceil, floor
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageFilter
from sqlalchemy import select

from app.db import get_db
from app.dataset_models import DatasetItem
from app.models import DatasetVersion

from app.completion_worker_client import CompletionWorkerError, check_completion_worker, invoke_completion_worker
from app.detector_runtime import detect, normalize_android_source
from app.fish_completion_lab import MAX_BYTES, _data_url, _mask_bytes, _persist, _png, _read_dataset_image, _read_persist, _runtime, _save_state
from app.recognition_pipeline import assess_detections
from app.segmentation.service import generate_fish_cutout

router = APIRouter(tags=["fish-completion-lab-v02"])
templates = Jinja2Templates(directory="app/templates")
V02_VERSION = "YUJIAN_FISH_COMPLETION_LAB_V0.2"
logger = logging.getLogger(__name__)
MIN_COMPONENT_ABS_PIXELS = 16
MIN_COMPONENT_RELATIVE_AREA = 0.0005
MAX_EFFECTIVE_REGIONS = 2
BOUNDARY_RING_DISTANCE_PX = 3




def _auto_progress(state: dict[str, Any]) -> list[dict[str, Any]]:
    timings = state.get("timings", {})
    detector = state.get("detector", {})
    segmentation = state.get("segmentation", {})
    analysis = state.get("auto_completion", {})
    mask = state.get("completion_mask", {})
    worker = state.get("worker", {})
    composition = state.get("composition", {})
    assets = state.get("assets", {})
    return [
        {"stage": "input", "label": "图片输入", "status": "READY", "elapsed_ms": timings.get("input_decode_ms"), "result": state.get("input", {}).get("filename")},
        {"stage": "detector", "label": "Detector / BBox", "status": "READY" if detector else "PENDING", "elapsed_ms": timings.get("detector_ms"), "result": {"bbox_pixel": detector.get("bbox_pixel"), "confidence": detector.get("confidence")}},
        {"stage": "sam", "label": "SAM 分割", "status": "READY" if segmentation else "PENDING", "elapsed_ms": timings.get("sam_ms"), "result": segmentation.get("quality")},
        {"stage": "analysis", "label": "完整度判断", "status": "READY" if analysis else "PENDING", "elapsed_ms": timings.get("analysis_ms"), "result": {"required": analysis.get("completion_required"), "ratio": analysis.get("completion_ratio")}},
        {"stage": "mask", "label": "Auto Completion Mask", "status": "AUTO_GENERATED" if mask.get("area_pixels", 0) else ("NOT_REQUIRED" if analysis.get("completion_required") is False else "PENDING"), "elapsed_ms": timings.get("mask_ms"), "result": mask.get("area_pixels", 0)},
        {"stage": "roi", "label": "ROI 提取", "status": "READY" if worker.get("roi") else "SKIPPED", "elapsed_ms": timings.get("roi_ms"), "result": worker.get("roi")},
        {"stage": "powerpaint", "label": "PowerPaint Worker（补全执行服务）", "status": worker.get("status", "PENDING"), "elapsed_ms": timings.get("worker_ms"), "result": {"configured": worker.get("endpoint_configured"), "health": worker.get("health_status"), "inference_time_ms": worker.get("inference_time_ms")}},
        {"stage": "compose", "label": "Protected Compose（受保护合成）", "status": "READY" if composition.get("visible_pixel_change_ratio") is not None else "PENDING", "elapsed_ms": timings.get("compose_ms"), "result": {"generated_pixels": composition.get("generated_pixels"), "visible_changed_pixels": composition.get("visible_changed_pixels"), "visible_pixel_change_ratio": composition.get("visible_pixel_change_ratio")}},
        {"stage": "final", "label": "Edge / Outline / Final Asset", "status": "READY" if assets.get("fish_clean.png") and assets.get("fish_gold_outline.png") and assets.get("fish_black_outline.png") else "PENDING", "elapsed_ms": timings.get("final_ms"), "result": {"edge_refined": bool(assets.get("edge_refined")), "final_asset": bool(assets.get("final_asset"))}},
    ]

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


def _component_masks(mask: np.ndarray) -> list[np.ndarray]:
    """Return connected components without changing the source mask."""
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[np.ndarray] = []
    for y, x in zip(*np.where(mask)):
        if seen[y, x]:
            continue
        current = np.zeros(mask.shape, dtype=bool)
        stack = [(int(y), int(x))]
        seen[y, x] = True
        current[y, x] = True
        while stack:
            cy, cx = stack.pop()
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    current[ny, nx] = True
                    stack.append((ny, nx))
        components.append(current)
    return components


def _remove_mask_noise(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Keep only the largest meaningful SAM component for analysis."""
    x1, y1, x2, y2 = bbox
    clipped = np.zeros_like(mask)
    clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    components = _component_masks(clipped)
    if not components:
        return clipped
    largest = max(components, key=lambda component: int(component.sum()))
    threshold = max(MIN_COMPONENT_ABS_PIXELS, int(largest.sum() * MIN_COMPONENT_RELATIVE_AREA))
    meaningful = np.zeros_like(mask)
    for component in components:
        if int(component.sum()) >= threshold and int(component.sum()) >= int(largest.sum() * 0.01):
            meaningful |= component
    return meaningful if meaningful.any() else largest


def _boundary_ring_metrics(mask: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    """Measure whether a candidate is an outer boundary ring."""
    if not candidate.any() or not mask.any():
        return 0.0, 0.0
    boundary = mask & ~(
        np.asarray(Image.fromarray((mask * 255).astype("uint8"), "L").filter(ImageFilter.MinFilter(3))) > 127
    )
    near_boundary = np.asarray(
        Image.fromarray((boundary * 255).astype("uint8"), "L").filter(ImageFilter.MaxFilter(2 * BOUNDARY_RING_DISTANCE_PX + 1))
    ) > 127
    overlap_ratio = float((candidate & near_boundary).sum() / max(1, int(candidate.sum())))
    boundary_pixels = int(boundary.sum())
    coverage = float((candidate & near_boundary).sum() / max(1, boundary_pixels))
    return round(overlap_ratio, 6), round(min(1.0, coverage), 6)


def _enclosed_holes(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Return only background holes enclosed by the visible mask.

    This intentionally does not dilate the outside contour. An outside contour
    is not evidence that the fish is incomplete and must never become a worker
    completion region by itself.
    """
    h, w = mask.shape
    x1, y1, x2, y2 = bbox
    region = mask[y1:y2, x1:x2]
    if not region.size:
        return np.zeros_like(mask)
    background = ~region
    reachable = np.zeros_like(background, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for x in range(region.shape[1]):
        for y in (0, region.shape[0] - 1):
            if background[y, x] and not reachable[y, x]:
                reachable[y, x] = True
                queue.append((y, x))
    for y in range(region.shape[0]):
        for x in (0, region.shape[1] - 1):
            if background[y, x] and not reachable[y, x]:
                reachable[y, x] = True
                queue.append((y, x))
    while queue:
        cy, cx = queue.popleft()
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < region.shape[0] and 0 <= nx < region.shape[1] and background[ny, nx] and not reachable[ny, nx]:
                reachable[ny, nx] = True
                queue.append((ny, nx))
    result = np.zeros_like(mask)
    result[y1:y2, x1:x2] = background & ~reachable
    return result


def analyze_completion_details(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    h, w = mask.shape
    x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(w, bbox[2]), min(h, bbox[3])
    if x2 <= x1 or y2 <= y1:
        empty = np.zeros_like(mask)
        return {"completion_required": False, "severity": "NOT_ELIGIBLE", "completion_ratio": 1.0, "completion_percent": 100.0, "reason": ["invalid_bbox"], "region_count": 0, "detection_method": "enclosed_hole_fill"}, empty, empty, empty
    analysis_visible = _remove_mask_noise(mask, (x1, y1, x2, y2))
    raw_candidate = _enclosed_holes(analysis_visible, (x1, y1, x2, y2))
    raw_components = _component_masks(raw_candidate)
    fish_area = max(1, int(analysis_visible.sum()))
    component_threshold = max(MIN_COMPONENT_ABS_PIXELS, int(fish_area * MIN_COMPONENT_RELATIVE_AREA))
    filtered_candidate = np.zeros_like(mask)
    for component in raw_components:
        if int(component.sum()) >= component_threshold:
            filtered_candidate |= component
    raw_regions = len(raw_components)
    # Keep the unfiltered structural candidate for diagnostics and ROI review;
    # only the accepted mask may reach the worker.
    candidate = raw_candidate
    regions = _component_count(filtered_candidate)
    missing = int(candidate.sum())
    ratio = missing / max(1, fish_area + missing)
    boundary_overlap_ratio, perimeter_coverage = _boundary_ring_metrics(analysis_visible, candidate)
    reasons: list[str] = []
    severity, required = "COMPLETION_NOT_REQUIRED", False
    if raw_regions > MAX_EFFECTIVE_REGIONS and regions == 0:
        severity, reasons = "NOT_ELIGIBLE", ["candidate_too_fragmented"]
    elif regions > MAX_EFFECTIVE_REGIONS:
        severity, reasons = "NOT_ELIGIBLE", ["too_many_candidate_regions"]
    elif boundary_overlap_ratio >= 0.9 and perimeter_coverage >= 0.35:
        severity, reasons = "NOT_ELIGIBLE", ["boundary_ring_candidate"]
    elif missing:
        reasons = ["internal_gap_detected"]
        if ratio <= .05: severity, required = "LIGHT", True
        elif ratio <= .10: severity, required = "MEDIUM", True
        elif ratio <= .15: severity, required = "HEAVY", True
        elif ratio <= .20: severity, required = "EXTREME_EXPERIMENTAL", True
        else: severity, required, reasons = "LARGE_EXPERIMENTAL", True, ["internal_gap_detected", "large_completion_area"]
    analysis = {"completion_required": required, "severity": severity, "completion_ratio": round(ratio, 6), "completion_percent": round(ratio * 100, 4), "reason": reasons or ["visible_mask_sufficient"], "region_count": regions, "candidate_area_pixels": missing, "raw_region_count": raw_regions, "boundary_overlap_ratio": boundary_overlap_ratio, "perimeter_coverage": perimeter_coverage, "detection_method": "enclosed_hole_fill"}
    accepted = filtered_candidate if required else np.zeros_like(mask)
    structural_envelope = analysis_visible | candidate
    return analysis, candidate, accepted, structural_envelope


def analyze_completion(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[dict[str, Any], np.ndarray]:
    analysis, candidate, _accepted, _envelope = analyze_completion_details(mask, bbox)
    return analysis, candidate


def _build_completion_roi(original: Image.Image, completion_mask: np.ndarray, test_id: str) -> dict[str, Any]:
    ys, xs = np.where(completion_mask)
    if not len(xs):
        raise ValueError("COMPLETION_MASK_EMPTY")
    h, w = completion_mask.shape
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    padding_ratio = min(1.0, max(0.5, float(os.getenv("FISH_COMPLETION_ROI_PADDING_RATIO", "0.75"))))
    pad_x, pad_y = max(8, int((x2 - x1) * padding_ratio)), max(8, int((y2 - y1) * padding_ratio))
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


def _mask_overlay_bytes(original: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> bytes:
    """Composite a diagnostic mask over the original without destroying the base image."""
    base = original.convert("RGBA")
    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    overlay[mask, :3] = color
    overlay[mask, 3] = 145
    return _png(Image.alpha_composite(base, Image.fromarray(overlay, "RGBA")))


@router.get("/debug/fish-completion-lab-v02", response_class=HTMLResponse)
def fish_completion_lab_v02_page(request: Request):
    return templates.TemplateResponse(request=request, name="fish_completion_auto.html", context={})


@router.post("/api/debug/fish-completion-lab-v02/auto-run")
async def auto_run(file: UploadFile | None = File(default=None), case_label: str = Form(default=""), source_type: str = Form(default="local_upload"), dataset_version: str = Form(default=""), dataset_item_id: str = Form(default=""), db=Depends(get_db)):
    total_started = datetime.now(timezone.utc)
    test_id = "FCL2_" + total_started.strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(2)
    source = None
    try:
        if source_type == "dataset_freeze" or dataset_item_id:
            if not dataset_version or not dataset_item_id:
                return _error(400, "DATASET_SELECTION_REQUIRED", "请先选择 Dataset Freeze（冻结数据集）图片", "input")
            dataset = db.get(DatasetVersion, dataset_version)
            if not dataset or dataset.status != "FROZEN":
                return _error(409, "DATASET_VERSION_NOT_FROZEN", "只能使用已冻结 Dataset Freeze（冻结数据集）版本", "input")
            try:
                item_id = int(dataset_item_id)
            except (TypeError, ValueError):
                return _error(400, "DATASET_ITEM_ID_INVALID", "数据集图片 ID 无效", "input")
            dataset_item = db.scalar(select(DatasetItem).where(DatasetItem.dataset_version == dataset_version, DatasetItem.id == item_id))
            if not dataset_item:
                return _error(404, "DATASET_ITEM_NOT_FOUND", "数据集图片不存在", "input")
            data = _read_dataset_image(dataset_item)
            input_filename = dataset_item.image_id or f"dataset-{dataset_item.id}"
        else:
            if file is None:
                return _error(400, "IMAGE_UPLOAD_REQUIRED", "请先选择本地图片", "input")
            data = await file.read(MAX_BYTES + 1)
            input_filename = file.filename or "uploaded"
        if not data or len(data) > MAX_BYTES:
            return _error(400, "INVALID_IMAGE_UPLOAD", "图片为空或超过 25 MiB", "input")
        decode_started = datetime.now(timezone.utc)
        with Image.open(io.BytesIO(data)) as uploaded:
            source = normalize_android_source(uploaded)
        input_decode_ms = round((datetime.now(timezone.utc) - decode_started).total_seconds() * 1000, 2)
        original = source.convert("RGB")
        detector_started = datetime.now(timezone.utc)
        detector_run = detect(source)
        assessment = assess_detections(detector_run.detections)
        detector_ms = round((datetime.now(timezone.utc) - detector_started).total_seconds() * 1000, 2)
        primary = assessment.primary
        if primary is None:
            return _error(422, "NO_RELIABLE_PRIMARY_FISH", "未找到可靠主鱼体", "detector")
        bbox_normalized = primary.box.normalized()
        bbox_pixel = _bbox_to_pixels(primary.box, source.width, source.height)
        if bbox_pixel[2] <= bbox_pixel[0] or bbox_pixel[3] <= bbox_pixel[1]:
            return _error(422, "INVALID_PRIMARY_BBOX", "主鱼体检测框无效", "detector", {"bbox_normalized": [bbox_normalized.x1, bbox_normalized.y1, bbox_normalized.x2, bbox_normalized.y2], "bbox_pixel": list(bbox_pixel)})
        sam_started = datetime.now(timezone.utc)
        result = generate_fish_cutout(source, primary.box)
        sam_ms = round((datetime.now(timezone.utc) - sam_started).total_seconds() * 1000, 2)
        raw_mask = result.mask.astype(bool)
        analysis_started = datetime.now(timezone.utc)
        analysis, completion_candidate, completion_mask, structural_envelope = analyze_completion_details(raw_mask, bbox_pixel)
        analysis_ms = round((datetime.now(timezone.utc) - analysis_started).total_seconds() * 1000, 2)
        analysis_visible_mask = _remove_mask_noise(raw_mask, bbox_pixel)
        worker = {"status": "NOT_REQUIRED", "skip_reason": "completion_not_required", "endpoint_configured": bool(os.getenv("FISH_COMPLETION_WORKER_URL", "").strip())}
        generated_bytes = None
        compose = {"generated_pixels": 0, "visible_changed_pixels": 0, "visible_pixel_change_ratio": 0.0}
        final_bytes = _png(Image.fromarray(np.dstack([np.asarray(original), np.where(raw_mask, 255, 0).astype("uint8")]), "RGBA"))
        roi = None
        mask_ms = round((datetime.now(timezone.utc) - analysis_started).total_seconds() * 1000, 2)
        original_uri = _persist(test_id, "01_original_image.png", _png(original), "image/png")
        completion_uri = _persist(test_id, "03_auto_completion_mask.png", _mask_bytes(completion_mask), "image/png")
        if analysis["severity"] == "NOT_ELIGIBLE":
            worker = {"status": "NOT_ELIGIBLE", "skip_reason": analysis["reason"][0], "endpoint_configured": bool(os.getenv("FISH_COMPLETION_WORKER_URL", "").strip())}
        elif analysis["completion_required"]:
            worker = {"status": "WORKER_UNAVAILABLE", "skip_reason": "worker_endpoint_not_configured", "endpoint_configured": bool(os.getenv("FISH_COMPLETION_WORKER_URL", "").strip())}
            try:
                roi_started = datetime.now(timezone.utc)
                roi = _build_completion_roi(original, completion_mask, test_id)
                roi_ms = round((datetime.now(timezone.utc) - roi_started).total_seconds() * 1000, 2)
                worker["roi"] = {k: roi[k] for k in ("box", "original_size", "worker_size")}
                if not worker["endpoint_configured"]:
                    raise CompletionWorkerError("COMPLETION_WORKER_NOT_CONFIGURED", "FISH_COMPLETION_WORKER_URL is not configured")
                health = check_completion_worker()
                worker["health_status"] = health.get("health_status")
                if health.get("status") != "READY":
                    raise CompletionWorkerError("COMPLETION_WORKER_HEALTH_NOT_READY", "Worker health check did not return READY")
                worker["status"] = "WORKER_RUNNING"
                worker_started = datetime.now(timezone.utc)
                worker_result = invoke_completion_worker(image_uri=roi["image_uri"], mask_uri=roi["mask_uri"], prompt=("Complete the missing part of the fish body. Preserve original fish species, anatomy, scales, fins and natural texture. Do not modify visible fish pixels."))
                worker_ms = round((datetime.now(timezone.utc) - worker_started).total_seconds() * 1000, 2)
                worker["inference_time_ms"] = worker_result.get("inference_time_ms")
                worker["model_version"] = worker_result.get("model_version")
                generated_ref = worker_result.get("result_uri")
                if generated_ref and generated_ref.startswith("data:"):
                    generated_bytes = base64.b64decode(generated_ref.split(",", 1)[1])
                elif (worker_result.get("generated_roi") or "").startswith("data:"):
                    generated_bytes = base64.b64decode(worker_result["generated_roi"].split(",", 1)[1])
                else:
                    generated_bytes = _read_persist(generated_ref)
                generated_image = Image.open(io.BytesIO(generated_bytes)).convert("RGB")
                compose_started = datetime.now(timezone.utc)
                final_bytes, compose, final_mask = _protected_compose(original, raw_mask, completion_mask, generated_image, roi)
                compose_ms = round((datetime.now(timezone.utc) - compose_started).total_seconds() * 1000, 2)
                worker = {"status": "WORKER_EXECUTED", "endpoint_configured": True, "health_status": health.get("health_status"), "skip_reason": None, "inference_time_ms": worker_result.get("inference_time_ms"), "model_version": worker_result.get("model_version"), "roi": {k: roi[k] for k in ("box", "original_size", "worker_size")}, "result": {"model_version": worker_result.get("model_version"), "inference_time_ms": worker_result.get("inference_time_ms")}}
            except CompletionWorkerError as exc:
                worker = {**worker, "status": "WORKER_TIMEOUT" if "TIMEOUT" in exc.error_code else "WORKER_FAILED", "error": {"error_code": exc.error_code, "message": str(exc)}}
            except Exception as exc:
                worker = {**worker, "status": "WORKER_FAILED", "error": {"error_code": "COMPLETION_WORKER_FAILED", "message": f"{exc.__class__.__name__}: {exc}"}}
        edge_bytes, final_mask = _edge_refine(original, final_bytes, raw_mask | completion_mask)
        debug_bytes = {
            "analysis_visible_mask": _mask_bytes(analysis_visible_mask),
            "structural_envelope": _mask_bytes(structural_envelope),
            "completion_candidate": _mask_bytes(completion_candidate),
            "auto_completion_mask": _mask_bytes(completion_mask),
            "completion_candidate_overlay": _mask_overlay_bytes(original, completion_candidate, (255, 128, 0)),
            "accepted_completion_overlay": _mask_overlay_bytes(original, completion_mask, (40, 150, 255)),
        }
        debug_uris = {key: _persist(test_id, f"03_{key}.png", content, "image/png") for key, content in debug_bytes.items()}
        asset_data = {"original_image": _data_url(_png(original), "image/png"), "sam_transparent": _data_url(result.cutout_png, "image/png"), "edge_refined": _data_url(edge_bytes, "image/png"), **{key: _data_url(content, "image/png") for key, content in debug_bytes.items()}}
        asset_uris = {"original_image": original_uri, "completion_mask": completion_uri, "sam_transparent": _persist(test_id, "02_sam_transparent.png", result.cutout_png, "image/png"), "edge_refined": _persist(test_id, "07_edge_refined.png", edge_bytes, "image/png"), **debug_uris}
        asset_uris["final_asset"] = _persist(test_id, "08_final_asset.png", final_bytes, "image/png")
        asset_data["final_asset"] = _data_url(final_bytes, "image/png")
        if roi:
            asset_uris.update({"completion_roi": roi["image_uri"], "completion_roi_mask": roi["mask_uri"]})
        if generated_bytes:
            asset_uris["generated_roi"] = _persist(test_id, "06_generated_roi.png", generated_bytes, "image/png")
            asset_data["generated_roi"] = _data_url(generated_bytes, "image/png")
        for name, content in _outline_assets(edge_bytes, final_mask).items():
            asset_uris[name] = _persist(test_id, name, content, "image/png")
            asset_data[name] = _data_url(content, "image/png")
        final_ms = round((datetime.now(timezone.utc) - total_started).total_seconds() * 1000, 2)
        timings = {"input_decode_ms": input_decode_ms, "detector_ms": detector_ms, "sam_ms": sam_ms, "analysis_ms": analysis_ms, "mask_ms": mask_ms, "roi_ms": locals().get("roi_ms"), "worker_ms": locals().get("worker_ms"), "compose_ms": locals().get("compose_ms"), "final_ms": final_ms, "total_ms": final_ms}
        state = {"report_version": V02_VERSION, "runtime": {**_runtime(test_id), "demo_version": V02_VERSION}, "completion_source": "AUTO", "completion_mode": "AUTO_COMPLETION", "timings": timings, "input": {"filename": input_filename, "case_label": case_label[:200], "source_type": source_type[:32], "dataset_version": dataset_version[:128] or None, "dataset_item_id": dataset_item_id[:80] or None, "width": source.width, "height": source.height, "size_bytes": len(data)}, "detector": {"model": detector_run.model_version, "assessment": assessment.status.value, "confidence": round(float(primary.confidence), 6), "bbox_normalized": [bbox_normalized.x1, bbox_normalized.y1, bbox_normalized.x2, bbox_normalized.y2], "bbox_pixel": list(bbox_pixel)}, "segmentation": {"model": "SAM_VIT_B", "quality": result.quality.value, "mask_area_pixels": int(raw_mask.sum()), "mask_area_ratio": round(result.mask_area_ratio, 6)}, "auto_completion": analysis, "completion_mask": {"area_pixels": int(completion_mask.sum()), "candidate_area_pixels": int(completion_candidate.sum()), "protected": True}, "worker": worker, "composition": compose, "trace": {"completion_mode": "AUTO_COMPLETION", "completion_required": analysis["completion_required"], "completion_ratio": analysis["completion_ratio"], "completion_area_pixels": int(completion_mask.sum()), "roi_bbox": worker.get("roi", {}).get("box"), "worker_configured": worker.get("endpoint_configured"), "worker_status": worker.get("status"), "worker_time": worker.get("inference_time_ms"), "generated_pixels": compose.get("generated_pixels", 0), "visible_changed_pixels": compose.get("visible_changed_pixels", 0), "analysis_visible_pixels": int(analysis_visible_mask.sum()), "candidate_area_pixels": int(completion_candidate.sum()), "accepted_area_pixels": int(completion_mask.sum()), "completion_percent": analysis["completion_percent"], "boundary_overlap_ratio": analysis["boundary_overlap_ratio"], "perimeter_coverage": analysis["perimeter_coverage"]}, "assets": asset_uris}
        state["progress"] = _auto_progress(state)
        _save_state(test_id, state)
        return {"status": "ok", "stage": "outline", "test_id": test_id, "data": {**state, "assets": asset_data}, "error": None}
    except Exception as exc:
        logger.exception("Fish Completion Lab v0.2 failed; test_id=%s", test_id)
        return _error(500, "AUTO_PIPELINE_FAILED", f"{exc.__class__.__name__}: {exc}", "pipeline")
    finally:
        if source is not None:
            source.close()
