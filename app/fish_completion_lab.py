"""Experimental Fish Completion Lab.

This module is deliberately isolated from the production recognition pipeline.
It reuses the production detector and SAM segmentation entry points, while
keeping manual masks and completion execution behind an experimental API.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import secrets
from datetime import datetime, timezone
from typing import Any

import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from google.cloud import storage
from PIL import Image
from pydantic import BaseModel, Field

from app.detector_runtime import detect, normalize_android_source
from app.recognition_pipeline import assess_detections
from app.segmentation.service import generate_fish_cutout

router = APIRouter(tags=["fish-completion-lab"])
templates = Jinja2Templates(directory="app/templates")
LAB_VERSION = "YUJIAN_FISH_COMPLETION_LAB_V0.1"
MAX_BYTES = 25 * 1024 * 1024
PREFIX = "experiments/fish_completion_lab/v0.1"


class MaskPayload(BaseModel):
    test_id: str = Field(min_length=1, max_length=80)
    masks: dict[str, str] = Field(default_factory=dict)


class RunPayload(BaseModel):
    test_id: str = Field(min_length=1, max_length=80)


class ReviewPayload(BaseModel):
    test_id: str = Field(min_length=1, max_length=80)
    review: dict[str, Any] = Field(default_factory=dict)


def _bucket():
    name = os.getenv("GCS_BUCKET", "").strip()
    return storage.Client().bucket(name) if name else None


def _persist(test_id: str, name: str, content: bytes, content_type: str = "application/octet-stream") -> str:
    bucket = _bucket()
    object_name = f"{PREFIX}/{test_id}/{name}"
    if bucket is None:
        path = os.path.join("var", "fish_completion_lab", test_id, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(content)
        return f"local://{path}"
    blob = bucket.blob(object_name)
    blob.upload_from_string(content, content_type=content_type)
    return f"gs://{bucket.name}/{object_name}"


def _read_persist(uri: str) -> bytes:
    if uri.startswith("gs://"):
        remainder = uri[len("gs://"):]
        bucket_name, object_name = remainder.split("/", 1)
        return storage.Client().bucket(bucket_name).blob(object_name).download_as_bytes()
    with open(uri.removeprefix("local://"), "rb"):
        return handle.read()


def _save_json(test_id: str, name: str, value: dict[str, Any]) -> str:
    return _persist(test_id, name, json.dumps(value, ensure_ascii=False, indent=2).encode(), "application/json")


def _data_url(data: bytes, media_type: str) -> str:
    return f"data:{media_type};base64,{base64.b64encode(data).decode()}"


def _png(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.save(out, format="PNG", optimize=True)
    return out.getvalue()


def _decode_mask(value: str, width: int, height: int) -> np.ndarray:
    raw = value.split(",", 1)[1] if "," in value else value
    try:
        image = Image.open(io.BytesIO(base64.b64decode(raw))).convert("L")
    except Exception as exc:
        raise HTTPException(400, "mask 必须是 PNG data URL") from exc
    if image.size != (width, height):
        raise HTTPException(400, f"mask 尺寸必须等于原图 {width}x{height}")
    return np.asarray(image, dtype=np.uint8) > 127


def _mask_bytes(mask: np.ndarray) -> bytes:
    return _png(Image.fromarray(np.where(mask, 255, 0).astype("uint8"), "L"))


def _components(mask: np.ndarray) -> list[int]:
    seen = np.zeros(mask.shape, dtype=bool)
    sizes = []
    h, w = mask.shape
    for y, x in zip(*np.where(mask & ~seen)):
        if seen[y, x]:
            continue
        stack = [(int(y), int(x))]
        seen[y, x] = True
        size = 0
        while stack:
            cy, cx = stack.pop()
            size += 1
            for ny, nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        sizes.append(size)
    return sorted(sizes, reverse=True)


def _stats(raw: np.ndarray, add: np.ndarray, remove: np.ndarray, occluder: np.ndarray, completion: np.ndarray) -> dict[str, Any]:
    refined = (raw | add) & ~remove
    illegal = completion & ~occluder
    refined_area = int(refined.sum())
    completion_area = int(completion.sum())
    denominator = refined_area + completion_area
    ratio = completion_area / denominator if denominator else 0.0
    if ratio == 0:
        level, eligible_reason = "LIGHT", "COMPLETION_NOT_REQUIRED"
    elif ratio <= .05:
        level, eligible_reason = "LIGHT", "WITHIN_LIGHT_LIMIT"
    elif ratio <= .10:
        level, eligible_reason = "MEDIUM", "WITHIN_MEDIUM_LIMIT"
    elif ratio <= .15:
        level, eligible_reason = "HEAVY", "WITHIN_HEAVY_LIMIT"
    elif ratio <= .20:
        level, eligible_reason = "EXTREME_EXPERIMENTAL", "WITHIN_EXPERIMENTAL_LIMIT"
    else:
        level, eligible_reason = "NOT_ELIGIBLE", "GENERATED_RATIO_OVER_20_PERCENT"
    regions = len(_components(completion))
    if regions > 2:
        eligible_reason = "COMPLETION_COMPLEX"
    return {
        "raw_sam_area_pixels": int(raw.sum()),
        "visible_added_pixels": int(add.sum()),
        "removed_nonfish_pixels": int(remove.sum()),
        "refined_visible_area_pixels": refined_area,
        "occluder_area_pixels": int(occluder.sum()),
        "occluder_region_count": len(_components(occluder)),
        "completion_area_pixels": completion_area,
        "completion_region_count": regions,
        "estimated_final_fish_area_pixels": refined_area + completion_area,
        "generated_pixel_ratio": round(ratio, 6),
        "completion_level": level,
        "completion_mask_valid": not bool(illegal.any()),
        "illegal_completion_pixels": int(illegal.sum()),
        "eligibility_reason": eligible_reason,
        "eligible_for_v0_1": bool(not illegal.any() and ratio <= .20 and regions <= 2),
    }


def _runtime(test_id: str) -> dict[str, Any]:
    return {
        "test_id": test_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "demo_version": LAB_VERSION,
        "service": os.getenv("K_SERVICE", "local"),
        "revision": os.getenv("K_REVISION", "unknown"),
        "commit": os.getenv("APP_GIT_COMMIT", "unknown"),
    }


def _state_uri(test_id: str) -> str:
    return f"{PREFIX}/{test_id}/16_test_report.json"


def _load_state(test_id: str) -> dict[str, Any]:
    bucket = _bucket()
    if bucket:
        blob = bucket.blob(_state_uri(test_id))
        if not blob.exists():
            raise HTTPException(404, "test_id 不存在")
        return json.loads(blob.download_as_text())
    path = os.path.join("var", "fish_completion_lab", test_id, "16_test_report.json")
    if not os.path.exists(path):
        raise HTTPException(404, "test_id 不存在")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _save_state(test_id: str, state: dict[str, Any]) -> None:
    _save_json(test_id, "16_test_report.json", state)


@router.get("/debug/fish-completion-lab", response_class=HTMLResponse)
def fish_completion_lab_page(request: Request):
    return templates.TemplateResponse(request=request, name="fish_completion_lab.html", context={})


@router.post("/api/debug/fish-completion-lab/prepare")
async def prepare(file: UploadFile = File(...), case_label: str = ""):
    data = await file.read(MAX_BYTES + 1)
    if not data or len(data) > MAX_BYTES:
        raise HTTPException(400, "图片为空或超过 25 MiB")
    test_id = "FCL_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(2)
    try:
        with Image.open(io.BytesIO(data)) as uploaded:
            source = normalize_android_source(uploaded)
        original = _png(source.convert("RGB"))
        detector_run = detect(source)
        assessment = assess_detections(detector_run.detections)
        primary = assessment.primary
        if primary is None:
            raise HTTPException(422, "NO_RELIABLE_PRIMARY_FISH")
        result = generate_fish_cutout(source, primary.box)
        raw_mask = result.mask.astype(bool)
        detector = {
            "model": detector_run.model_version,
            "detections_count": len(detector_run.detections),
            "primary_bbox_normalized": [primary.box.normalized().x1, primary.box.normalized().y1, primary.box.normalized().x2, primary.box.normalized().y2],
            "primary_confidence": round(float(primary.confidence), 6),
            "assessment": assessment.status.value,
            "primary_selection": "confidence × sqrt(area)",
        }
        test_id_uri = _persist(test_id, "01_original_image.png", original, "image/png")
        detector_uri = _persist(test_id, "02_detector_metadata.json", json.dumps(detector).encode(), "application/json")
        raw_mask_uri = _persist(test_id, "03_sam_raw_mask.png", _mask_bytes(raw_mask), "image/png")
        raw_transparent = result.cutout_png
        raw_transparent_uri = _persist(test_id, "04_sam_transparent_raw.png", raw_transparent, "image/png")
        state = {
            "report_version": LAB_VERSION,
            "runtime": _runtime(test_id),
            "input": {"image_id": hashlib.sha256(data).hexdigest()[:16], "filename": file.filename or "uploaded", "case_label": case_label[:200], "width": source.width, "height": source.height, "orientation": "landscape" if source.width >= source.height else "portrait", "size_bytes": len(data)},
            "detector": detector,
            "segmentation": {"model": "SAM_VIT_B", "quality": result.quality.value, "mask_area_ratio": round(result.mask_area_ratio, 6), "edge_ratio": round(result.edge_ratio, 6), "connected_components": result.connected_components},
            "mask_refinement": {"formula": "(raw_sam OR visible_add) AND NOT remove"},
            "completion": {"engine": "PowerPaintCompletionEngine", "model_type": os.getenv("FISH_COMPLETION_MODEL_TYPE", "powerpaint"), "model_version": os.getenv("FISH_COMPLETION_MODEL_VERSION") or None, "model_uri": os.getenv("FISH_COMPLETION_MODEL_URI") or None, "species_condition": "OFF", "fixed_prompt_id": "FIXED_FISH_COMPLETION_V0.1", "generation_count": 0, "retry_count": 0, "status": "NOT_RUN"},
            "composition": {"observed_pixel_change_ratio": None, "final_visible_pixels": None, "generated_pixels": None, "formula": "Refined Visible + Generated Completion"},
            "preview": {"asset": "A_SMART_CROP_B_SAM_RAW_C_REFINED_VISIBLE_D_AI_COMPLETED", "display_transform": "SHARED"},
            "human_review": {},
            "cost": {"gpu_active_seconds": None, "estimated_compute_cost_usd": None, "cost_reason": "PRICING_NOT_CONFIGURED"},
            "errors": {},
            "assets": {"original": test_id_uri, "detector_metadata": detector_uri, "sam_raw_mask": raw_mask_uri, "sam_transparent": raw_transparent_uri},
        }
        _save_state(test_id, state)
        return {**state, "test_id": test_id, "original": _data_url(original, "image/png"), "sam_raw_mask": _data_url(_mask_bytes(raw_mask), "image/png"), "sam_transparent": _data_url(raw_transparent, "image/png")}
    finally:
        source.close()


@router.post("/api/debug/fish-completion-lab/masks")
async def save_masks(payload: MaskPayload):
    state = _load_state(payload.test_id)
    width, height = state["input"]["width"], state["input"]["height"]
    masks = {name: _decode_mask(payload.masks.get(name, ""), width, height) if payload.masks.get(name) else np.zeros((height, width), dtype=bool) for name in ("visible_add", "remove", "occluder", "completion_canonical")}
    raw_mask = np.asarray(Image.open(io.BytesIO(_read_persist(state["assets"]["sam_raw_mask"]))).convert("L")) > 127
    stats = _stats(raw_mask, masks["visible_add"], masks["remove"], masks["occluder"], masks["completion_canonical"])
    if not stats["completion_mask_valid"]:
        raise HTTPException(422, {"error_code": "COMPLETION_MASK_NOT_SUBSET_OF_OCCLUDER", "illegal_pixels": stats["illegal_completion_pixels"]})
    original = Image.open(io.BytesIO(_read_persist(state["assets"]["original"]))).convert("RGB")
    refined = (raw_mask | masks["visible_add"]) & ~masks["remove"]
    refined_png = _png(Image.fromarray(np.where(refined, 255, 0).astype("uint8"), "L"))
    refined_fish = _png(Image.fromarray(np.dstack([np.asarray(original), np.where(refined, 255, 0).astype("uint8")]), "RGBA"))
    for name, mask in (("05_visible_add_mask.png", masks["visible_add"]), ("06_remove_mask.png", masks["remove"]), ("07_refined_visible_mask.png", refined), ("09_occluder_mask.png", masks["occluder"]), ("10_completion_mask_canonical.png", masks["completion_canonical"])):
        _persist(payload.test_id, name, _mask_bytes(mask), "image/png")
    _persist(payload.test_id, "08_refined_visible_fish.png", refined_fish, "image/png")
    state["mask_refinement"].update(stats)
    state["occlusion"] = {"occluder_area_pixels": stats["occluder_area_pixels"], "occluder_region_count": stats["occluder_region_count"]}
    state["completion_mask"] = {k: stats[k] for k in ("completion_area_pixels", "completion_region_count", "estimated_final_fish_area_pixels", "generated_pixel_ratio", "completion_level", "eligible_for_v0_1", "eligibility_reason", "completion_mask_valid")}
    state["assets"].update({"refined_visible": f"{PREFIX}/{payload.test_id}/08_refined_visible_fish.png"})
    _save_state(payload.test_id, state)
    return {"test_id": payload.test_id, "statistics": stats, "refined_visible": _data_url(refined_fish, "image/png")}


@router.post("/api/debug/fish-completion-lab/run")
def run_completion(payload: RunPayload):
    state = _load_state(payload.test_id)
    if not state.get("completion_mask", {}).get("eligible_for_v0_1"):
        raise HTTPException(422, {"error_code": "COMPLETION_NOT_ELIGIBLE", "message": "请先提交合法且不超过 20% 的 Completion Mask"})
    if not os.getenv("FISH_COMPLETION_ENABLED", "").strip().lower() == "true":
        state["completion"]["status"] = "COMPLETION_UNAVAILABLE"
        state["errors"]["completion"] = {"error_code": "COMPLETION_WORKER_UNAVAILABLE", "message": "FISH_COMPLETION_ENABLED=true 且真实 PowerPaint Worker 未配置"}
        _save_state(payload.test_id, state)
        raise HTTPException(503, state["errors"]["completion"])
    raise HTTPException(503, {"error_code": "COMPLETION_WORKER_NOT_IMPLEMENTED", "message": "真实 PowerPaint Worker 尚未配置；禁止 Mock Completion"})


@router.post("/api/debug/fish-completion-lab/review")
def save_review(payload: ReviewPayload):
    state = _load_state(payload.test_id)
    state["human_review"] = payload.review
    state["status"] = payload.review.get("status", "IN_REVIEW")
    _save_json(payload.test_id, "15_review.json", payload.review)
    _save_state(payload.test_id, state)
    return {"status": "saved", "test_id": payload.test_id}


@router.get("/api/debug/fish-completion-lab/report/{test_id}")
def report(test_id: str):
    return _load_state(test_id)
