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
import logging
import os
import secrets
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from google.cloud import storage
from PIL import Image
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.db import get_db
from app.dataset_models import DatasetItem
from app.detector_runtime import detect, normalize_android_source
from app.recognition_pipeline import assess_detections
from app.models import DatasetVersion
from app.segmentation.service import generate_fish_cutout
from app.completion_worker_client import (
    CompletionWorkerError,
    check_completion_worker,
    invoke_completion_worker,
)
from app.completion_decision import AUTO_COMPLETION, MANUAL_DEBUG, decide_completion

router = APIRouter(tags=["fish-completion-lab"])
templates = Jinja2Templates(directory="app/templates")
LAB_VERSION = "YUJIAN_FISH_COMPLETION_LAB_V0.3"
MAX_BYTES = 25 * 1024 * 1024
PREFIX = "experiments/fish_completion_lab/v0.3"
logger = logging.getLogger(__name__)


def _json_error(status_code: int, error_code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error_code": error_code, "message": message})



def _progress(state: dict[str, Any]) -> list[dict[str, Any]]:
    timings = state.get("timings", {})
    detector = state.get("detector", {})
    completion = state.get("completion", {})
    assets = state.get("assets", {})
    composition = state.get("composition", {})
    return [
        {"stage": "input", "label": "图片输入", "status": "READY", "elapsed_ms": timings.get("input_decode_ms"), "result": state.get("input", {}).get("filename")},
        {"stage": "detector", "label": "Detector / BBox", "status": "READY" if detector else "PENDING", "elapsed_ms": timings.get("detector_ms"), "result": {"bbox_pixels": detector.get("bbox_pixels"), "bbox_normalized": detector.get("primary_bbox_normalized"), "confidence": detector.get("primary_confidence"), "area_ratio": detector.get("bbox_area_ratio")}},
        {"stage": "sam", "label": "SAM 分割", "status": "READY" if state.get("segmentation") else "PENDING", "elapsed_ms": timings.get("sam_ms"), "result": state.get("segmentation", {}).get("quality")},
        {"stage": "mask", "label": "Mask 标注", "status": "READY" if state.get("completion_mask") else "PENDING", "elapsed_ms": timings.get("mask_edit_ms"), "result": state.get("completion_mask", {}).get("completion_area_pixels")},
        {"stage": "roi", "label": "ROI 提取", "status": "READY" if assets.get("completion_roi") else "PENDING", "elapsed_ms": timings.get("roi_ms"), "result": state.get("roi")},
        {"stage": "powerpaint", "label": "PowerPaint", "status": completion.get("worker_status") or completion.get("status", "PENDING"), "elapsed_ms": timings.get("worker_ms"), "result": completion.get("model_version")},
        {"stage": "compose", "label": "Protected Compose", "status": "READY" if assets.get("final_asset") else "PENDING", "elapsed_ms": timings.get("compose_ms"), "result": {"visible_pixel_change_ratio": composition.get("visible_pixel_change_ratio"), "generated_pixels": composition.get("generated_pixels")}},
    ]


class MaskPayload(BaseModel):
    test_id: str = Field(min_length=1, max_length=80)
    masks: dict[str, str] = Field(default_factory=dict)


class RunPayload(BaseModel):
    test_id: str = Field(min_length=1, max_length=80)
    completion_mode: str = Field(default=AUTO_COMPLETION, max_length=20)


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
    with open(uri.removeprefix("local://"), "rb") as handle:
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

def _apply_masks(test_id: str, state: dict[str, Any], masks: dict[str, np.ndarray], *, source: str) -> tuple[dict[str, Any], bytes]:
    """Persist masks and update state for automatic or manual execution."""
    mask_started = time.perf_counter()
    width, height = state["input"]["width"], state["input"]["height"]
    raw_mask = np.asarray(Image.open(io.BytesIO(_read_persist(state["assets"]["sam_raw_mask"]))).convert("L")) > 127
    if raw_mask.shape != (height, width):
        raise HTTPException(500, "SAM mask dimensions do not match input")
    zeros = np.zeros((height, width), dtype=bool)
    stats = _stats(raw_mask, masks.get("visible_add", zeros), masks.get("remove", zeros), masks.get("occluder", zeros), masks.get("completion_canonical", zeros))
    if not stats["completion_mask_valid"]:
        raise HTTPException(422, {"error_code": "COMPLETION_MASK_NOT_SUBSET_OF_OCCLUDER", "illegal_pixels": stats["illegal_completion_pixels"]})
    original = Image.open(io.BytesIO(_read_persist(state["assets"]["original"]))).convert("RGB")
    refined = (raw_mask | masks.get("visible_add", zeros)) & ~masks.get("remove", zeros)
    refined_fish = _png(Image.fromarray(np.dstack([np.asarray(original), np.where(refined, 255, 0).astype("uint8")]), "RGBA"))
    asset_uris = {}
    for name, mask in (
        ("05_visible_add_mask.png", masks.get("visible_add", zeros)),
        ("06_remove_mask.png", masks.get("remove", zeros)),
        ("07_refined_visible_mask.png", refined),
        ("09_occluder_mask.png", masks.get("occluder", zeros)),
        ("10_completion_mask.png", masks.get("completion_canonical", zeros)),
    ):
        asset_uris[name] = _persist(test_id, name, _mask_bytes(mask), "image/png")
    asset_uris["refined_visible"] = _persist(test_id, "08_refined_visible_fish.png", refined_fish, "image/png")
    state.setdefault("timings", {})["mask_generation_ms" if source == "AUTO" else "mask_edit_ms"] = round((time.perf_counter() - mask_started) * 1000, 2)
    state["mask_refinement"].update(stats)
    state["occlusion"] = {"occluder_area_pixels": stats["occluder_area_pixels"], "occluder_region_count": stats["occluder_region_count"]}
    state["completion_mask"] = {key: stats[key] for key in ("completion_area_pixels", "completion_region_count", "estimated_final_fish_area_pixels", "generated_pixel_ratio", "completion_level", "eligible_for_v0_1", "eligibility_reason", "completion_mask_valid")}
    state["visible_fish_mask"] = asset_uris["07_refined_visible_mask.png"]
    state["assets"].update({"visible_add": asset_uris["05_visible_add_mask.png"], "remove": asset_uris["06_remove_mask.png"], "refined_visible_mask": asset_uris["07_refined_visible_mask.png"], "occluder": asset_uris["09_occluder_mask.png"], "occluder_mask": asset_uris["09_occluder_mask.png"], "completion_mask": asset_uris["10_completion_mask.png"], "completion_mask_canonical": asset_uris["10_completion_mask.png"], "refined_visible": asset_uris["refined_visible"]})
    state["completion_decision"]["source"] = source
    state["completion_decision"]["mask_generated"] = bool(masks.get("completion_canonical", zeros).any())
    state["progress"] = _progress(state)
    return stats, refined_fish



@router.get("/debug/fish-completion-lab-v02", response_class=HTMLResponse)
@router.get("/debug/fish-completion-lab", response_class=HTMLResponse)
def fish_completion_lab_page(request: Request):
    return templates.TemplateResponse(request=request, name="fish_completion_lab.html", context={})


@router.get("/api/debug/fish-completion-lab-v02/datasets")
@router.get("/api/debug/fish-completion-lab/datasets")
def completion_datasets(db=Depends(get_db)):
    rows = db.scalars(select(DatasetVersion).where(DatasetVersion.status == "FROZEN").order_by(DatasetVersion.created_at.desc())).all()
    return [{"dataset_version": row.dataset_version, "status": row.status, "pipeline_type": getattr(row, "pipeline_type", "WHOLE_IMAGE_V1"), "image_count": row.train_count + row.val_count + row.test_count, "train_count": row.train_count, "val_count": row.val_count, "test_count": row.test_count, "species_count": row.species_count} for row in rows]


@router.get("/api/debug/fish-completion-lab-v02/datasets/{dataset_version}/images")
@router.get("/api/debug/fish-completion-lab/datasets/{dataset_version}/images")
def completion_dataset_images(dataset_version: str, split: str | None = None, species: str | None = None, limit: int = Query(default=60, ge=1, le=200), offset: int = Query(default=0, ge=0), db=Depends(get_db)):
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
    return [{"dataset_version": row.dataset_version, "dataset_item_id": row.id, "batch_id": row.batch_id, "image_id": row.image_id, "species": row.species_name, "species_key": row.species_key, "split": row.split, "gcs_uri": row.gcs_uri, "preview_url": f"/media/{row.batch_id}/{row.image_id}"} for row in rows]


def _read_dataset_image(item: DatasetItem) -> bytes:
    uri = (item.gcs_uri or "").strip()
    if not uri.startswith("gs://") or "/" not in uri[5:]:
        raise HTTPException(status_code=422, detail="DATASET_IMAGE_URI_INVALID")
    bucket_name, object_name = uri[5:].split("/", 1)
    try:
        return storage.Client().bucket(bucket_name).blob(object_name).download_as_bytes(timeout=120)
    except Exception as exc:
        logger.exception("Dataset image read failed; dataset_item_id=%s", item.id)
        raise HTTPException(status_code=502, detail="DATASET_IMAGE_READ_FAILED") from exc


@router.post("/api/debug/fish-completion-lab/prepare")
async def prepare(file: UploadFile | None = File(default=None), case_label: str = Form(default=""), source_type: str = Form(default="local_upload"), dataset_version: str = Form(default=""), dataset_item_id: str = Form(default=""), db=Depends(get_db)):
    prepare_started = time.perf_counter()
    test_id = "FCL_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(2)
    source = None
    try:
        if source_type == "dataset_freeze" or dataset_item_id:
            if not dataset_version or not dataset_item_id:
                raise HTTPException(status_code=400, detail="DATASET_SELECTION_REQUIRED")
            dataset = db.get(DatasetVersion, dataset_version)
            if not dataset or dataset.status != "FROZEN":
                raise HTTPException(status_code=409, detail="DATASET_VERSION_NOT_FROZEN")
            try:
                item_id = int(dataset_item_id)
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail="DATASET_ITEM_ID_INVALID") from exc
            dataset_item = db.scalar(select(DatasetItem).where(DatasetItem.dataset_version == dataset_version, DatasetItem.id == item_id))
            if not dataset_item:
                raise HTTPException(status_code=404, detail="DATASET_ITEM_NOT_FOUND")
            data = _read_dataset_image(dataset_item)
            input_filename = dataset_item.image_id or f"dataset-{dataset_item.id}"
        else:
            if file is None:
                raise HTTPException(status_code=400, detail="IMAGE_UPLOAD_REQUIRED")
            data = await file.read(MAX_BYTES + 1)
            input_filename = file.filename or "uploaded"
        if not data or len(data) > MAX_BYTES:
            return _json_error(400, "INVALID_IMAGE_UPLOAD", "图片为空或超过 25 MiB")
        decode_started = time.perf_counter()
        with Image.open(io.BytesIO(data)) as uploaded:
            source = normalize_android_source(uploaded)
        input_decode_ms = round((time.perf_counter() - decode_started) * 1000, 2)
        original = _png(source.convert("RGB"))
        detector_started = time.perf_counter()
        detector_run = detect(source)
        assessment = assess_detections(detector_run.detections)
        detector_ms = round((time.perf_counter() - detector_started) * 1000, 2)
        primary = assessment.primary
        if primary is None:
            raise HTTPException(422, "NO_RELIABLE_PRIMARY_FISH")
        sam_started = time.perf_counter()
        result = generate_fish_cutout(source, primary.box)
        sam_ms = round((time.perf_counter() - sam_started) * 1000, 2)
        raw_mask = result.mask.astype(bool)
        normalized = primary.box.normalized()
        bbox_normalized = [normalized.x1, normalized.y1, normalized.x2, normalized.y2]
        bbox_pixels = [round(normalized.x1 * source.width), round(normalized.y1 * source.height), round(normalized.x2 * source.width), round(normalized.y2 * source.height)]
        bbox_area_ratio = round(max(0, bbox_pixels[2] - bbox_pixels[0]) * max(0, bbox_pixels[3] - bbox_pixels[1]) / max(1, source.width * source.height), 6)
        detector = {
            "model": detector_run.model_version,
            "detections_count": len(detector_run.detections),
            "primary_bbox_normalized": bbox_normalized,
            "bbox_pixels": bbox_pixels,
            "bbox_area_ratio": bbox_area_ratio,
            "primary_confidence": round(float(primary.confidence), 6),
            "assessment": assessment.status.value,
            "primary_selection": "confidence × sqrt(area)",
        }
        test_id_uri = _persist(test_id, "01_original_image.png", original, "image/png")
        detector_uri = _persist(test_id, "02_detector_metadata.json", json.dumps(detector).encode(), "application/json")
        raw_mask_uri = _persist(test_id, "03_sam_raw_mask.png", _mask_bytes(raw_mask), "image/png")
        raw_transparent = result.cutout_png
        raw_transparent_uri = _persist(test_id, "04_sam_transparent_raw.png", raw_transparent, "image/png")
        decision = decide_completion(raw_mask, tuple(bbox_pixels), (source.height, source.width), case_label=case_label, segmentation_quality=result.quality.value)
        state = {
            "report_version": LAB_VERSION,
            "runtime": _runtime(test_id),
            "input": {"image_id": hashlib.sha256(data).hexdigest()[:16], "filename": input_filename, "case_label": case_label[:200], "source_type": source_type[:32], "dataset_version": dataset_version[:128] or None, "dataset_item_id": dataset_item_id[:80] or None, "width": source.width, "height": source.height, "orientation": "landscape" if source.width >= source.height else "portrait", "size_bytes": len(data)},
            "detector": detector,
            "segmentation": {"model": "SAM_VIT_B", "quality": result.quality.value, "mask_area_ratio": round(result.mask_area_ratio, 6), "edge_ratio": round(result.edge_ratio, 6), "connected_components": result.connected_components},
            "mask_refinement": {"formula": "(raw_sam OR visible_add) AND NOT remove"},
            "completion_mode": AUTO_COMPLETION,
            "completion_decision": decision.as_dict(),
            "completion": {"engine": "PowerPaintCompletionEngine", "model_type": os.getenv("FISH_COMPLETION_MODEL_TYPE", "powerpaint"), "model_version": os.getenv("FISH_COMPLETION_MODEL_VERSION") or None, "model_uri": os.getenv("FISH_COMPLETION_MODEL_URI") or None, "species_condition": "OFF", "fixed_prompt_id": "FIXED_FISH_COMPLETION_V0.1", "generation_count": 0, "retry_count": 0, "status": "NOT_REQUIRED" if not decision.completion_required else "MASK_READY"},
            "composition": {"observed_pixel_change_ratio": None, "final_visible_pixels": None, "generated_pixels": None, "formula": "Refined Visible + Generated Completion"},
            "preview": {"asset": "A_SMART_CROP_B_SAM_RAW_C_REFINED_VISIBLE_CANDIDATE_D_AI_COMPLETED", "display_transform": "SHARED"},
            "human_review": {},
            "cost": {"gpu_active_seconds": None, "estimated_compute_cost_usd": None, "cost_reason": "PRICING_NOT_CONFIGURED"},
            "errors": {},
            "assets": {"original": test_id_uri, "detector_metadata": detector_uri, "sam_raw_mask": raw_mask_uri, "sam_transparent": raw_transparent_uri},
            "timings": {"input_decode_ms": input_decode_ms, "detector_ms": detector_ms, "sam_ms": sam_ms, "mask_edit_ms": None, "mask_generation_ms": None, "roi_ms": None, "worker_ms": None, "compose_ms": None, "prepare_total_ms": None},
        }
        auto_masks = {"visible_add": np.zeros_like(raw_mask, dtype=bool), "remove": np.zeros_like(raw_mask, dtype=bool), "occluder": decision.occluder_mask, "completion_canonical": decision.completion_mask}
        statistics, refined_fish = _apply_masks(test_id, state, auto_masks, source="AUTO")
        state["completion"]["status"] = "NOT_REQUIRED" if not decision.completion_required else "MASK_READY"
        state["timings"]["prepare_total_ms"] = round((time.perf_counter() - prepare_started) * 1000, 2)
        state["progress"] = _progress(state)
        _save_state(test_id, state)
        return {**state, "test_id": test_id, "statistics": statistics, "progress": state["progress"], "original": _data_url(original, "image/png"), "sam_raw_mask": _data_url(_mask_bytes(raw_mask), "image/png"), "sam_transparent": _data_url(raw_transparent, "image/png"), "refined_visible": _data_url(refined_fish, "image/png")}
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
        return JSONResponse(status_code=exc.status_code, content={
            "error_code": detail.get("error_code", "FISH_COMPLETION_REQUEST_FAILED"),
            "message": detail.get("message", str(detail.get("detail", "请求失败"))),
        })
    except Exception as exc:
        logger.exception("Fish Completion Lab prepare failed; test_id=%s", test_id)
        return _json_error(500, "FISH_COMPLETION_PREPARE_FAILED", f"{exc.__class__.__name__}: {exc}")
    finally:
        if source is not None:
            source.close()


@router.post("/api/debug/fish-completion-lab/masks")
async def save_masks(payload: MaskPayload):
    mask_started = time.perf_counter()
    state = _load_state(payload.test_id)
    state.setdefault("timings", {})
    width, height = state["input"]["width"], state["input"]["height"]
    completion_value = payload.masks.get("completion_mask") or payload.masks.get("completion_canonical")
    occluder_value = payload.masks.get("occluder_mask") or payload.masks.get("occluder")
    masks = {
        "visible_add": _decode_mask(payload.masks.get("visible_add", ""), width, height) if payload.masks.get("visible_add") else np.zeros((height, width), dtype=bool),
        "remove": _decode_mask(payload.masks.get("remove", ""), width, height) if payload.masks.get("remove") else np.zeros((height, width), dtype=bool),
        "occluder": _decode_mask(occluder_value, width, height) if occluder_value else np.zeros((height, width), dtype=bool),
        "completion_canonical": _decode_mask(completion_value, width, height) if completion_value else np.zeros((height, width), dtype=bool),
    }
    statistics, refined_fish = _apply_masks(payload.test_id, state, masks, source="MANUAL")
    state["completion_mode"] = MANUAL_DEBUG
    state["completion_decision"] = {**state.get("completion_decision", {}), "mode": MANUAL_DEBUG, "source": "MANUAL", "status": "MASK_READY", "completion_required": bool(statistics.get("completion_area_pixels")), "mask_generated": False}
    state["progress"] = _progress(state)
    _save_state(payload.test_id, state)
    return {"test_id": payload.test_id, "statistics": statistics, "timings": state["timings"], "progress": state["progress"], "refined_visible": _data_url(refined_fish, "image/png")}



def _lab_asset_uri(test_id: str, name: str) -> str:
    bucket = _bucket()
    object_name = f"{PREFIX}/{test_id}/{name}"
    return f"gs://{bucket.name}/{object_name}" if bucket else f"local://var/fish_completion_lab/{test_id}/{name}"


def _build_completion_roi(test_id: str, state: dict[str, Any]) -> tuple[str, str, Image.Image, np.ndarray, tuple[int, int, int, int]]:
    original = Image.open(io.BytesIO(_read_persist(state["assets"]["original"]))).convert("RGB")
    mask_uri = state["assets"].get("completion_mask") or _lab_asset_uri(test_id, "10_completion_mask.png")
    completion = np.asarray(Image.open(io.BytesIO(_read_persist(mask_uri))).convert("L")) > 127
    ys, xs = np.where(completion)
    if not len(xs):
        raise HTTPException(422, {"error_code": "COMPLETION_MASK_EMPTY", "message": "Completion mask is empty"})
    height, width = completion.shape
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    padding_ratio = min(1.0, max(0.5, float(os.getenv("FISH_COMPLETION_ROI_PADDING_RATIO", "0.75"))))
    pad_x, pad_y = max(8, int((x2 - x1) * padding_ratio)), max(8, int((y2 - y1) * padding_ratio))
    x1, y1, x2, y2 = max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y)
    roi = original.crop((x1, y1, x2, y2))
    roi_mask = Image.fromarray(np.where(completion[y1:y2, x1:x2], 255, 0).astype("uint8"), "L")
    longest = max(roi.size)
    if longest > 512:
        scale = 512 / longest
        size = (max(8, int(roi.width * scale) // 8 * 8), max(8, int(roi.height * scale) // 8 * 8))
        roi, roi_mask = roi.resize(size, Image.Resampling.LANCZOS), roi_mask.resize(size, Image.Resampling.NEAREST)
    roi_uri = _persist(test_id, "08_completion_roi.png", _png(roi), "image/png")
    roi_mask_uri = _persist(test_id, "08_completion_roi_mask.png", _png(roi_mask), "image/png")
    return roi_uri, roi_mask_uri, roi, np.asarray(roi_mask) > 127, (x1, y1, x2, y2)


def _compose_completion(state: dict[str, Any], generated: Image.Image, box: tuple[int, int, int, int]) -> tuple[bytes, float]:
    original = Image.open(io.BytesIO(_read_persist(state["assets"]["original"]))).convert("RGB")
    canonical_uri = state["assets"].get("completion_mask") or _lab_asset_uri(state["runtime"]["test_id"], "10_completion_mask.png")
    canonical = np.asarray(Image.open(io.BytesIO(_read_persist(canonical_uri))).convert("L")) > 127
    refined_uri = state["assets"].get("refined_visible")
    refined = np.asarray(Image.open(io.BytesIO(_read_persist(refined_uri))).convert("RGBA"))[:, :, 3] > 127 if refined_uri else np.zeros(canonical.shape, dtype=bool)
    generated = generated.convert("RGB")
    x1, y1, x2, y2 = box
    if generated.size != (x2 - x1, y2 - y1):
        generated = generated.resize((x2 - x1, y2 - y1), Image.Resampling.LANCZOS)
    canvas = np.asarray(original).copy()
    generated_pixels = np.asarray(generated)
    target_mask = canonical[y1:y2, x1:x2]
    canvas[y1:y2, x1:x2][target_mask] = generated_pixels[target_mask]
    alpha = np.where(refined | canonical, 255, 0).astype("uint8")
    rgba = np.dstack([canvas, alpha])
    changed = np.any(canvas != np.asarray(original), axis=2)
    observed = refined & ~canonical
    ratio = float((changed & observed).sum() / max(1, observed.sum()))
    return _png(Image.fromarray(rgba, "RGBA")), ratio


@router.post("/api/debug/fish-completion-lab/run")
def run_completion(payload: RunPayload):
    started = time.perf_counter()
    requested_mode = (payload.completion_mode or AUTO_COMPLETION).strip().upper()
    if requested_mode == "MANUAL":
        requested_mode = MANUAL_DEBUG
    if requested_mode not in (AUTO_COMPLETION, MANUAL_DEBUG):
        raise HTTPException(400, {"error_code": "COMPLETION_MODE_UNSUPPORTED", "message": "completion_mode 必须是 AUTO_COMPLETION 或 MANUAL_DEBUG"})
    state = _load_state(payload.test_id)
    state.setdefault("timings", {})
    state["completion_mode"] = requested_mode
    decision = state.get("completion_decision", {})
    if requested_mode == AUTO_COMPLETION and decision.get("status") == "NOT_ELIGIBLE":
        raise HTTPException(422, {"error_code": "COMPLETION_NOT_ELIGIBLE", "message": "可见鱼体不足 50%，禁止自动补全"})
    mask_state = state.get("completion_mask", {})
    if not mask_state.get("eligible_for_v0_1"):
        raise HTTPException(422, {"error_code": "COMPLETION_NOT_ELIGIBLE", "message": "Completion Mask 不满足当前安全阈值"})
    if not decision.get("completion_required", mask_state.get("completion_area_pixels", 0) > 0) or mask_state.get("completion_area_pixels", 0) == 0:
        state["completion"].update({"status": "NOT_REQUIRED", "generation_count": 0, "retry_count": 0})
        state["composition"]["total_processing_ms"] = round((time.perf_counter() - started) * 1000, 2)
        state["timings"]["total_ms"] = state["composition"]["total_processing_ms"]
        state["progress"] = _progress(state)
        _save_state(payload.test_id, state)
        return {"test_id": payload.test_id, "status": "COMPLETION_NOT_REQUIRED", "timings": state["timings"], "progress": state["progress"], "report": state}
    if os.getenv("FISH_COMPLETION_ENABLED", "").strip().lower() == "false" or not os.getenv("FISH_COMPLETION_WORKER_URL", "").strip():
        error = {"error_code": "COMPLETION_WORKER_UNAVAILABLE", "message": "真实 PowerPaint Worker 未配置；请设置 FISH_COMPLETION_WORKER_URL"}
        state["completion"]["status"] = "COMPLETION_UNAVAILABLE"
        state["errors"]["completion"] = error
        _save_state(payload.test_id, state)
        raise HTTPException(503, error)
    try:
        roi_started = time.perf_counter()
        roi_uri, roi_mask_uri, roi, roi_mask, box = _build_completion_roi(payload.test_id, state)
        state["timings"]["roi_ms"] = round((time.perf_counter() - roi_started) * 1000, 2)
        state["roi"] = {"bbox_pixels": list(box), "original_size": [state["input"]["width"], state["input"]["height"]], "roi_size": list(roi.size)}
        worker_started = time.perf_counter()
        worker = invoke_completion_worker(
            image_uri=roi_uri,
            mask_uri=roi_mask_uri,
            prompt=(
                "Complete the missing part of the fish body. Preserve original species, "
                "body shape, scales, fins and natural texture. Do not modify visible fish pixels."
            ),
            task="fish_completion",
        )
        state["timings"]["worker_ms"] = round((time.perf_counter() - worker_started) * 1000, 2)
        generated_ref = worker.get("result_uri")
        if generated_ref and generated_ref.startswith("data:"):
            generated_bytes = base64.b64decode(generated_ref.split(",", 1)[1])
        elif (worker.get("generated_roi") or "").startswith("data:"):
            generated_bytes = base64.b64decode(worker["generated_roi"].split(",", 1)[1])
        else:
            generated_bytes = _read_persist(generated_ref)
        if hashlib.sha256(generated_bytes).digest() == hashlib.sha256(_png(roi)).digest():
            raise CompletionWorkerError("WORKER_RETURNED_INPUT", "Worker output is byte-identical to the ROI input")
        generated = Image.open(io.BytesIO(generated_bytes)).convert("RGB")
        compose_started = time.perf_counter()
        final_png, observed_change_ratio = _compose_completion(state, generated, box)
        state["timings"]["compose_ms"] = round((time.perf_counter() - compose_started) * 1000, 2)
        generated_uri = _persist(payload.test_id, "09_generated_roi.png", generated_bytes, "image/png")
        final_uri = _persist(payload.test_id, "10_final_asset.png", final_png, "image/png")
        total_ms = round((time.perf_counter() - started) * 1000, 2)
        state["completion"].update({"status": "WORKER_EXECUTED", "worker_status": "WORKER_EXECUTED", "model_version": worker["model_version"], "generation_count": 1, "retry_count": 0, "inference_time_ms": worker["inference_time_ms"], "gpu_info": worker.get("gpu_info")})
        state["composition"].update({"observed_pixel_change_ratio": observed_change_ratio, "visible_pixel_change_ratio": observed_change_ratio, "visible_changed_pixels": 0 if observed_change_ratio == 0 else None, "generated_pixels": mask_state["completion_area_pixels"], "total_processing_ms": total_ms})
        state["timings"]["total_ms"] = total_ms
        state["assets"].update({"completion_roi": roi_uri, "completion_roi_mask": roi_mask_uri, "generated_roi": generated_uri, "final_asset": final_uri})
        state["progress"] = _progress(state)
        _save_json(payload.test_id, "12_test_report.json", state)
        _save_state(payload.test_id, state)
        return {"test_id": payload.test_id, "status": "WORKER_EXECUTED", "worker_status": "WORKER_EXECUTED", "model_version": worker["model_version"], "processing_ms": total_ms, "timings": state["timings"], "progress": state["progress"], "generated_roi": _data_url(generated_bytes, "image/png"), "final_asset": _data_url(final_png, "image/png"), "report": state}
    except CompletionWorkerError as exc:
        error = {"error_code": exc.error_code, "message": str(exc), "status_code": exc.status_code}
    except Exception as exc:
        error = {"error_code": "COMPLETION_PIPELINE_FAILED", "message": f"{exc.__class__.__name__}: {exc}"}
    state["completion"]["status"] = "COMPLETION_FAILED"
    state["errors"]["completion"] = error
    state["composition"]["total_processing_ms"] = round((time.perf_counter() - started) * 1000, 2)
    state["timings"]["total_ms"] = state["composition"]["total_processing_ms"]
    state["progress"] = _progress(state)
    _save_state(payload.test_id, state)
    raise HTTPException(503, error)


@router.get("/api/debug/fish-completion-lab-v02/worker-status")
@router.get("/api/debug/fish-completion-lab/worker-status")
def worker_status():
    try:
        return check_completion_worker()
    except CompletionWorkerError as exc:
        return JSONResponse(
            status_code=503,
            content={
                "endpoint_configured": True,
                "status": "WORKER_UNAVAILABLE",
                "error_code": exc.error_code,
                "message": str(exc),
            },
        )


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