"""Debug-only fish segmentation and Fish Hero preview API."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from app.detector_runtime import detect, normalize_android_source
from app.recognition_pipeline import assess_detections
from app.segmentation.mask_generator import SegmentationModelNotConfigured
from app.segmentation.service import generate_fish_cutout

MAX_DEBUG_IMAGE_BYTES = 25 * 1024 * 1024
DEMO_VERSION = "FISH_HERO_PREVIEW_DEMO_v0.2-A"
CHECKPOINT_LABEL = "sam_vit_b_01ec64"
router = APIRouter(tags=["fish-segmentation-demo"])
templates = Jinja2Templates(directory="app/templates")


class FishHeroReviewRequest(BaseModel):
    test_id: str = Field(min_length=1, max_length=80)
    image_id: str = Field(min_length=1, max_length=128)
    timestamp: str = Field(min_length=1, max_length=80)
    detector_summary: dict[str, Any] = Field(default_factory=dict)
    segmentation_summary: dict[str, Any] = Field(default_factory=dict)
    eligibility: dict[str, Any] = Field(default_factory=dict)
    human: dict[str, Any] = Field(default_factory=dict)


async def _read_image(file: UploadFile) -> bytes:
    if file.content_type and file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="仅支持 JPG、PNG、WEBP 图片")
    data = await file.read(MAX_DEBUG_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="请选择图片")
    if len(data) > MAX_DEBUG_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="图片不能超过 25 MiB")
    return data


def _data_url(content: bytes, media_type: str) -> str:
    return f"data:{media_type};base64," + base64.b64encode(content).decode("ascii")


def _png_data_url(image: Image.Image) -> str:
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return _data_url(output.getvalue(), "image/png")


def _mask_overlay(source: Image.Image, mask: Any, bbox: Any) -> str:
    image = source.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (13, 148, 136, 0))
    alpha = Image.fromarray((mask.astype("uint8") * 120), mode="L")
    overlay.putalpha(alpha)
    image.alpha_composite(overlay)
    draw = ImageDraw.Draw(image)
    b = bbox.normalized()
    draw.rectangle(
        (
            round(b.x1 * image.width),
            round(b.y1 * image.height),
            round(b.x2 * image.width),
            round(b.y2 * image.height),
        ),
        outline="#dc2626",
        width=max(3, image.width // 500),
    )
    return _png_data_url(image)


def _expanded_crop_box(bbox: Any, width: int, height: int, padding_ratio: float = 0.12) -> tuple[int, int, int, int]:
    b = bbox.normalized()
    box_width = max(1.0, (b.x2 - b.x1) * width)
    box_height = max(1.0, (b.y2 - b.y1) * height)
    left = max(0, int((b.x1 * width) - box_width * padding_ratio))
    top = max(0, int((b.y1 * height) - box_height * padding_ratio))
    right = min(width, int((b.x2 * width) + box_width * padding_ratio + 0.999))
    bottom = min(height, int((b.y2 * height) + box_height * padding_ratio + 0.999))
    return left, top, max(left + 1, right), max(top + 1, bottom)


def _smart_crop(source: Image.Image, bbox: Any) -> tuple[bytes, tuple[float, float, float, float]]:
    left, top, right, bottom = _expanded_crop_box(bbox, source.width, source.height)
    crop = source.crop((left, top, right, bottom)).convert("RGB")
    return _png_bytes(crop), (
        left / source.width,
        top / source.height,
        right / source.width,
        bottom / source.height,
    )


def _png_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _runtime() -> dict[str, str]:
    return {
        "service": os.getenv("K_SERVICE", "yujian-fish-segmentation-demo"),
        "revision": os.getenv("K_REVISION", "unknown"),
        "commit": os.getenv("APP_GIT_COMMIT", "unknown"),
        "demo_version": DEMO_VERSION,
        "detector_model": "DET_FISH_v0.1",
        "segmentation_model": "SAM_VIT_B",
        "segmentation_checkpoint": CHECKPOINT_LABEL,
        "hero_gate": "DISABLED_IN_V0.2-A",
    }


def _orientation(source: Image.Image) -> str:
    return "landscape" if source.width >= source.height else "portrait"


def _review_path() -> Path:
    path = Path("var/fish_hero_demo/reviews.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@router.get("/debug/fish-segmentation", response_class=HTMLResponse)
def fish_segmentation_page(request: Request):
    return templates.TemplateResponse(request=request, name="fish_segmentation.html", context={})


@router.post("/api/debug/fish-segmentation")
async def fish_segmentation(file: UploadFile = File(..., alias="image")) -> dict[str, Any]:
    data = await _read_image(file)
    source = None
    image_id = hashlib.sha256(data).hexdigest()[:16]
    try:
        with Image.open(io.BytesIO(data)) as uploaded:
            source = normalize_android_source(uploaded)
        detector_run = detect(source)
        assessment = assess_detections(detector_run.detections)
        assessment_name = assessment.status.value
        primary = assessment.primary
        eligible = assessment_name == "ready" and primary is not None
        eligibility_reason = (
            "ELIGIBLE_SINGLE_PRIMARY_FISH"
            if eligible
            else ("MULTIPLE_FISH" if assessment_name == "multiple_fish" else "NO_RELIABLE_PRIMARY_FISH")
        )
        response: dict[str, Any] = {
            "report_version": "YUJIAN_FISH_HERO_TEST_REPORT_V0.2-A",
            "runtime": _runtime(),
            "input": {
                "image_id": image_id,
                "filename": file.filename or "uploaded-image",
                "image_width": source.width,
                "image_height": source.height,
                "orientation": _orientation(source),
                "upload_size_bytes": len(data),
            },
            "detector": {
                "model": detector_run.model_version,
                "detections_count": len(detector_run.detections),
                "strong_detection_count": len(detector_run.detections),
                "primary_detection_found": primary is not None,
                "primary_confidence": round(float(primary.confidence), 6) if primary else None,
                "primary_bbox_normalized": (
                    [primary.box.normalized().x1, primary.box.normalized().y1, primary.box.normalized().x2, primary.box.normalized().y2]
                    if primary else None
                ),
                "primary_bbox_pixels": (
                    [round(primary.box.normalized().x1 * source.width), round(primary.box.normalized().y1 * source.height),
                     round(primary.box.normalized().x2 * source.width), round(primary.box.normalized().y2 * source.height)]
                    if primary else None
                ),
                "bbox_area_ratio": (
                    (primary.box.normalized().x2 - primary.box.normalized().x1)
                    * (primary.box.normalized().y2 - primary.box.normalized().y1)
                    if primary else None
                ),
                "bbox_touches_image_edge": (
                    primary.box.normalized().x1 <= 0.001 or primary.box.normalized().y1 <= 0.001
                    or primary.box.normalized().x2 >= 0.999 or primary.box.normalized().y2 >= 0.999
                    if primary else None
                ),
                "assessment": assessment_name,
                "reason": assessment.reason,
                "detector_processing_ms": detector_run.latency_ms,
                "primary_selection": "confidence × sqrt(area)",
            },
            "eligibility": {
                "single_fish_hero_evaluation": eligible,
                "eligibility_reason": eligibility_reason,
            },
            "hero_preview": {
                "original": "READY",
                "smart_crop": "UNAVAILABLE" if not primary else "READY",
                "transparent": "UNAVAILABLE",
            },
            "errors": {
                "detector_error": None,
                "segmentation_error": None,
                "preview_error": None,
                "review_save_error": None,
            },
        }
        response["original"] = _data_url(_png_bytes(source), "image/png")
        if primary is None:
            response["smart_crop_reason"] = "NO_RELIABLE_PRIMARY_FISH"
            response["transparent_fish"] = None
            response["mask_overlay"] = None
            response["original"] = _data_url(_png_bytes(source), "image/png")
            return response

        smart_crop, crop_box = _smart_crop(source, primary.box)
        response["smart_crop"] = _data_url(smart_crop, "image/png")
        response["smart_crop_meta"] = {
            "source": "PRIMARY_DETECTION",
            "crop_normalized": list(crop_box),
            "crop_padding_ratio": 0.12,
            "foreground_fit": "CONTAIN",
            "background_mode": "ORIGINAL_BLUR",
        }
        result = generate_fish_cutout(source, primary.box)
        response["segmentation"] = {
            "executed": True,
            "model": "SAM_VIT_B",
            "prompt_type": "DETECTOR_PRIMARY_BBOX",
            "quality": result.quality.value,
            "quality_reason": result.reason,
            "mask_area_ratio": round(result.mask_area_ratio, 6),
            "edge_ratio": round(result.edge_ratio, 6),
            "connected_components": result.connected_components,
            "mask_width": int(result.mask.shape[1]),
            "mask_height": int(result.mask.shape[0]),
            "processing_ms": result.processing_ms,
        }
        response["hero_preview"]["transparent"] = "READY"
        response["transparent_fish"] = _data_url(result.cutout_png, "image/png")
        response["mask_overlay"] = _mask_overlay(source, result.mask, primary.box)
        response["transparent_hero_meta"] = {
            "transparent_source": "SAM_MASK",
            "hero_fit": "CONTAIN",
            "hero_background": "MORNING_LAKE_V1",
            "fish_rgb_modified": False,
            "fish_generated": False,
        }
        return response
    except HTTPException:
        raise
    except SegmentationModelNotConfigured as exc:
        raise HTTPException(status_code=503, detail={"error_code": "SEGMENTATION_MODEL_UNAVAILABLE", "message": str(exc)}) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"error_code": "SEGMENTATION_FAILED", "message": str(exc)}) from exc
    finally:
        if source is not None:
            source.close()


@router.post("/api/debug/fish-hero-review")
def save_fish_hero_review(payload: FishHeroReviewRequest) -> dict[str, Any]:
    path = _review_path()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload.model_dump(), ensure_ascii=False) + "\n")
    return {"status": "saved", "test_id": payload.test_id}
