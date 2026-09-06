"""Debug-only fish segmentation preview API."""

from __future__ import annotations

import base64
import io
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw

from app.detector_runtime import detect, normalize_android_source
from app.recognition_pipeline import assess_detections
from app.segmentation.mask_generator import SegmentationModelNotConfigured
from app.segmentation.service import generate_fish_cutout

MAX_DEBUG_IMAGE_BYTES = 25 * 1024 * 1024
router = APIRouter(tags=["fish-segmentation-demo"])
templates = Jinja2Templates(directory="app/templates")


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


def _mask_overlay(source: Image.Image, mask: Any, bbox: Any) -> str:
    image = source.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (13, 148, 136, 0))
    alpha = Image.fromarray((mask.astype("uint8") * 120), mode="L")
    overlay.putalpha(alpha)
    image.alpha_composite(overlay)
    draw = ImageDraw.Draw(image)
    b = bbox.normalized()
    draw.rectangle(
        (round(b.x1 * image.width), round(b.y1 * image.height), round(b.x2 * image.width), round(b.y2 * image.height)),
        outline="#dc2626",
        width=max(3, image.width // 500),
    )
    output = io.BytesIO()
    image.convert("RGB").save(output, format="PNG", optimize=True)
    return _data_url(output.getvalue(), "image/png")


@router.get("/debug/fish-segmentation", response_class=HTMLResponse)
def fish_segmentation_page(request: Request):
    return templates.TemplateResponse(request=request, name="fish_segmentation.html", context={})


@router.post("/api/debug/fish-segmentation")
async def fish_segmentation(file: UploadFile = File(..., alias="image")) -> dict[str, Any]:
    data = await _read_image(file)
    source = None
    try:
        with Image.open(io.BytesIO(data)) as uploaded:
            source = normalize_android_source(uploaded)
        detector_run = detect(source)
        assessment = assess_detections(detector_run.detections)
        if assessment.primary is None:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "NO_PRIMARY_FISH",
                    "detector": detector_run.model_version,
                    "assessment": assessment.status.value,
                    "reason": assessment.reason,
                },
            )
        result = generate_fish_cutout(source, assessment.primary.box)
        bbox = assessment.primary.box.normalized()
        return {
            "model_version": detector_run.model_version,
            "detector": {
                "confidence": round(float(assessment.primary.confidence), 6),
                "bbox_normalized": [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
                "assessment": assessment.status.value,
                "reason": assessment.reason,
                "latency_ms": detector_run.latency_ms,
                "primary_selection": "confidence × sqrt(area)",
            },
            "segmentation": {
                "model": "SAM",
                "prompt": "DET_FISH_v0.1 primary bbox",
                "width": result.width,
                "height": result.height,
                "quality": result.quality.value,
                "reason": result.reason,
                "mask_area_ratio": round(result.mask_area_ratio, 6),
                "edge_ratio": round(result.edge_ratio, 6),
                "connected_components": result.connected_components,
                "processing_ms": result.processing_ms,
            },
            "mask_overlay": _mask_overlay(source, result.mask, assessment.primary.box),
            "transparent_fish": _data_url(result.cutout_png, "image/png"),
        }
    except HTTPException:
        raise
    except SegmentationModelNotConfigured as exc:
        raise HTTPException(status_code=503, detail={"error_code": "SEGMENTATION_MODEL_UNAVAILABLE", "message": str(exc)}) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"error_code": "SEGMENTATION_FAILED", "message": str(exc)}) from exc
    finally:
        if source is not None:
            source.close()
