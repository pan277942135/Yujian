"""Read-only Android ↔ backend detector parity probe."""

from __future__ import annotations

import base64
import io
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw
from starlette.requests import Request

from app.detector_runtime import (
    ANDROID_MAX_SOURCE_DIMENSION,
    YOLOX_LETTERBOX_FILL,
    detect,
    normalize_android_source,
)
from app.recognition_pipeline import assess_detections, load_contract


MAX_DEBUG_IMAGE_BYTES = 25 * 1024 * 1024
router = APIRouter(tags=["detector-parity"])
templates = Jinja2Templates(directory="app/templates")


async def _read_debug_image(file: UploadFile) -> bytes:
    if file.content_type and file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="仅支持 JPG、PNG、WEBP 图片")
    data = await file.read(MAX_DEBUG_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="请选择图片")
    if len(data) > MAX_DEBUG_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="图片不能超过 25 MiB")
    return data


def _pixel_box(box: Any, width: int, height: int) -> list[int]:
    normalized = box.normalized()
    return [
        round(normalized.x1 * width),
        round(normalized.y1 * height),
        round(normalized.x2 * width),
        round(normalized.y2 * height),
    ]


def _normalized_xywh(box: Any) -> list[float]:
    normalized = box.normalized()
    return [
        round(normalized.x1, 6),
        round(normalized.y1, 6),
        round(normalized.width, 6),
        round(normalized.height, 6),
    ]


def _overlay(source: Image.Image, detections: tuple[Any, ...], primary: Any | None) -> str:
    image = source.copy().convert("RGB")
    draw = ImageDraw.Draw(image)
    for detection in detections:
        box = _pixel_box(detection.box, image.width, image.height)
        is_primary = primary is detection
        color = "#dc2626" if is_primary else "#f59e0b"
        draw.rectangle(box, outline=color, width=max(3, image.width // 500))
        draw.text((box[0] + 6, box[1] + 6), f"{detection.confidence:.4f}", fill=color)
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


@router.get("/debug/detector-parity", response_class=HTMLResponse)
def detector_parity_page(request: Request):
    return templates.TemplateResponse(request=request, name="detector_parity.html", context={})


@router.post("/api/debug/detector-parity")
async def detector_parity(file: UploadFile = File(..., alias="image")) -> dict[str, Any]:
    data = await _read_debug_image(file)
    try:
        with Image.open(io.BytesIO(data)) as uploaded:
            original_width, original_height = uploaded.size
            image_format = uploaded.format or "unknown"
            source = normalize_android_source(uploaded)
        try:
            detector_run = detect(source)
            assessment = assess_detections(detector_run.detections)
            primary = assessment.primary
            contract = load_contract()
            detector_cfg = contract["detector"]
            detections = [
                {
                    "confidence": round(float(item.confidence), 6),
                    "bbox_pixel": _pixel_box(item.box, source.width, source.height),
                    "bbox_normalized": _normalized_xywh(item.box),
                    "bbox_area_ratio": round(float(item.area_ratio), 6),
                    "class_name": item.class_name,
                }
                for item in detector_run.detections
            ]
            primary_payload = None
            if primary is not None:
                primary_payload = {
                    "confidence": round(float(primary.confidence), 6),
                    "bbox_pixel": _pixel_box(primary.box, source.width, source.height),
                    "bbox_normalized": _normalized_xywh(primary.box),
                    "bbox_area_ratio": round(float(primary.area_ratio), 6),
                }
            return {
                "model_version": detector_run.model_version,
                "image": {
                    "filename": file.filename or "image",
                    "format": image_format,
                    "original_width": original_width,
                    "original_height": original_height,
                    "width": source.width,
                    "height": source.height,
                },
                "detector": {
                    **(primary_payload or {"confidence": None, "bbox_pixel": None, "bbox_normalized": None, "bbox_area_ratio": 0.0}),
                    "model_version": detector_run.model_version,
                    "detections": detections,
                    "primary_selection": "confidence × sqrt(area)",
                    "assessment": assessment.status.value,
                    "reason": assessment.reason,
                    "latency_ms": detector_run.latency_ms,
                },
                "preprocess": {
                    "exif_transpose": True,
                    "source_color_mode": "RGB",
                    "max_source_dimension": ANDROID_MAX_SOURCE_DIMENSION,
                    "source_resize": {
                        "input_width": original_width,
                        "input_height": original_height,
                        "output_width": source.width,
                        "output_height": source.height,
                    },
                    "letterbox": {
                        "input_size": detector_run.input_size,
                        "fill": YOLOX_LETTERBOX_FILL,
                        "scale": detector_run.input_scale,
                        "draw_width": detector_run.input_draw_width,
                        "draw_height": detector_run.input_draw_height,
                        "placement": "top_left",
                    },
                    "channel_order": "BGR",
                    "tensor_layout": "NCHW",
                    "dtype": "float32",
                    "normalization": "none (0..255 float32)",
                    "weak_confidence": float(detector_cfg["weak_confidence"]),
                },
                "overlay": {"media_type": "image/png", "data_url": _overlay(source, detector_run.detections, primary)},
            }
        finally:
            source.close()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Detector parity 推理失败：{exc}") from exc
