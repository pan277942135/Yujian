"""Coordinator for the non-production fish segmentation demo."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from PIL import Image

from app.recognition_pipeline import BBox
from app.segmentation.cutout_builder import build_png_cutout
from app.segmentation.mask_generator import generate_mask
from app.segmentation.quality_gate import SegmentationQuality, assess_mask


@dataclass(frozen=True)
class FishSegmentationResult:
    quality: SegmentationQuality
    reason: str
    width: int
    height: int
    mask_area_ratio: float
    edge_ratio: float
    connected_components: int
    processing_ms: float
    cutout_png: bytes
    mask: object


def generate_fish_cutout(image: Image.Image, bbox: BBox) -> FishSegmentationResult:
    started = perf_counter()
    source = image.convert("RGB")
    mask = generate_mask(source, bbox)
    roi = _pixel_box(bbox, source.width, source.height)
    quality, metrics = assess_mask(mask, roi)
    cutout = build_png_cutout(source, mask)
    return FishSegmentationResult(
        quality=quality,
        reason=str(metrics.get("reason", "unknown")),
        width=source.width,
        height=source.height,
        mask_area_ratio=float(metrics.get("mask_area_ratio", 0.0)),
        edge_ratio=float(metrics.get("edge_ratio", 0.0)),
        connected_components=int(metrics.get("connected_components", 0)),
        processing_ms=round((perf_counter() - started) * 1000.0, 1),
        cutout_png=cutout,
        mask=mask,
    )


def _pixel_box(bbox: BBox, width: int, height: int) -> tuple[int, int, int, int]:
    b = bbox.normalized()
    import math

    left = max(0, min(width - 1, math.floor(b.x1 * width)))
    top = max(0, min(height - 1, math.floor(b.y1 * height)))
    right = max(left + 1, min(width, math.ceil(b.x2 * width)))
    bottom = max(top + 1, min(height, math.ceil(b.y2 * height)))
    return left, top, right, bottom
