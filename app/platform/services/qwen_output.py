"""Deterministic Qwen RGB -> transparent fish RGBA post-processing.

This module deliberately re-segments the generated Qwen image. It never reuses
the original Visible Fish mask and never runs a generative model.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from app.detector_runtime import detect, normalize_android_source
from app.recognition_pipeline import BBox, assess_detections
from app.segmentation.service import generate_fish_cutout


class QwenOutputError(RuntimeError):
    """A classified failure while producing the transparent Qwen asset."""

    def __init__(self, error_code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}


@dataclass(frozen=True)
class QwenOutputArtifacts:
    qwen_result_rgb: bytes
    fish_mask_raw: bytes
    fish_mask: bytes
    transparent_fish: bytes
    metadata: dict[str, Any]


def _png(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _mask_png(mask: np.ndarray) -> bytes:
    return _png(Image.fromarray(np.where(mask, 255, 0).astype("uint8"), mode="L"))


def normalise_qwen_rgb(data: bytes) -> bytes:
    """Decode the Worker result and preserve it as a lossless RGB PNG."""

    try:
        with Image.open(io.BytesIO(data)) as source:
            image = source.convert("RGB")
            if image.width <= 0 or image.height <= 0:
                raise ValueError("image has invalid dimensions")
            return _png(image)
    except Exception as exc:
        raise QwenOutputError(
            "QWEN_RGBA_EXPORT_FAILED",
            f"Qwen RGB 结果无法读取: {exc}",
        ) from exc


def _label_components(binary: np.ndarray) -> tuple[np.ndarray, list[int]]:
    source = np.asarray(binary, dtype=bool)
    if source.ndim != 2:
        raise QwenOutputError("QWEN_SEGMENTATION_INVALID_MASK", "Fish mask 必须是二维单通道矩阵")

    height, width = source.shape
    labels = np.full((height, width), -1, dtype=np.int32)
    sizes: list[int] = []
    label = 0
    for y in range(height):
        for x in range(width):
            if not source[y, x] or labels[y, x] >= 0:
                continue
            labels[y, x] = label
            stack = [(y, x)]
            count = 0
            while stack:
                current_y, current_x = stack.pop()
                count += 1
                for next_y, next_x in (
                    (current_y - 1, current_x),
                    (current_y + 1, current_x),
                    (current_y, current_x - 1),
                    (current_y, current_x + 1),
                ):
                    if (
                        0 <= next_y < height
                        and 0 <= next_x < width
                        and source[next_y, next_x]
                        and labels[next_y, next_x] < 0
                    ):
                        labels[next_y, next_x] = label
                        stack.append((next_y, next_x))
            sizes.append(count)
            label += 1
    return labels, sizes


def _dilate_once(mask: np.ndarray) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    height, width = source.shape
    padded = np.pad(source, 1, mode="constant", constant_values=False)
    output = np.zeros_like(source)
    for dy in range(3):
        for dx in range(3):
            output |= padded[dy : dy + height, dx : dx + width]
    return output


def _refine_mask(raw_mask: np.ndarray) -> np.ndarray:
    """Remove islands/fill small holes without eroding fish fins."""

    mask = np.asarray(raw_mask, dtype=bool)
    labels, sizes = _label_components(mask)
    if not sizes:
        raise QwenOutputError("QWEN_ALPHA_EMPTY", "SAM 没有返回有效鱼体区域")

    largest_label = int(np.argmax(sizes))
    largest_size = sizes[largest_label]
    refined = labels == largest_label

    # Keep a meaningful component that touches the main body after a one-pixel
    # dilation. This preserves thin fin/tail fragments without retaining remote
    # background islands.
    adjacent = _dilate_once(refined)
    min_adjacent_size = max(8, int(largest_size * 0.0005))
    for component_label, component_size in enumerate(sizes):
        if component_label == largest_label or component_size < min_adjacent_size:
            continue
        component = labels == component_label
        if np.any(component & adjacent):
            refined |= component

    # Fill only enclosed small holes. Border-connected background remains
    # transparent, and no erosion is applied to the fish edge.
    background_labels, background_sizes = _label_components(~refined)
    border_labels = set(
        np.concatenate(
            (
                background_labels[0, :],
                background_labels[-1, :],
                background_labels[:, 0],
                background_labels[:, -1],
            )
        ).tolist()
    )
    hole_limit = max(64, min(4096, int(max(1, largest_size) * 0.02)))
    for hole_label, hole_size in enumerate(background_sizes):
        if hole_label not in border_labels and hole_size <= hole_limit:
            refined[background_labels == hole_label] = True

    return refined


def _alpha_from_mask(mask: np.ndarray) -> np.ndarray:
    """Create a one-pixel anti-aliased alpha edge without expanding the subject."""

    source = Image.fromarray(np.where(mask, 255, 0).astype("uint8"), mode="L")
    blurred = np.asarray(source.filter(ImageFilter.GaussianBlur(radius=0.8)), dtype=np.uint8)
    allowed = _dilate_once(mask)
    alpha = np.where(allowed, blurred, 0).astype("uint8")
    alpha[np.asarray(mask, dtype=bool)] = 255
    return alpha


def _decontaminate_edge_rgb(rgb: np.ndarray, mask: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Remove fully transparent RGB and replace one-pixel edge RGB from fish neighbors."""

    output = np.asarray(rgb, dtype=np.uint8).copy()
    output[alpha == 0] = 0
    edge = (alpha > 0) & ~np.asarray(mask, dtype=bool)
    if not np.any(edge):
        return output

    height, width = mask.shape
    accumulated = np.zeros((height, width, 3), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            source_y0 = max(0, -dy)
            source_y1 = min(height, height - dy)
            source_x0 = max(0, -dx)
            source_x1 = min(width, width - dx)
            target_y0 = max(0, dy)
            target_y1 = min(height, height + dy)
            target_x0 = max(0, dx)
            target_x1 = min(width, width + dx)
            source_mask = np.asarray(mask, dtype=bool)[source_y0:source_y1, source_x0:source_x1]
            target_mask = np.zeros((height, width), dtype=bool)
            target_mask[target_y0:target_y1, target_x0:target_x1] = source_mask
            shifted_rgb = np.zeros_like(output)
            shifted_rgb[target_y0:target_y1, target_x0:target_x1] = output[
                source_y0:source_y1, source_x0:source_x1
            ]
            accumulated += shifted_rgb * target_mask[:, :, None]
            counts += target_mask

    valid = edge & (counts > 0)
    if np.any(valid):
        output[valid] = np.rint(accumulated[valid] / counts[valid, None]).astype("uint8")
    return output


def _fallback_bbox() -> BBox:
    # Qwen normally returns a single centered fish. The inset leaves a safe
    # margin for SAM while avoiding a full-frame prompt.
    return BBox(0.06, 0.06, 0.94, 0.94)


def _resolve_bbox(source: Image.Image) -> tuple[BBox, dict[str, Any]]:
    """Use the production Detector contract, then one conservative SAM fallback."""

    fallback = _fallback_bbox()
    detector_info: dict[str, Any] = {
        "status": "FALLBACK",
        "reason": "detector_no_fish",
        "bbox_normalized": [fallback.x1, fallback.y1, fallback.x2, fallback.y2],
    }
    detector_source = None
    try:
        detector_source = normalize_android_source(source)
        detector_run = detect(detector_source)
        assessment = assess_detections(detector_run.detections)
        primary = assessment.primary
        if primary is not None:
            box = primary.box.normalized()
            return box, {
                "status": "PASS",
                "model": detector_run.model_version,
                "assessment": assessment.status.value,
                "confidence": round(float(primary.confidence), 6),
                "bbox_normalized": [box.x1, box.y1, box.x2, box.y2],
            }
        detector_info["reason"] = "detector_no_fish"
        detector_info["assessment"] = assessment.status.value
    except Exception as exc:
        detector_info["reason"] = "detector_unavailable"
        detector_info["error"] = f"{exc.__class__.__name__}: {exc}"[:500]
    finally:
        if detector_source is not None and detector_source is not source:
            detector_source.close()
    return fallback, detector_info


def process_qwen_output(data: bytes) -> QwenOutputArtifacts:
    """Re-segment a Qwen RGB result and export the four required artifacts."""

    try:
        rgb_bytes = normalise_qwen_rgb(data)
        with Image.open(io.BytesIO(rgb_bytes)) as decoded:
            source = decoded.convert("RGB")
    except QwenOutputError:
        raise
    except Exception as exc:
        raise QwenOutputError("QWEN_RGBA_EXPORT_FAILED", f"Qwen RGB 结果无法读取: {exc}") from exc

    try:
        bbox, detector_info = _resolve_bbox(source)
        try:
            segmentation = generate_fish_cutout(source, bbox)
        except Exception as exc:
            raise QwenOutputError(
                "QWEN_SEGMENTATION_FAILED",
                f"Qwen 结果二次 SAM 分割失败: {exc}",
                {"detector": detector_info},
            ) from exc

        raw_mask = np.asarray(segmentation.mask, dtype=bool)
        if raw_mask.shape != (source.height, source.width):
            raise QwenOutputError(
                "QWEN_SEGMENTATION_INVALID_MASK",
                f"Fish mask 尺寸 {raw_mask.shape} 与 Qwen 图片 {(source.height, source.width)} 不一致",
                {"detector": detector_info},
            )
        if not np.any(raw_mask):
            raise QwenOutputError(
                "QWEN_SEGMENTATION_NO_FISH",
                "Qwen 结果中没有可分割的鱼体",
                {"detector": detector_info},
            )

        final_mask = _refine_mask(raw_mask)
        foreground_ratio = float(final_mask.mean())
        if foreground_ratio <= 0.0005:
            raise QwenOutputError(
                "QWEN_ALPHA_EMPTY",
                "最终鱼体 Alpha 区域过小",
                {"foreground_ratio": foreground_ratio, "detector": detector_info},
            )
        if foreground_ratio > 0.95:
            raise QwenOutputError(
                "QWEN_ALPHA_FULL_FRAME",
                "最终鱼体 Alpha 覆盖了几乎整张图片",
                {"foreground_ratio": foreground_ratio, "detector": detector_info},
            )

        source_rgb = np.asarray(source, dtype=np.uint8)
        alpha = _alpha_from_mask(final_mask)
        composed_rgb = _decontaminate_edge_rgb(source_rgb, final_mask, alpha)
        rgba = np.dstack((composed_rgb, alpha)).astype("uint8")
        total_pixels = int(alpha.size)
        metadata = {
            "method": "DETECTOR_SAM_RESEGMENTATION",
            "detector": detector_info,
            "sam_model": f"SAM_{os.getenv('SEGMENTATION_MODEL_TYPE', 'vit_b').strip().upper()}",
            "width": int(source.width),
            "height": int(source.height),
            "format": "PNG",
            "mode": "RGBA",
            "channels": 4,
            "alpha_min": int(alpha.min()),
            "alpha_max": int(alpha.max()),
            "alpha_coverage": round(float(np.count_nonzero(alpha)) / total_pixels, 6),
            "transparent_pixel_ratio": round(float(np.count_nonzero(alpha == 0)) / total_pixels, 6),
            "opaque_pixel_ratio": round(float(np.count_nonzero(alpha == 255)) / total_pixels, 6),
            "foreground_ratio": round(foreground_ratio, 6),
            "raw_foreground_ratio": round(float(raw_mask.mean()), 6),
            "edge_feather_px": 1,
            "rgb_preserved_inside_fish": True,
        }
        return QwenOutputArtifacts(
            qwen_result_rgb=rgb_bytes,
            fish_mask_raw=_mask_png(raw_mask),
            fish_mask=_mask_png(final_mask),
            transparent_fish=_png(Image.fromarray(rgba, mode="RGBA")),
            metadata=metadata,
        )
    finally:
        source.close()


__all__ = [
    "QwenOutputArtifacts",
    "QwenOutputError",
    "normalise_qwen_rgb",
    "process_qwen_output",
]
