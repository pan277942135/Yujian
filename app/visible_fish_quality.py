"""Quality metrics and gate for the fish pixels sent to Qwen.

This module deliberately operates on masks only. It does not change Detector/SAM
inference; it makes the existing raw_sam + visible_add - remove contract
observable and blocks unsafe Qwen inputs.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Sequence

import numpy as np


def _components(mask: np.ndarray) -> list[int]:
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    sizes: list[int] = []
    height, width = mask.shape
    for y, x in zip(*np.where(mask & ~seen)):
        if seen[y, x]:
            continue
        stack = [(int(y), int(x))]
        seen[y, x] = True
        size = 0
        while stack:
            cy, cx = stack.pop()
            size += 1
            for ny in range(cy - 1, cy + 2):
                for nx in range(cx - 1, cx + 2):
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not seen[ny, nx]
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        sizes.append(size)
    return sorted(sizes, reverse=True)


def _hole_pixels(mask: np.ndarray) -> int:
    """Return enclosed background pixels using a padded 4-connected flood fill."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    background = ~padded
    visited = np.zeros(background.shape, dtype=bool)
    height, width = background.shape
    queue: deque[tuple[int, int]] = deque()
    for x in range(width):
        queue.append((0, x))
        queue.append((height - 1, x))
    for y in range(1, height - 1):
        queue.append((y, 0))
        queue.append((y, width - 1))
    while queue:
        y, x = queue.popleft()
        if not (0 <= y < height and 0 <= x < width):
            continue
        if visited[y, x] or not background[y, x]:
            continue
        visited[y, x] = True
        queue.extend(((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)))
    return int((background & ~visited).sum())


def _downsample_mask(mask: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return np.asarray(mask, dtype=bool)
    height, width = mask.shape
    target_height = int(np.ceil(height / scale))
    target_width = int(np.ceil(width / scale))
    padded = np.pad(
        np.asarray(mask, dtype=bool),
        ((0, target_height * scale - height), (0, target_width * scale - width)),
        mode="constant",
        constant_values=False,
    )
    return padded.reshape(target_height, scale, target_width, scale).any(axis=(1, 3))


def _clamped_bbox(
    bbox_pixels: Sequence[int] | None,
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    height, width = shape
    if not bbox_pixels or len(bbox_pixels) != 4:
        return (0, 0, width, height)
    x1, y1, x2, y2 = (int(value) for value in bbox_pixels)
    return (
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )


def analyze_visible_fish_quality(
    raw_mask: np.ndarray,
    refined_mask: np.ndarray,
    bbox_pixels: Sequence[int] | None,
    image_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Measure whether a refined visible mask is safe to send to Qwen.

    refined_mask is expected to have already been produced by the existing
    (raw_sam | visible_add) & ~remove formula. The function never mutates
    either input.
    """
    raw = np.asarray(raw_mask, dtype=bool)
    refined = np.asarray(refined_mask, dtype=bool)
    if raw.ndim != 2 or refined.ndim != 2 or raw.shape != refined.shape:
        raise ValueError("raw_mask and refined_mask must be same-shaped 2D masks")
    shape = tuple(image_shape or raw.shape)
    if shape != raw.shape:
        raise ValueError("image_shape must match mask shape")

    raw_area = int(raw.sum())
    refined_area = int(refined.sum())
    metric_scale = max(1, int(np.ceil(max(raw.shape) / 1024)))
    metric_refined = _downsample_mask(refined, metric_scale)
    metric_area = int(metric_refined.sum())
    component_sizes = _components(metric_refined)
    component_count = len(component_sizes)
    largest_component = component_sizes[0] if component_sizes else 0
    largest_component_ratio = largest_component / metric_area if metric_area else 0.0

    original_bbox = _clamped_bbox(bbox_pixels, raw.shape)
    x1, y1, x2, y2 = (
        int(np.floor(original_bbox[0] / metric_scale)),
        int(np.floor(original_bbox[1] / metric_scale)),
        int(np.ceil(original_bbox[2] / metric_scale)),
        int(np.ceil(original_bbox[3] / metric_scale)),
    )
    x1, y1, x2, y2 = _clamped_bbox((x1, y1, x2, y2), metric_refined.shape)
    bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
    inside_bbox = int(metric_refined[y1:y2, x1:x2].sum()) if bbox_area else 0
    bbox_coverage_ratio = inside_bbox / bbox_area if bbox_area else 0.0

    holes = _hole_pixels(metric_refined)
    bbox_fill_ratio = metric_area / bbox_area if bbox_area else 0.0
    hole_ratio = holes / max(1, metric_area + holes)

    edge_pixels = int(
        metric_refined[0, :].sum()
        + metric_refined[-1, :].sum()
        + metric_refined[:, 0].sum()
        + metric_refined[:, -1].sum()
    )
    if metric_refined.shape[0] == 1 or metric_refined.shape[1] == 1:
        edge_pixels = int(metric_refined.sum())
    edge_truncation = min(1.0, edge_pixels / max(1, metric_area))

    retained = int((raw & refined).sum())
    visible_retention_ratio = retained / raw_area if raw_area else 0.0

    segment_presence: list[bool] = []
    if x2 > x1 and y2 > y1:
        span = x2 - x1
        for index in range(3):
            sx1 = x1 + (span * index) // 3
            sx2 = x1 + (span * (index + 1)) // 3
            segment_presence.append(bool(refined[y1:y2, sx1:sx2].any()))
    structural_break = refined_area > 0 and sum(segment_presence) < 2
    raw_equivalent = bool(np.array_equal(raw, refined))
    refinement_applied = not raw_equivalent

    invalid_reasons: list[str] = []
    warning_reasons: list[str] = []
    if refined_area == 0:
        invalid_reasons.append("VISIBLE_FISH_EMPTY")
    if component_count >= 4 or largest_component_ratio < 0.55:
        invalid_reasons.append("VISIBLE_FISH_DISCONNECTED")
    elif component_count > 1 or largest_component_ratio < 0.82:
        warning_reasons.append("VISIBLE_FISH_MULTIPLE_COMPONENTS")
    if bbox_area <= 0 or bbox_coverage_ratio < 0.12:
        invalid_reasons.append("VISIBLE_FISH_BBOX_COVERAGE_LOW")
    elif bbox_coverage_ratio < 0.28:
        warning_reasons.append("VISIBLE_FISH_BBOX_COVERAGE_LOW")
    if hole_ratio > 0.45:
        invalid_reasons.append("VISIBLE_FISH_HOLES_LARGE")
    elif hole_ratio > 0.18:
        warning_reasons.append("VISIBLE_FISH_HOLES_PRESENT")
    if structural_break:
        invalid_reasons.append("VISIBLE_FISH_HEAD_BODY_TAIL_BREAK")
    if edge_truncation > 0.65:
        invalid_reasons.append("VISIBLE_FISH_EDGE_TRUNCATION")
    elif edge_truncation > 0.35:
        warning_reasons.append("VISIBLE_FISH_EDGE_TRUNCATION")
    if raw_area and visible_retention_ratio < 0.70:
        invalid_reasons.append("VISIBLE_FISH_RAW_RETENTION_LOW")
    elif raw_area and visible_retention_ratio < 0.92:
        warning_reasons.append("VISIBLE_FISH_RAW_RETENTION_LOW")
    quality = "INVALID" if invalid_reasons else "WARNING" if warning_reasons else "GOOD"
    reasons = invalid_reasons + warning_reasons
    return {
        "visible_fish_quality": quality,
        "quality_gate_passed": quality == "GOOD",
        "quality_reasons": reasons,
        "invalid_reasons": invalid_reasons,
        "warning_reasons": warning_reasons,
        "raw_sam_area": raw_area,
        "refined_visible_area": refined_area,
        "bbox_coverage_ratio": round(bbox_coverage_ratio, 6),
        "bbox_fill_ratio": round(bbox_fill_ratio, 6),
        "connected_components": component_count,
        "largest_component_ratio": round(largest_component_ratio, 6),
        "hole_ratio": round(hole_ratio, 6),
        "edge_truncation": round(edge_truncation, 6),
        "visible_retention_ratio": round(visible_retention_ratio, 6),
        "raw_equivalent": raw_equivalent,
        "refinement_applied": refinement_applied,
        "component_sizes": component_sizes[:8],
        "structure_segment_presence": segment_presence,
        "metric_scale": metric_scale,
    }


def quality_status(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("visible_fish_quality") or value.get("quality") or value.get("status")
    return str(value or "INVALID").strip().upper()


__all__ = ["analyze_visible_fish_quality", "quality_status"]
