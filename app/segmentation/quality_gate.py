"""Quality checks for experimental segmentation masks."""

from __future__ import annotations

from enum import Enum

import numpy as np


class SegmentationQuality(str, Enum):
    GOOD = "GOOD"
    WARNING = "WARNING"
    INVALID = "INVALID"


def assess_mask(mask: np.ndarray, roi: tuple[int, int, int, int]) -> tuple[SegmentationQuality, dict]:
    if mask.ndim != 2 or not mask.any():
        return SegmentationQuality.INVALID, {"reason": "empty_mask"}

    left, top, right, bottom = roi
    roi_mask = mask[top:bottom, left:right]
    roi_area = max(1, roi_mask.size)
    subject_area = int(roi_mask.sum())
    area_ratio = subject_area / roi_area
    if subject_area == 0 or area_ratio < 0.03:
        return SegmentationQuality.INVALID, {"reason": "subject_area_too_small", "mask_area_ratio": area_ratio}

    border = np.zeros_like(roi_mask, dtype=bool)
    border[:1, :] = True
    border[-1:, :] = True
    border[:, :1] = True
    border[:, -1:] = True
    edge_ratio = float((roi_mask & border).sum()) / max(1, subject_area)

    small = _downsample(roi_mask, 128, 128)
    components = _component_count(small)
    if edge_ratio > 0.55:
        quality = SegmentationQuality.WARNING
        reason = "mask_touches_roi_edge"
    elif components > 8:
        quality = SegmentationQuality.WARNING
        reason = "fragmented_mask"
    elif area_ratio > 0.85:
        quality = SegmentationQuality.WARNING
        reason = "subject_area_too_large"
    else:
        quality = SegmentationQuality.GOOD
        reason = "mask_passed_basic_checks"

    return quality, {
        "reason": reason,
        "mask_area_ratio": area_ratio,
        "edge_ratio": edge_ratio,
        "connected_components": components,
    }


def _downsample(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    from PIL import Image

    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    return np.asarray(image.resize((width, height), Image.Resampling.NEAREST)) > 0


def _component_count(mask: np.ndarray) -> int:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    count = 0
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            count += 1
            stack = [(y, x)]
            visited[y, x] = True
            while stack:
                cy, cx = stack.pop()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
    return count