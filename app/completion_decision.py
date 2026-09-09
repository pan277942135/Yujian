"""Conservative automatic completion decision engine for the Fish Completion Lab."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np

AUTO_COMPLETION = "AUTO_COMPLETION"
MANUAL_DEBUG = "MANUAL_DEBUG"


@dataclass(frozen=True)
class CompletionDecision:
    mode: str
    status: str
    completion_required: bool
    severity: str
    completion_ratio: float
    occlusion_detected: bool
    completion_mask: np.ndarray
    occluder_mask: np.ndarray
    case_class: str
    reason: str
    source: str = "AUTO"

    def as_dict(self, *, include_masks: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "mode": self.mode,
            "status": self.status,
            "completion_required": self.completion_required,
            "severity": self.severity,
            "completion_ratio": self.completion_ratio,
            "occlusion_detected": self.occlusion_detected,
            "case_class": self.case_class,
            "reason": self.reason,
            "source": self.source,
            "mask_generated": bool(self.completion_mask.any()),
        }
        if include_masks:
            result["completion_mask"] = self.completion_mask
            result["occluder_mask"] = self.occluder_mask
        return result


def _normalise_label(value: str) -> str:
    return re.sub(r"\s+", "", (value or "").strip().lower())


def _has_any(label: str, tokens: tuple[str, ...]) -> bool:
    return any(token in label for token in tokens)


def _bbox_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = [int(value) for value in bbox]
    x1, y1 = max(0, min(width, x1)), max(0, min(height, y1))
    x2, y2 = max(x1, min(width, x2)), max(y1, min(height, y2))
    mask = np.zeros(shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    result = mask.copy()
    for _ in range(max(0, int(radius))):
        result = (
            result
            | np.roll(result, 1, axis=0)
            | np.roll(result, -1, axis=0)
            | np.roll(result, 1, axis=1)
            | np.roll(result, -1, axis=1)
        )
        result[0, :] = False
        result[-1, :] = False
        result[:, 0] = False
        result[:, -1] = False
    return result


def _candidate_ring(raw_mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    for radius in (12, 8, 5, 3, 1):
        candidate = _dilate(raw_mask, radius) & ~raw_mask & _bbox_mask(raw_mask.shape, bbox)
        ratio = float(candidate.sum() / max(1, raw_mask.sum() + candidate.sum()))
        if candidate.any() and ratio <= 0.20:
            return candidate
    return np.zeros_like(raw_mask, dtype=bool)


def decide_completion(
    raw_mask: np.ndarray,
    detector_bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    case_label: str = "",
    segmentation_quality: str = "",
) -> CompletionDecision:
    """Return a safe decision; ambiguous images never receive a fabricated mask."""
    height, width = image_size
    mask = np.asarray(raw_mask, dtype=bool)
    if mask.shape != (height, width):
        raise ValueError("raw_mask shape must match image_size")

    label = _normalise_label(case_label)
    bbox = _bbox_mask(mask.shape, detector_bbox)
    raw_area = int(mask.sum())
    bbox_area = int(bbox.sum())
    visible_ratio = raw_area / max(1, bbox_area)

    if raw_area == 0 or visible_ratio < 0.50 or _has_any(label, ("case4", "严重缺失", "严重", "over50", "not_eligible")):
        return CompletionDecision(AUTO_COMPLETION, "NOT_ELIGIBLE", False, "SEVERE", 0.0, False, np.zeros_like(mask), np.zeros_like(mask), "SEVERE_MISSING", "VISIBLE_FISH_AREA_INSUFFICIENT")

    if _has_any(label, ("case1", "完整", "complete", "not_required")):
        return CompletionDecision(AUTO_COMPLETION, "NOT_REQUIRED", False, "NONE", 0.0, False, np.zeros_like(mask), np.zeros_like(mask), "FULL", "VISIBLE_FISH_COMPLETE")

    is_occlusion = _has_any(label, ("case3", "遮挡", "occlud", "hand", "手", "鱼护", "工具", "object"))
    is_missing = _has_any(label, ("case2", "缺失", "missing", "tail", "尾", "鳍", "fin", "边缘"))
    complex_background = _has_any(label, ("case5", "复杂背景", "水草", "石头", "杂物", "complex"))

    candidate = _candidate_ring(mask, detector_bbox) if (is_occlusion or is_missing) else np.zeros_like(mask, dtype=bool)
    candidate_ratio = float(candidate.sum() / max(1, raw_area + candidate.sum()))
    if candidate.any() and candidate_ratio <= 0.20:
        severity = "MEDIUM" if candidate_ratio <= 0.10 else "HEAVY"
        case_class = "OCCLUSION" if is_occlusion else ("COMPLEX_BACKGROUND" if complex_background else "LIGHT_MISSING")
        return CompletionDecision(AUTO_COMPLETION, "MASK_READY", True, severity, round(candidate_ratio, 6), is_occlusion, candidate, candidate.copy(), case_class, "AUTO_MASK_FROM_HIGH_CONFIDENCE_DEBUG_SIGNAL")

    return CompletionDecision(AUTO_COMPLETION, "REVIEW_REQUIRED", False, "REVIEW", 0.0, False, np.zeros_like(mask), np.zeros_like(mask), "COMPLEX_BACKGROUND" if complex_background else "AMBIGUOUS", "NO_HIGH_CONFIDENCE_COMPLETION_SIGNAL")
