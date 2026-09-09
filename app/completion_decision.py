"""Structural AUTO_COMPLETION decision engine.

The engine uses only segmentation geometry and detector/runtime signals. Test
labels are metadata and are intentionally not accepted as decision inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

AUTO_COMPLETION = "AUTO_COMPLETION"
MANUAL_DEBUG = "MANUAL_DEBUG"

STRUCTURAL_THRESHOLDS: dict[str, float] = {
    "min_component_pixels": 16,
    "min_support_bins": 2,
    "max_internal_gap_fraction": 0.45,
    "local_width_drop_ratio": 0.55,
    "min_decision_confidence": 0.70,
    "min_large_experimental_confidence": 0.65,
    "max_pathological_completion_regions": 64,
}


@dataclass(frozen=True)
class CompletionDecision:
    mode: str
    status: str
    completion_required: bool
    execution_allowed: bool
    severity: str
    completion_ratio: float
    occlusion_detected: bool
    completion_mask: np.ndarray
    occluder_mask: np.ndarray
    estimated_full_fish_mask: np.ndarray
    clean_visible_mask: np.ndarray
    axis_origin_xy: tuple[float, float]
    axis_vector_xy: tuple[float, float]
    axis_endpoints_xy: tuple[tuple[float, float], tuple[float, float]]
    case_class: str
    reason: list[str]
    decision_confidence: float
    sam_bbox_fill_ratio: float
    raw_component_count: int
    clean_component_count: int
    noise_removed_pixels: int
    axis_angle_deg: float
    axis_length_px: float
    structural_gap_count: int
    candidate_area_pixels: int
    completion_area_pixels: int
    completion_region_count: int
    ring_candidate_rejected: bool
    source: str = "AUTO"

    @property
    def completion_percent(self) -> float:
        return float(self.completion_ratio) * 100.0

    def as_dict(self, *, include_masks: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "mode": self.mode,
            "source": self.source,
            "status": self.status,
            "case_class": self.case_class,
            "completion_required": self.completion_required,
            "execution_allowed": bool(self.execution_allowed),
            "severity": self.severity,
            "decision_confidence": round(float(self.decision_confidence), 6),
            "completion_ratio": round(float(self.completion_ratio), 6),
            "estimated_missing_ratio": round(float(self.completion_ratio), 6),
            "completion_percent": round(float(self.completion_ratio) * 100.0, 2),
            "sam_bbox_fill_ratio": round(float(self.sam_bbox_fill_ratio), 6),
            "structural_gap_count": int(self.structural_gap_count),
            "raw_component_count": int(self.raw_component_count),
            "clean_component_count": int(self.clean_component_count),
            "noise_removed_pixels": int(self.noise_removed_pixels),
            "axis_angle_deg": round(float(self.axis_angle_deg), 3),
            "axis_length_px": round(float(self.axis_length_px), 3),
            "candidate_area_pixels": int(self.candidate_area_pixels),
            "completion_area_pixels": int(self.completion_area_pixels),
            "completion_region_count": int(self.completion_region_count),
            "ring_candidate_rejected": bool(self.ring_candidate_rejected),
            "reason": list(self.reason),
            "mask_generated": bool(self.completion_mask.any()),
            "visible_overlap_pixels": int((self.completion_mask & self.clean_visible_mask).sum()),
        }
        if include_masks:
            result.update({
                "completion_mask": self.completion_mask,
                "occluder_mask": self.occluder_mask,
                "estimated_full_fish_mask": self.estimated_full_fish_mask,
                "clean_visible_mask": self.clean_visible_mask,
            })
        return result


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result: list[np.ndarray] = []
    for y, x in zip(*np.where(mask)):
        if seen[y, x]:
            continue
        stack = [(int(y), int(x))]
        seen[y, x] = True
        pixels: list[tuple[int, int]] = []
        while stack:
            cy, cx = stack.pop()
            pixels.append((cy, cx))
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        component = np.zeros_like(mask, dtype=bool)
        if pixels:
            py, px = zip(*pixels)
            component[np.asarray(py), np.asarray(px)] = True
        result.append(component)
    return sorted(result, key=lambda item: int(item.sum()), reverse=True)


def _bbox_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = [int(value) for value in bbox]
    x1, y1 = max(0, min(width, x1)), max(0, min(height, y1))
    x2, y2 = max(x1, min(width, x2)), max(y1, min(height, y2))
    result = np.zeros(shape, dtype=bool)
    result[y1:y2, x1:x2] = True
    return result


def _clean_visible_mask(raw_mask: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    raw = np.asarray(raw_mask, dtype=bool)
    components = _components(raw)
    raw_count = len(components)
    if not components:
        return np.zeros_like(raw), {"raw_component_count": 0, "clean_component_count": 0, "noise_removed_pixels": 0}
    largest = int(components[0].sum())
    minimum = max(int(STRUCTURAL_THRESHOLDS["min_component_pixels"]), int(largest * 0.01))
    clean = np.zeros_like(raw)
    for component in components:
        if int(component.sum()) >= minimum:
            clean |= component
    if not clean.any():
        clean = components[0].copy()
    return clean, {
        "raw_component_count": raw_count,
        "clean_component_count": len(_components(clean)),
        "noise_removed_pixels": int(raw.sum() - clean.sum()),
    }


def _principal_axis(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    ys, xs = np.where(mask)
    if len(xs) < 2:
        return np.array([0.0, 0.0]), np.array([1.0, 0.0]), 0.0, 0.0
    points = np.column_stack((xs.astype(float), ys.astype(float)))
    origin = points.mean(axis=0)
    covariance = np.cov(points - origin, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    vector = eigenvectors[:, int(np.argmax(eigenvalues))]
    if vector[0] < 0:
        vector = -vector
    vector = vector / max(float(np.linalg.norm(vector)), 1e-9)
    projected = (points - origin) @ vector
    axis_length = float(projected.max() - projected.min())
    angle = float(np.degrees(np.arctan2(vector[1], vector[0])))
    return origin, vector, angle, axis_length


def _profile(mask: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> dict[str, Any]:
    ys, xs = np.where(mask)
    points = np.column_stack((xs.astype(float), ys.astype(float)))
    perpendicular = np.array([-axis[1], axis[0]])
    u = (points - origin) @ axis
    v = (points - origin) @ perpendicular
    u_min, u_max = float(u.min()), float(u.max())
    axis_length = max(1.0, u_max - u_min)
    bins = int(np.clip(round(axis_length / 8.0), 16, 96))
    indices = np.clip(((u - u_min) / axis_length * bins).astype(int), 0, bins - 1)
    widths = np.zeros(bins, dtype=float)
    centers = np.full(bins, np.nan, dtype=float)
    counts = np.zeros(bins, dtype=int)
    for index in range(bins):
        values = v[indices == index]
        if len(values):
            widths[index] = float(values.max() - values.min() + 1.0)
            centers[index] = float(np.median(values))
            counts[index] = len(values)
    occupied = counts >= max(2, int(mask.sum() / max(1, bins * 100)))
    return {
        "origin": origin,
        "axis": axis,
        "perpendicular": perpendicular,
        "u_min": u_min,
        "u_max": u_max,
        "bins": bins,
        "widths": widths,
        "centers": centers,
        "counts": counts,
        "occupied": occupied,
    }


def _segments(values: np.ndarray) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    start: int | None = None
    for index, present in enumerate(values):
        if present and start is None:
            start = index
        elif not present and start is not None:
            result.append((start, index - 1))
            start = None
    if start is not None:
        result.append((start, len(values) - 1))
    return result


def _interpolate_band(profile: dict[str, Any], start_bin: int, end_bin: int) -> np.ndarray:
    height, width = profile["mask_shape"]
    yy, xx = np.mgrid[0:height, 0:width]
    origin = profile["origin"]
    axis = profile["axis"]
    perpendicular = profile["perpendicular"]
    points = np.column_stack((xx.ravel(), yy.ravel()))
    u = ((points - origin) @ axis).reshape((height, width))
    v = ((points - origin) @ perpendicular).reshape((height, width))
    u_min, u_max = profile["u_min"], profile["u_max"]
    bins = profile["bins"]
    bin_width = max(1e-6, (u_max - u_min) / bins)
    gap_u1 = u_min + start_bin * bin_width
    gap_u2 = u_min + (end_bin + 1) * bin_width
    left = max(0, start_bin - 3)
    right = min(bins - 1, end_bin + 3)
    known = np.where(profile["occupied"] & np.isfinite(profile["centers"]))[0]
    if len(known) < 2:
        return np.zeros((height, width), dtype=bool)
    sample_bins = np.unique(np.asarray([
        known[np.argmin(np.abs(known - left))],
        known[np.argmin(np.abs(known - right))],
    ]))
    if len(sample_bins) < 2:
        return np.zeros((height, width), dtype=bool)
    sample_u = u_min + (sample_bins + 0.5) * bin_width
    center_v = np.interp(u, sample_u, profile["centers"][sample_bins])
    half_width = np.interp(u, sample_u, profile["widths"][sample_bins] / 2.0)
    return (u >= gap_u1) & (u <= gap_u2) & (np.abs(v - center_v) <= half_width)


def _severity(ratio: float) -> str:
    if ratio <= 0:
        return "NONE"
    if ratio <= 0.05:
        return "LIGHT"
    if ratio <= 0.10:
        return "MEDIUM"
    if ratio <= 0.20:
        return "HEAVY"
    if ratio <= 0.35:
        return "LARGE_EXPERIMENTAL"
    return "VERY_LARGE_EXPERIMENTAL"


def _decision_confidence(axis_support: float, bridge_support: float, width_support: float, segmentation_quality: str, has_gap: bool) -> float:
    quality_score = {"GOOD": 0.95, "WARNING": 0.75, "INVALID": 0.35}.get(str(segmentation_quality).upper(), 0.70)
    if not has_gap:
        return round(float(0.65 * axis_support + 0.20 * quality_score + 0.15), 6)
    return round(float(0.25 * axis_support + 0.35 * bridge_support + 0.25 * width_support + 0.15 * quality_score), 6)


def _empty_decision(raw: np.ndarray, bbox: np.ndarray, *, reason: list[str], segmentation_quality: str) -> "CompletionDecision":
    return CompletionDecision(
        mode=AUTO_COMPLETION,
        status="REVIEW_REQUIRED",
        completion_required=False,
        execution_allowed=False,
        severity="NONE",
        completion_ratio=0.0,
        occlusion_detected=False,
        completion_mask=np.zeros_like(raw),
        occluder_mask=np.zeros_like(raw),
        estimated_full_fish_mask=np.zeros_like(raw),
        clean_visible_mask=np.zeros_like(raw),
        axis_origin_xy=(0.0, 0.0),
        axis_vector_xy=(1.0, 0.0),
        axis_endpoints_xy=((0.0, 0.0), (0.0, 0.0)),
        case_class="AMBIGUOUS",
        reason=reason,
        decision_confidence=0.0,
        sam_bbox_fill_ratio=float(raw.sum() / max(1, bbox.sum())),
        raw_component_count=0,
        clean_component_count=0,
        noise_removed_pixels=0,
        axis_angle_deg=0.0,
        axis_length_px=0.0,
        structural_gap_count=0,
        candidate_area_pixels=0,
        completion_area_pixels=0,
        completion_region_count=0,
        ring_candidate_rejected=True,
    )


def decide_completion(
    raw_mask: np.ndarray,
    detector_bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    detector_assessment: str = "",
    segmentation_quality: str = "",
    original_image: Any | None = None,
) -> CompletionDecision:
    """Infer a structural envelope from visible fish geometry, never labels."""
    del original_image
    height, width = image_size
    raw = np.asarray(raw_mask, dtype=bool)
    if raw.shape != (height, width):
        raise ValueError("raw_mask shape must match image_size")

    bbox = _bbox_mask(raw.shape, detector_bbox)
    clean, component_stats = _clean_visible_mask(raw)
    if not clean.any():
        return _empty_decision(raw, bbox, reason=["NO_VISIBLE_FISH_MASK"], segmentation_quality=segmentation_quality)

    origin, axis, angle, axis_length = _principal_axis(clean)
    profile = _profile(clean, origin, axis)
    profile["mask_shape"] = raw.shape
    segments = _segments(profile["occupied"])
    gaps: list[tuple[int, int]] = []
    if len(segments) >= 2:
        for left, right in zip(segments, segments[1:]):
            gap_start, gap_end = left[1] + 1, right[0] - 1
            gap_bins = gap_end - gap_start + 1
            support = min(left[1] - left[0] + 1, right[1] - right[0] + 1)
            if gap_bins > 0 and support >= STRUCTURAL_THRESHOLDS["min_support_bins"] and gap_bins / max(1, profile["bins"]) <= STRUCTURAL_THRESHOLDS["max_internal_gap_fraction"]:
                gaps.append((gap_start, gap_end))

    widths = profile["widths"]
    occupied = profile["occupied"]
    local_defects: list[tuple[int, int]] = []
    for index in range(2, len(widths) - 2):
        if not occupied[index] or not occupied[index - 2] or not occupied[index + 2]:
            continue
        expected = (widths[index - 2] + widths[index + 2]) / 2.0
        if expected > 4 and widths[index] < expected * STRUCTURAL_THRESHOLDS["local_width_drop_ratio"]:
            local_defects.append((index, index))

    all_gaps = gaps + local_defects
    estimated = clean.copy()
    for start, end in all_gaps:
        estimated |= _interpolate_band(profile, start, end)
    candidate = estimated & ~clean & bbox
    completion_area = int(candidate.sum())
    final_area = int(clean.sum() + completion_area)
    completion_ratio = completion_area / max(1, final_area)
    completion_regions = len(_components(candidate))
    axis_support = min(1.0, axis_length / max(1.0, max(height, width)))
    bridge_support = 1.0 if gaps else 0.35
    width_support = 0.85 if all_gaps else 0.35
    confidence = _decision_confidence(axis_support, bridge_support, width_support, segmentation_quality, bool(all_gaps))

    reasons: list[str] = []
    if detector_assessment:
        reasons.append("DETECTOR_ASSESSMENT_RECORDED")
    if component_stats["noise_removed_pixels"]:
        reasons.append("SMALL_COMPONENTS_REMOVED")
    if gaps:
        reasons.append("INTERNAL_STRUCTURAL_GAP")
    if local_defects:
        reasons.append("LOCAL_CONTOUR_DEFECT")
    if not all_gaps:
        reasons.append("NO_STRUCTURAL_GAP")
    if float(clean.sum() / max(1, int(bbox.sum()))) < 0.50:
        reasons.append("LOW_SAM_BBOX_FILL_DEBUG_ONLY")
    reasons.append("RING_CANDIDATE_REJECTED")

    assessment_ambiguous = "incomplete" in str(detector_assessment).lower() and axis_length < max(height, width) * 0.35
    if assessment_ambiguous:
        status, case_class = "REVIEW_REQUIRED", "AMBIGUOUS"
        reasons.append("INSUFFICIENT_STRUCTURAL_SPAN")
    elif not all_gaps and component_stats["clean_component_count"] == 1:
        status, case_class = "NOT_REQUIRED", "COMPLETE"
    elif not candidate.any() or confidence < STRUCTURAL_THRESHOLDS["min_decision_confidence"]:
        status, case_class = "REVIEW_REQUIRED", "COMPLEX_BACKGROUND" if component_stats["clean_component_count"] > 1 else "AMBIGUOUS"
    elif completion_ratio > 0.20:
        status, case_class = "LARGE_EXPERIMENTAL", "LARGE_MISSING"
    elif gaps:
        status, case_class = "MASK_READY", "STRUCTURAL_OCCLUSION"
    else:
        status, case_class = "MASK_READY", "LOCAL_MISSING"

    required = status in {"MASK_READY", "LARGE_EXPERIMENTAL"}
    execution_allowed = bool(
        required
        and confidence >= (STRUCTURAL_THRESHOLDS["min_large_experimental_confidence"] if status == "LARGE_EXPERIMENTAL" else STRUCTURAL_THRESHOLDS["min_decision_confidence"])
        and completion_regions <= STRUCTURAL_THRESHOLDS["max_pathological_completion_regions"]
        and not bool((candidate & clean).any())
    )
    if status == "LARGE_EXPERIMENTAL" and not execution_allowed:
        status, required = "REVIEW_REQUIRED", False
    if status == "NOT_REQUIRED":
        execution_allowed = False
    if not execution_allowed and status == "REVIEW_REQUIRED":
        reasons.append("STRUCTURAL_CONFIDENCE_BELOW_EXECUTION_THRESHOLD")
    reasons.append("STRUCTURAL_EVIDENCE_SUFFICIENT" if execution_allowed else ("NO_COMPLETION_REQUIRED" if status == "NOT_REQUIRED" else "STRUCTURAL_REVIEW_REQUIRED"))

    return CompletionDecision(
        mode=AUTO_COMPLETION,
        status=status,
        completion_required=required,
        execution_allowed=execution_allowed,
        severity=_severity(completion_ratio),
        completion_ratio=round(completion_ratio, 6),
        occlusion_detected=bool(gaps),
        completion_mask=candidate,
        occluder_mask=np.zeros_like(raw),
        estimated_full_fish_mask=estimated,
        clean_visible_mask=clean,
        axis_origin_xy=(float(origin[0]), float(origin[1])),
        axis_vector_xy=(float(axis[0]), float(axis[1])),
        axis_endpoints_xy=((float(origin[0] - axis[0] * axis_length / 2), float(origin[1] - axis[1] * axis_length / 2)), (float(origin[0] + axis[0] * axis_length / 2), float(origin[1] + axis[1] * axis_length / 2))),
        case_class=case_class,
        reason=reasons,
        decision_confidence=confidence,
        sam_bbox_fill_ratio=float(clean.sum() / max(1, int(bbox.sum()))),
        raw_component_count=component_stats["raw_component_count"],
        clean_component_count=component_stats["clean_component_count"],
        noise_removed_pixels=component_stats["noise_removed_pixels"],
        axis_angle_deg=angle,
        axis_length_px=axis_length,
        structural_gap_count=len(all_gaps),
        candidate_area_pixels=completion_area,
        completion_area_pixels=completion_area,
        completion_region_count=completion_regions,
        ring_candidate_rejected=True,
    )
