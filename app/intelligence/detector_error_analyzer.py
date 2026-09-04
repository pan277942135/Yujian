"""Detector error analysis for reviewed App inference records.

The analyzer is advisory only.  It reports issues where a reviewed reference
exists (or where the review explicitly says that a fish was present); it never
turns a candidate detector box into a label or starts a training job.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Iterable


def _bbox_xywh(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if isinstance(value, dict):
        if {"x", "y", "width", "height"} <= set(value):
            value = [value["x"], value["y"], value["width"], value["height"]]
        elif {"x1", "y1", "x2", "y2"} <= set(value):
            x1, y1, x2, y2 = (float(value[k]) for k in ("x1", "y1", "x2", "y2"))
            value = [x1, y1, x2 - x1, y2 - y1]
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x, y, width, height = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if min(x, y, width, height) < 0 or width <= 0 or height <= 0 or x + width > 1.00001 or y + height > 1.00001:
        return None
    return x, y, width, height


def _candidate_boxes(record: dict[str, Any]) -> list[tuple[float, float, float, float]]:
    detection = (
        record.get("detection")
        if isinstance(record.get("detection"), dict)
        else record.get("detector")
        if isinstance(record.get("detector"), dict)
        else {}
    )
    raw = detection.get("detections") or record.get("detections")
    if isinstance(raw, list):
        result = []
        for item in raw:
            if isinstance(item, dict):
                box = _bbox_xywh(item.get("candidate_bbox") or item.get("bbox"))
                if box:
                    result.append(box)
        if result:
            return result
    box = _bbox_xywh(detection.get("candidate_bbox") or record.get("candidate_bbox"))
    return [box] if box else []


def _accepted_box(record: dict[str, Any]) -> tuple[float, float, float, float] | None:
    return _bbox_xywh(record.get("accepted_bbox") or record.get("accepted_bbox_json") or ((record.get("review") or {}).get("accepted_bbox") if isinstance(record.get("review"), dict) else None))


def _area(box: tuple[float, float, float, float]) -> float:
    return box[2] * box[3]


def _iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    ix1, iy1 = max(lx, rx), max(ly, ry)
    ix2, iy2 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _area(left) + _area(right) - intersection
    return intersection / union if union else 0.0


def analyze_detector_errors(
    records: Iterable[dict[str, Any]],
    *,
    detector_version: str | None = None,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    versions: Counter[str] = Counter()
    for record in records:
        image_id = str(record.get("image_id") or "").strip()
        detection = (
            record.get("detection")
            if isinstance(record.get("detection"), dict)
            else record.get("detector")
            if isinstance(record.get("detector"), dict)
            else {}
        )
        version = str(detector_version or detection.get("detector_version") or record.get("detector_version") or "unknown")
        versions[version] += 1
        boxes = _candidate_boxes(record)
        accepted = _accepted_box(record)
        status = str(record.get("status") or record.get("review_status") or "").upper()
        human_present = bool(record.get("human_present") or record.get("reviewed_fish_present") or accepted)
        quality_text = " ".join(str(record.get(key) or "") for key in ("quality_reason", "quality", "notes")).lower()
        if human_present and not boxes:
            issues.append({"image_id": image_id, "issue_type": "missed_detection", "detector_version": version, "severity": "P0"})
            continue
        if accepted and boxes:
            best = max(boxes, key=lambda box: _iou(box, accepted))
            overlap = _iou(best, accepted)
            if overlap < iou_threshold:
                issues.append({"image_id": image_id, "issue_type": "bbox_misaligned", "detector_version": version, "iou": round(overlap, 6), "severity": "P1"})
            ratio = _area(best) / _area(accepted)
            if ratio > 1.5:
                issues.append({"image_id": image_id, "issue_type": "bbox_too_large", "detector_version": version, "area_ratio_to_reference": round(ratio, 6), "severity": "P1"})
            elif ratio < 0.5:
                issues.append({"image_id": image_id, "issue_type": "bbox_too_small", "detector_version": version, "area_ratio_to_reference": round(ratio, 6), "severity": "P1"})
        if accepted and len(boxes) > 1:
            issues.append({"image_id": image_id, "issue_type": "multiple_fish", "detector_version": version, "detection_count": len(boxes), "severity": "P1"})
        if record.get("occluded") or "occlu" in quality_text:
            issues.append({"image_id": image_id, "issue_type": "occlusion", "detector_version": version, "severity": "P2"})

    counts = Counter(issue["issue_type"] for issue in issues)
    resolved_version = detector_version or (versions.most_common(1)[0][0] if versions else "unknown")
    return {
        "report_version": "DETECTOR_ERROR_REPORT_V1",
        "detector_version": resolved_version,
        "sample_count": sum(versions.values()),
        "issue_count": len(issues),
        "error_counts": dict(sorted(counts.items())),
        "issues": issues,
        "improvement_task": {
            "task_type": "DETECTOR_IMPROVEMENT",
            "status": "OPEN" if issues else "NO_ACTION",
            "detector_version": resolved_version,
            "requirements": {"error_types": sorted(counts), "hard_cases": len(issues)},
            "safety": {
                "creates_batch": False,
                "changes_labels": False,
                "auto_freeze": False,
                "auto_train": False,
            },
        },
    }


__all__ = ["analyze_detector_errors"]
