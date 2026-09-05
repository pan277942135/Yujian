"""Read-only historical crop parity audit against the canonical crop contract."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable

from app.crop_contract import DEFAULT_EXPAND_RATIO, JPEG_QUALITY, canonical_crop

AUDIT_VERSION = "HISTORICAL_CROP_AUDIT_V1"
ACCEPTED_REVIEW_STATUSES = {"ACCEPTED", "TRAINING_READY"}


def _value(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _parse_bbox(raw: Any) -> list[float] | None:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        values = [float(value) for value in raw]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        return None
    if values[2] <= 0 or values[3] <= 0:
        return None
    if values[0] + values[2] > 1.00001 or values[1] + values[3] > 1.00001:
        return None
    return values


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compare_crop_bytes(source_bytes: bytes, existing_crop_bytes: bytes, accepted_bbox: list[float]) -> dict[str, Any]:
    """Compare an existing historical crop with a freshly generated canonical crop."""

    canonical = canonical_crop(source_bytes, accepted_bbox)
    existing_sha = _sha256(existing_crop_bytes)
    canonical_sha = _sha256(canonical.jpeg_bytes)
    status = "MATCH" if existing_sha == canonical_sha else "REBUILD_REQUIRED"
    return {
        "audit_status": status,
        "rebuild_required": status == "REBUILD_REQUIRED",
        "manual_fix_required": False,
        "existing_sha256": existing_sha,
        "canonical_sha256": canonical_sha,
        "existing_bytes": len(existing_crop_bytes),
        "canonical_bytes": len(canonical.jpeg_bytes),
        "canonical_pixel_box": list(canonical.pixel_box),
        "canonical_width": canonical.width,
        "canonical_height": canonical.height,
    }


def audit_review_record(
    record: Any,
    read_uri: Callable[[str], tuple[bytes, Any]],
) -> dict[str, Any]:
    """Audit one accepted DatasetCropReview without mutating DB or GCS."""

    dataset_version = str(_value(record, "source_dataset_version") or "")
    image_id = str(_value(record, "image_id") or "")
    crop_status = str(_value(record, "crop_status") or "")
    base = {
        "source_dataset_version": dataset_version,
        "image_id": image_id,
        "review_status": str(_value(record, "review_status") or ""),
        "crop_status": crop_status,
        "crop_status_drift": crop_status != "READY",
    }

    bbox = _parse_bbox(_value(record, "accepted_bbox_json"))
    if bbox is None:
        return {
            **base,
            "audit_status": "INVALID_BBOX",
            "rebuild_required": False,
            "manual_fix_required": True,
            "error": "accepted_bbox is missing or invalid",
        }

    crop_uri = str(_value(record, "crop_uri") or "").strip()
    if not crop_uri:
        return {
            **base,
            "accepted_bbox": bbox,
            "audit_status": "MISSING_CROP",
            "rebuild_required": True,
            "manual_fix_required": False,
            "error": "historical crop_uri is missing",
        }

    source_uri = str(_value(record, "source_image_gcs_uri") or "").strip()
    if not source_uri:
        return {
            **base,
            "accepted_bbox": bbox,
            "audit_status": "SOURCE_UNAVAILABLE",
            "rebuild_required": False,
            "manual_fix_required": True,
            "error": "source image URI is missing",
        }

    try:
        source_bytes, _ = read_uri(source_uri)
    except Exception as exc:
        return {
            **base,
            "accepted_bbox": bbox,
            "audit_status": "SOURCE_UNAVAILABLE",
            "rebuild_required": False,
            "manual_fix_required": True,
            "error": str(exc)[:500],
        }

    try:
        existing_crop_bytes, _ = read_uri(crop_uri)
    except Exception as exc:
        return {
            **base,
            "accepted_bbox": bbox,
            "audit_status": "CROP_UNAVAILABLE",
            "rebuild_required": True,
            "manual_fix_required": False,
            "error": str(exc)[:500],
        }

    try:
        comparison = compare_crop_bytes(source_bytes, existing_crop_bytes, bbox)
    except Exception as exc:
        return {
            **base,
            "accepted_bbox": bbox,
            "audit_status": "AUDIT_ERROR",
            "rebuild_required": False,
            "manual_fix_required": True,
            "error": str(exc)[:500],
        }

    return {**base, "accepted_bbox": bbox, **comparison}


__all__ = [
    "ACCEPTED_REVIEW_STATUSES",
    "AUDIT_VERSION",
    "audit_review_record",
    "compare_crop_bytes",
]
