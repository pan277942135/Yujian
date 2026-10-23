"""Source-bound segmentation adapter for the product recognition route.

The product route owns the detector result and the uploaded bytes.  This
adapter deliberately consumes both instead of running a second detector or
accepting a client supplied box.  SAM output is evidence for review only:
the experimental quality heuristic is not a production qualification gate.
"""

from __future__ import annotations

import hashlib
import io
import logging
from typing import Any

from PIL import Image, ImageOps

from app.recognition_pipeline import BBox
from app.segmentation.mask_generator import SegmentationModelNotConfigured
from app.segmentation.service import FishSegmentationResult, generate_fish_cutout


logger = logging.getLogger(__name__)


def _bbox(value: Any) -> BBox | None:
    if not isinstance(value, dict):
        return None
    try:
        box = BBox(
            float(value["x1"]),
            float(value["y1"]),
            float(value["x2"]),
            float(value["y2"]),
        ).normalized()
    except (KeyError, TypeError, ValueError):
        return None
    if box.width <= 0 or box.height <= 0:
        return None
    return box


def _binding(*, data: bytes, image: Image.Image, bbox: BBox | None, detector: dict, session_id: str) -> dict:
    return {
        "session_id": session_id,
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "source_width": image.width,
        "source_height": image.height,
        "coordinate_space": "EXIF_TRANSPOSED_SOURCE_IMAGE_NORMALIZED",
        "bbox": {
            "x1": round(bbox.x1, 6),
            "y1": round(bbox.y1, 6),
            "x2": round(bbox.x2, 6),
            "y2": round(bbox.y2, 6),
        }
        if bbox
        else None,
        "detector_model_version": detector.get("model_version"),
        "detector_onnx_sha256": detector.get("onnx_sha256"),
        "bbox_source": "SERVER_DETECTOR_QUALITY_GATE_PRIMARY",
    }


def _candidate(result: FishSegmentationResult, binding: dict) -> dict:
    # Do not serialize mask pixels or the cutout.  A good heuristic result is
    # still an unqualified candidate until GT IoU/Dice/boundary and visual
    # sign-off are recorded against this exact source and detector box.
    return {
        "status": "CANDIDATE_REVIEW_ONLY",
        "reason_code": "SAM_HEURISTIC_NOT_PRODUCTION_QUALIFIED",
        "trust_state": "REVIEW_ONLY",
        "model_version": "SAM_VIT_B",
        "quality": result.quality.value,
        "quality_metrics": {
            "reason": result.reason,
            "mask_area_ratio": round(result.mask_area_ratio, 6),
            "edge_ratio": round(result.edge_ratio, 6),
            "connected_components": result.connected_components,
        },
        "processing_ms": result.processing_ms,
        "mask": None,
        "production_qualified": False,
        "source_binding": binding,
    }


def build_product_segmentation(*, data: bytes, result: dict, session_id: str) -> dict:
    """Build safe product segmentation metadata from one inference result.

    The function is intentionally fail-closed.  No SAM configuration yields a
    real detector-backed soft-region state, while a generated mask remains
    review-only and cannot drive contour/gold rendering.
    """

    detector = result.get("detector") or {}
    gate = result.get("quality_gate") or {}
    try:
        with Image.open(io.BytesIO(data)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (OSError, ValueError) as exc:
        logger.warning("product_segmentation_source_decode_failed type=%s", type(exc).__name__)
        return {
            "status": "NONE",
            "reason_code": "SOURCE_DECODE_FAILED",
            "trust_state": "NONE",
            "model_version": None,
            "quality": None,
            "mask": None,
            "production_qualified": False,
        }

    detections = detector.get("detections") or []
    primary = _bbox(gate.get("primary_bbox"))
    if primary is None and detections:
        # Compatibility for older deterministic seams only.  Production
        # _predict_bytes always supplies quality_gate.primary_bbox, which is
        # authoritative over detector list ordering.
        primary = _bbox((detections[0] or {}).get("bbox"))
    binding = _binding(data=data, image=image, bbox=primary, detector=detector, session_id=session_id)
    classifier_allowed = gate.get("classifier_allowed")
    if classifier_allowed is None:
        classifier_allowed = result.get("classification_ran") is True
    if primary is None or not classifier_allowed:
        return {
            "status": "NONE",
            "reason_code": "DETECTOR_QUALITY_GATE_BLOCKED",
            "trust_state": "NONE",
            "model_version": None,
            "quality": None,
            "mask": None,
            "production_qualified": False,
            "source_binding": binding,
        }

    try:
        segmentation = generate_fish_cutout(image, primary)
    except SegmentationModelNotConfigured:
        return {
            "status": "TRUSTED_BBOX_ONLY",
            "reason_code": "SAM_NOT_CONFIGURED",
            "trust_state": "BBOX_ONLY",
            "model_version": None,
            "quality": None,
            "mask": None,
            "production_qualified": False,
            "source_binding": binding,
        }
    except (RuntimeError, OSError, ValueError) as exc:
        logger.warning("product_segmentation_unavailable type=%s", type(exc).__name__)
        return {
            "status": "TRUSTED_BBOX_ONLY",
            "reason_code": "SAM_RUNTIME_UNAVAILABLE",
            "trust_state": "BBOX_ONLY",
            "model_version": None,
            "quality": None,
            "mask": None,
            "production_qualified": False,
            "source_binding": binding,
        }
    return _candidate(segmentation, binding)


__all__ = ["build_product_segmentation"]
