"""Protected read-only API for historical crop parity audit."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.crop_audit import ACCEPTED_REVIEW_STATUSES, AUDIT_VERSION, audit_review_record
from app.crop_contract import DEFAULT_EXPAND_RATIO, JPEG_QUALITY
from app.db import get_db
from app.frozen_crop_bridge import _read_uri
from app.models import DatasetCropReview

router = APIRouter(prefix="/api/crop-audit", tags=["crop-audit"])


@router.get("/historical")
def historical_crop_audit(
    dataset_version: str | None = Query(default=None, max_length=128),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=200),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Compare persisted accepted crops with the current canonical crop bytes.

    This endpoint is intentionally read-only: it never calls ``db.commit`` and
    never writes or deletes a GCS object.  It only reads accepted review rows,
    source images and historical crop artifacts.
    """

    filters = [DatasetCropReview.review_status.in_(sorted(ACCEPTED_REVIEW_STATUSES))]
    if dataset_version:
        filters.append(DatasetCropReview.source_dataset_version == dataset_version)

    total = int(
        db.scalar(
            select(func.count())
            .select_from(DatasetCropReview)
            .where(*filters)
        )
        or 0
    )
    rows = db.scalars(
        select(DatasetCropReview)
        .where(*filters)
        .order_by(DatasetCropReview.source_dataset_version, DatasetCropReview.image_id)
        .offset(offset)
        .limit(limit)
    ).all()

    items = [audit_review_record(row, _read_uri) for row in rows]
    counts: dict[str, int] = {}
    for item in items:
        status = str(item.get("audit_status") or "UNKNOWN")
        counts[status] = counts.get(status, 0) + 1

    return {
        "audit_version": AUDIT_VERSION,
        "read_only": True,
        "canonical_contract": {
            "exif_transpose": True,
            "color_mode": "RGB",
            "accepted_bbox_format": "normalized_xywh",
            "expand_ratio": DEFAULT_EXPAND_RATIO,
            "left_top_rounding": "floor",
            "right_bottom_rounding": "ceil",
            "clamp_to_image_bounds": True,
            "jpeg_quality": JPEG_QUALITY,
        },
        "dataset_version": dataset_version,
        "total": total,
        "offset": offset,
        "limit": limit,
        "returned": len(items),
        "has_more": offset + len(items) < total,
        "page_counts": counts,
        "items": items,
    }


__all__ = ["router"]
