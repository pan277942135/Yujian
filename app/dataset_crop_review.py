"""BBox-only review endpoints for immutable Frozen Dataset rows."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.frozen_crop_bridge import _read_uri, load_frozen_dataset
from app.models import DatasetCropReview, DatasetCropReviewEvent

router = APIRouter(prefix="/api/dataset-crop-review", tags=["dataset-crop-review"])


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _box(value: Any) -> list[float] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(0 <= item <= 1 for item in result) or result[2] <= 0 or result[3] <= 0:
        return None
    if result[0] + result[2] > 1.00001 or result[1] + result[3] > 1.00001:
        return None
    return result


class DatasetCropReviewUpdate(BaseModel):
    decision: str = Field(default="ACCEPTED", max_length=32)
    accepted_bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    reviewer: str = Field(default="crop-review", max_length=256)
    notes: str | None = Field(default=None, max_length=4000)


def _ensure_rows(db: Session, dataset_version: str) -> list[dict[str, Any]]:
    loaded = load_frozen_dataset(db, dataset_version)
    existing = {
        row.image_id: row
        for row in db.scalars(
            select(DatasetCropReview).where(DatasetCropReview.source_dataset_version == dataset_version)
        ).all()
    }
    for base in loaded["rows"]:
        if base["image_id"] in existing:
            continue
        row = DatasetCropReview(
            source_dataset_version=dataset_version,
            source_manifest_uri=base["source_manifest_uri"],
            image_id=base["image_id"],
            source_image_id=base["source_image_id"],
            source_image_gcs_uri=base["source_image_gcs_uri"],
            species_key=base["species_key"],
            species_name=base["species_name"],
            class_index=base["class_index"],
            split=base["split"],
            group_id=base.get("group_id") or None,
            review_status="BBOX_REQUIRED",
        )
        db.add(row)
        existing[base["image_id"]] = row
    db.commit()
    return loaded["rows"]


def _item(base: dict[str, Any], review: DatasetCropReview | None) -> dict[str, Any]:
    return {
        **base,
        "source_type": "FROZEN_DATASET",
        "candidate_bbox": _box(review.candidate_bbox_json) if review else None,
        "accepted_bbox": _box(review.accepted_bbox_json) if review else None,
        "bbox_source": review.bbox_source if review else None,
        "detector_version": review.detector_version if review else None,
        "status": review.review_status if review else "BBOX_REQUIRED",
        "reviewer": review.reviewer if review else None,
        "reviewed_at": review.reviewed_at.isoformat() if review and review.reviewed_at else None,
        "media_url": f"/api/dataset-crop-review/{base['source_dataset_version']}/{base['image_id']}/image",
    }


def _populate_candidate(review: DatasetCropReview, base: dict[str, Any], db: Session) -> None:
    if review.candidate_bbox_json:
        return
    try:
        from io import BytesIO
        from PIL import Image
        from app.detector_runtime import detect
        data, _ = _read_uri(base["source_image_gcs_uri"])
        with Image.open(BytesIO(data)) as source_image:
            run = detect(source_image.convert("RGB"))
        review.detector_version = run.model_version
        if run.detections:
            box = run.detections[0].box
            review.candidate_bbox_json = json.dumps([box.x1, box.y1, box.width, box.height], separators=(",", ":"))
        db.add(review)
        db.commit()
    except Exception:
        return


@router.get("/{dataset_version}/summary")
def summary(dataset_version: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    rows = _ensure_rows(db, dataset_version)
    counts_rows = db.execute(
        select(DatasetCropReview.review_status, func.count())
        .where(DatasetCropReview.source_dataset_version == dataset_version)
        .group_by(DatasetCropReview.review_status)
    ).all()
    counts = {str(status): int(count) for status, count in counts_rows}
    accepted = counts.get("ACCEPTED", 0) + counts.get("TRAINING_READY", 0)
    candidate_bbox_count = int(
        db.scalar(
            select(func.count())
            .select_from(DatasetCropReview)
            .where(
                DatasetCropReview.source_dataset_version == dataset_version,
                DatasetCropReview.candidate_bbox_json.is_not(None),
            )
        )
        or 0
    )
    accepted_bbox_count = int(
        db.scalar(
            select(func.count())
            .select_from(DatasetCropReview)
            .where(
                DatasetCropReview.source_dataset_version == dataset_version,
                DatasetCropReview.accepted_bbox_json.is_not(None),
            )
        )
        or 0
    )
    return {
        "dataset_version": dataset_version,
        "source_type": "FROZEN_DATASET",
        "total_images": len(rows),
        "bbox_required": max(len(rows) - accepted, 0),
        "accepted": accepted,
        "rejected": counts.get("REJECTED", 0),
        "candidate_bbox_count": candidate_bbox_count,
        "accepted_bbox_count": accepted_bbox_count,
        "counts": counts,
    }


@router.get("/{dataset_version}/items")
def items(
    dataset_version: str,
    status: str = Query(default="BBOX_REQUIRED", max_length=32),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    # Direct unit callers do not receive FastAPI's coerced query values.
    if not isinstance(page, int):
        page = 1
    if not isinstance(page_size, int):
        page_size = 50
    rows = _ensure_rows(db, dataset_version)
    reviews = {
        row.image_id: row
        for row in db.scalars(
            select(DatasetCropReview).where(DatasetCropReview.source_dataset_version == dataset_version)
        ).all()
    }
    if not isinstance(status, str):
        status = "BBOX_REQUIRED"
    normalized = status.strip().upper()
    allowed = {"BBOX_REQUIRED", "ACCEPTED", "REJECTED", "TRAINING_READY", "ALL"}
    if normalized not in allowed:
        raise HTTPException(status_code=400, detail="invalid dataset crop review status")
    selected = [
        _item(base, reviews.get(base["image_id"]))
        for base in rows
        if normalized == "ALL" or (reviews.get(base["image_id"]) is not None and reviews[base["image_id"]].review_status == normalized)
    ]
    total = len(selected)
    start = (page - 1) * page_size
    page_items = selected[start : start + page_size]
    row_by_image_id = {row["image_id"]: row for row in rows}
    for item in page_items:
        review = reviews.get(item["image_id"])
        if review:
            base = row_by_image_id[item["image_id"]]
            _populate_candidate(review, base, db)
            item.update(_item(base, review))
    return {"dataset_version": dataset_version, "source_type": "FROZEN_DATASET", "total": total, "page": page, "page_size": page_size, "items": page_items}


@router.patch("/{dataset_version}/{image_id}")
def update(
    dataset_version: str,
    image_id: str,
    payload: DatasetCropReviewUpdate,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    rows = _ensure_rows(db, dataset_version)
    base = next((row for row in rows if row["image_id"] == image_id), None)
    if not base:
        raise HTTPException(status_code=404, detail="image not found in frozen dataset")
    decision = payload.decision.strip().upper()
    if decision == "SKIP":
        decision = "BBOX_REQUIRED"
    if decision not in {"BBOX_REQUIRED", "ACCEPTED", "REJECTED", "TRAINING_READY"}:
        raise HTTPException(status_code=400, detail="invalid dataset crop review decision")
    box = _box(payload.accepted_bbox)
    if decision in {"ACCEPTED", "TRAINING_READY"} and box is None:
        raise HTTPException(status_code=400, detail={"error": "ACCEPTED_BBOX_REQUIRED", "reason": "Frozen Dataset 只需人工确认 bbox"})
    row = db.scalar(
        select(DatasetCropReview).where(
            DatasetCropReview.source_dataset_version == dataset_version,
            DatasetCropReview.image_id == image_id,
        )
    )
    before = {"status": row.review_status, "accepted_bbox_json": row.accepted_bbox_json}
    row.accepted_bbox_json = json.dumps(box, separators=(",", ":")) if box is not None else None
    row.bbox_source = "accepted_review" if box is not None else None
    row.review_status = decision
    row.reviewer = payload.reviewer.strip() or "crop-review"
    row.reviewed_at = _now()
    row.updated_at = _now()
    db.add(
        DatasetCropReviewEvent(
            source_dataset_version=dataset_version,
            image_id=image_id,
            action="dataset_crop_bbox_review",
            reviewer=row.reviewer,
            before_json=json.dumps(before, ensure_ascii=False),
            after_json=json.dumps({"status": decision, "accepted_bbox": box}, ensure_ascii=False),
        )
    )
    db.commit()
    return _item(base, row)


@router.get("/{dataset_version}/{image_id}/image")
def image(dataset_version: str, image_id: str, db: Session = Depends(get_db)) -> Response:
    loaded = load_frozen_dataset(db, dataset_version)
    base = next((row for row in loaded["rows"] if row["image_id"] == image_id), None)
    if not base:
        raise HTTPException(status_code=404, detail="image not found in frozen dataset")
    try:
        data, _ = _read_uri(base["source_image_gcs_uri"])
    except Exception as exc:
        raise HTTPException(status_code=404, detail="source image unavailable") from exc
    media_type = "image/jpeg"
    uri = base["source_image_gcs_uri"].lower()
    if uri.endswith(".png"):
        media_type = "image/png"
    elif uri.endswith(".webp"):
        media_type = "image/webp"
    return Response(content=data, media_type=media_type)


__all__ = ["router"]
