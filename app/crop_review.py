"""Human review bridge for normal Batch images entering the crop pipeline.

The App inference contract already has a review endpoint for ``InferenceAsset``.
Normal uploaded batches use ``ImageAsset`` instead, so this small bridge keeps
the two sources explicit without promoting a detector/presence suggestion to a
training label.  Only an explicit reviewer decision writes
``accepted_bbox_json``.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from starlette.requests import Request

from app.db import get_db
from app.models import Batch, BatchCropReview, ImageAsset, ReviewEvent
from app.presence import FishPresenceResult


router = APIRouter(tags=["crop-review"])
templates = Jinja2Templates(directory="app/templates")

REVIEW_STATUSES = {"REVIEW_REQUIRED", "ACCEPTED", "REJECTED", "TRAINING_READY"}
ACCEPTED_STATUSES = {"ACCEPTED", "TRAINING_READY"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _bbox(value: Any) -> list[float] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    if isinstance(value, dict):
        if "bbox" in value:
            value = value["bbox"]
        elif {"x", "y", "width", "height"} <= set(value):
            value = [value["x"], value["y"], value["width"], value["height"]]
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in values):
        return None
    if values[2] <= 0 or values[3] <= 0 or values[0] + values[2] > 1.00001 or values[1] + values[3] > 1.00001:
        return None
    return values


def _vertices_bbox(vertices: Any) -> list[float] | None:
    if not isinstance(vertices, list) or not vertices:
        return None
    try:
        xs = [float(item.get("x", 0.0) or 0.0) for item in vertices if isinstance(item, dict)]
        ys = [float(item.get("y", 0.0) or 0.0) for item in vertices if isinstance(item, dict)]
    except (TypeError, ValueError):
        return None
    if not xs or not ys:
        return None
    left, right = max(0.0, min(xs)), min(1.0, max(xs))
    top, bottom = max(0.0, min(ys)), min(1.0, max(ys))
    return _bbox([left, top, right - left, bottom - top])


def _candidate_boxes(presence: FishPresenceResult | None) -> list[dict[str, Any]]:
    if not presence or not presence.evidence_json:
        return []
    try:
        evidence = json.loads(presence.evidence_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    objects = evidence.get("objects") if isinstance(evidence, dict) else []
    result: list[dict[str, Any]] = []
    for item in objects or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip().lower()
        # Vision's saved evidence contains contextual objects as well.  Fish
        # body candidates are preferred, while unnamed boxes remain visible
        # for a human when a provider omitted the label.
        if name and "fish" not in name and name not in {"carp", "fish", "ray-finned fish"}:
            continue
        box = _bbox(item.get("bbox")) or _vertices_bbox(item.get("vertices"))
        if box is None:
            continue
        result.append({"bbox": box, "confidence": float(item.get("score", 0.0) or 0.0), "label": item.get("name")})
    result.sort(key=lambda row: float(row.get("confidence", 0.0) or 0.0), reverse=True)
    return result


def _review_row(db: Session, image: ImageAsset) -> BatchCropReview | None:
    return db.scalar(select(BatchCropReview).where(BatchCropReview.image_asset_id == image.id))


def _item(db: Session, image: ImageAsset, review: BatchCropReview | None, presence: FishPresenceResult | None) -> dict[str, Any]:
    candidates = _candidate_boxes(presence)
    accepted = _bbox(review.accepted_bbox_json) if review else None
    return {
        "batch_id": image.batch_id,
        "image_id": image.image_id,
        "file_name": image.file_name,
        "media_url": f"/media/{image.batch_id}/{image.image_id}",
        "source_image": image.gcs_uri,
        "suggested_species": image.truth_species or image.claimed_species,
        "candidate_bbox": candidates[0]["bbox"] if candidates else None,
        "candidate_boxes": candidates,
        "candidate_bbox_source": "presence_detector" if candidates else None,
        "accepted_bbox": accepted,
        "accepted_species_key": review.accepted_species_key if review else None,
        "accepted_species_name": review.accepted_species_name if review else None,
        "detector_version": (review.detector_version if review else None) or (presence.model_version if presence else None),
        "status": (review.status if review else "REVIEW_REQUIRED"),
        "reviewer": review.reviewer if review else None,
        "reviewed_at": review.reviewed_at.isoformat() if review and review.reviewed_at else None,
        "notes": review.notes if review else None,
    }


class CropReviewUpdate(BaseModel):
    decision: str = Field(default="ACCEPTED", max_length=32)
    accepted_bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    accepted_bbox_json: str | None = Field(default=None, max_length=512)
    species_key: str | None = Field(default=None, max_length=128)
    species_name: str | None = Field(default=None, max_length=128)
    accepted_species_key: str | None = Field(default=None, max_length=128)
    accepted_species_name: str | None = Field(default=None, max_length=128)
    reviewer: str = Field(default="crop-review", max_length=256)
    notes: str | None = Field(default=None, max_length=4000)


def _find_image(db: Session, batch_id: str, image_id: str) -> ImageAsset:
    image = db.scalar(select(ImageAsset).where(ImageAsset.batch_id == batch_id, ImageAsset.image_id == image_id))
    if not image:
        raise HTTPException(status_code=404, detail="image not found in batch")
    return image


@router.get("/crop-review", response_class=HTMLResponse)
def crop_review_page(request: Request):
    return templates.TemplateResponse(request=request, name="crop_review.html", context={})


@router.get("/api/crop-review/{batch_id}/summary")
def crop_review_summary(batch_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    if not db.get(Batch, batch_id):
        raise HTTPException(status_code=404, detail="batch not found")
    total = db.scalar(select(func.count()).select_from(ImageAsset).where(ImageAsset.batch_id == batch_id)) or 0
    rows = db.execute(
        select(BatchCropReview.status, func.count())
        .where(BatchCropReview.batch_id == batch_id)
        .group_by(BatchCropReview.status)
    ).all()
    counts = {str(status): int(count) for status, count in rows}
    reviewed = sum(counts.get(status, 0) for status in ACCEPTED_STATUSES)
    return {
        "batch_id": batch_id,
        "total_images": int(total),
        "reviewed": reviewed,
        "review_required": max(int(total) - counts.get("ACCEPTED", 0) - counts.get("TRAINING_READY", 0) - counts.get("REJECTED", 0), 0),
        "accepted": counts.get("ACCEPTED", 0),
        "training_ready": counts.get("TRAINING_READY", 0),
        "rejected": counts.get("REJECTED", 0),
        "counts": counts,
    }


@router.get("/api/crop-review/{batch_id}/items")
def crop_review_items(
    batch_id: str,
    status: str = Query(default="REVIEW_REQUIRED", max_length=32),
    limit: int = Query(default=24, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if not db.get(Batch, batch_id):
        raise HTTPException(status_code=404, detail="batch not found")
    # Calling the handler directly in unit tests does not run FastAPI's
    # dependency/default injection, so normalise Query marker defaults.
    if not isinstance(status, str):
        status = "REVIEW_REQUIRED"
    if not isinstance(limit, int):
        limit = 24
    if not isinstance(offset, int):
        offset = 0
    normalized = status.strip().upper()
    if normalized not in REVIEW_STATUSES | {"ALL"}:
        raise HTTPException(status_code=400, detail="invalid crop review status")
    images = db.scalars(select(ImageAsset).where(ImageAsset.batch_id == batch_id).order_by(ImageAsset.id)).all()
    image_ids = [image.id for image in images]
    reviews = {
        row.image_asset_id: row
        for row in db.scalars(select(BatchCropReview).where(BatchCropReview.batch_id == batch_id)).all()
    }
    presences = {}
    if image_ids:
        presences = {
            row.image_asset_id: row
            for row in db.scalars(select(FishPresenceResult).where(FishPresenceResult.image_asset_id.in_(image_ids))).all()
        }
    selected = [
        _item(db, image, reviews.get(image.id), presences.get(image.id))
        for image in images
        if normalized == "ALL" or reviews.get(image.id, None) is None and normalized == "REVIEW_REQUIRED" or reviews.get(image.id) and reviews[image.id].status == normalized
    ]
    return {"batch_id": batch_id, "status": normalized, "total": len(selected), "offset": offset, "limit": limit, "items": selected[offset : offset + limit]}


@router.patch("/api/crop-review/{batch_id}/{image_id}")
def update_crop_review(batch_id: str, image_id: str, payload: CropReviewUpdate, db: Session = Depends(get_db)) -> dict[str, Any]:
    image = _find_image(db, batch_id, image_id)
    decision = payload.decision.strip().upper()
    if decision == "SKIP":
        decision = "REVIEW_REQUIRED"
    if decision not in REVIEW_STATUSES:
        raise HTTPException(status_code=400, detail="decision must be REVIEW_REQUIRED, ACCEPTED, REJECTED, or TRAINING_READY")
    box = _bbox(payload.accepted_bbox if payload.accepted_bbox is not None else payload.accepted_bbox_json)
    if decision in ACCEPTED_STATUSES and box is None:
        raise HTTPException(status_code=400, detail={"error": "ACCEPTED_BBOX_REQUIRED", "reason": "请提供人工确认的 accepted_bbox"})
    species_key = (payload.species_key or payload.accepted_species_key or "").strip() or None
    species_name = (payload.species_name or payload.accepted_species_name or "").strip() or None
    if decision in ACCEPTED_STATUSES and not (species_key or species_name):
        raise HTTPException(status_code=400, detail={"error": "SPECIES_REQUIRED", "reason": "请明确确认真实鱼种"})
    row = _review_row(db, image)
    before = {
        "status": row.status if row else "REVIEW_REQUIRED",
        "accepted_bbox_json": row.accepted_bbox_json if row else None,
        "species_key": row.species_key if row else None,
        "species_name": row.species_name if row else None,
    }
    if row is None:
        row = BatchCropReview(batch_id=batch_id, image_asset_id=image.id, image_id=image.image_id)
        db.add(row)
        db.flush()
    presence = db.scalar(select(FishPresenceResult).where(FishPresenceResult.image_asset_id == image.id))
    candidates = _candidate_boxes(presence)
    row.candidate_bbox_json = json.dumps(candidates[0]["bbox"], separators=(",", ":")) if candidates else row.candidate_bbox_json
    row.detector_version = presence.model_version if presence else row.detector_version
    row.accepted_bbox_json = json.dumps(box, separators=(",", ":")) if box is not None else None
    row.species_key = species_key
    row.species_name = species_name
    row.status = decision
    row.reviewer = payload.reviewer.strip() or "crop-review"
    row.reviewed_at = _now()
    row.notes = payload.notes
    row.updated_at = _now()
    after = {
        "status": row.status,
        "accepted_bbox_json": row.accepted_bbox_json,
        "species_key": row.species_key,
        "species_name": row.species_name,
    }
    db.add(
        ReviewEvent(
            image_asset_id=image.id,
            action="crop_bbox_review",
            reviewer=row.reviewer,
            before_json=json.dumps(before, ensure_ascii=False),
            after_json=json.dumps(after, ensure_ascii=False),
        )
    )
    db.commit()
    db.refresh(row)
    return _item(db, image, row, presence)


__all__ = ["CropReviewUpdate", "crop_review_items", "crop_review_summary", "router", "templates", "update_crop_review"]
