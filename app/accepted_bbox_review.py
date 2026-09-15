"""Historical accepted_bbox confirmation for the cumulative approved pool.

New batches write this gate from the single/bulk review screens. This page is
the safe backfill path for legacy approved images that predate that gate.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from starlette.requests import Request

from app.crop_review import _candidate_boxes
from app.db import get_db
from app.detector_runtime import detect, normalize_android_source
from app.frozen_crop_bridge import _read_uri
from app.models import BatchCropReview, ImageAsset, ReviewEvent
from app.presence import FishPresenceResult

router = APIRouter(tags=["dataset-accepted-bbox"])
templates = Jinja2Templates(directory="app/templates")

ACCEPTED_STATUSES = {"ACCEPTED", "TRAINING_READY"}


class AcceptedBBoxUpdate(BaseModel):
    decision: str = Field(default="ACCEPTED", max_length=32)
    accepted_bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    reviewer: str = Field(default="dataset-accepted-bbox", max_length=256)
    notes: str | None = Field(default=None, max_length=4000)


class AcceptedBBoxBulkItem(AcceptedBBoxUpdate):
    batch_id: str
    image_id: str


class AcceptedBBoxBulk(BaseModel):
    items: list[AcceptedBBoxBulkItem] = Field(min_length=1, max_length=200)


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
    return [round(item, 6) for item in result]


def _reviewed(row: BatchCropReview | None) -> bool:
    return bool(row and row.status in ACCEPTED_STATUSES and _box(row.accepted_bbox_json))


def _candidate(row: BatchCropReview | None, presence: FishPresenceResult | None) -> list[float] | None:
    if row and _box(row.candidate_bbox_json):
        return _box(row.candidate_bbox_json)
    candidates = _candidate_boxes(presence)
    return _box(candidates[0].get("bbox")) if candidates else None


def _item(
    image: ImageAsset,
    review: BatchCropReview | None,
    presence: FishPresenceResult | None,
) -> dict[str, Any]:
    candidate = _candidate(review, presence)
    accepted = _box(review.accepted_bbox_json) if review else None
    status = review.status if review else "REVIEW_REQUIRED"
    return {
        "batch_id": image.batch_id,
        "image_id": image.image_id,
        "file_name": image.file_name,
        "media_url": f"/media/{image.batch_id}/{image.image_id}",
        "source_url": image.source_url,
        "claimed_species": image.claimed_species,
        "truth_species": image.truth_species,
        "species": (review.species_name if review else None) or image.truth_species,
        "review_status": image.review_status,
        "candidate_bbox": candidate,
        "accepted_bbox": accepted,
        "status": status,
        "bbox_status": "ACCEPTED" if _reviewed(review) else ("CANDIDATE" if candidate else "MISSING"),
        "detector_version": review.detector_version if review else (presence.model_version if presence else None),
        "reviewer": review.reviewer if review else None,
        "reviewed_at": review.reviewed_at.isoformat() if review and review.reviewed_at else None,
        "notes": review.notes if review else None,
    }


def _find_image(db: Session, batch_id: str, image_id: str) -> ImageAsset:
    image = db.scalar(
        select(ImageAsset).where(
            ImageAsset.batch_id == batch_id,
            ImageAsset.image_id == image_id,
            ImageAsset.review_status == "approved",
        )
    )
    if not image:
        raise HTTPException(status_code=404, detail="仅人工已通过图片可进入 accepted_bbox 补确认")
    return image


def _review_row(db: Session, image: ImageAsset) -> BatchCropReview | None:
    return db.scalar(select(BatchCropReview).where(BatchCropReview.image_asset_id == image.id))


def _apply(
    db: Session,
    image: ImageAsset,
    *,
    decision: str,
    accepted_bbox: list[float] | None,
    reviewer: str,
    notes: str | None,
) -> BatchCropReview:
    decision = decision.strip().upper()
    if decision == "ACCEPT":
        decision = "ACCEPTED"
    if decision not in {"ACCEPTED", "REVIEW_REQUIRED", "REJECTED", "TRAINING_READY"}:
        raise HTTPException(status_code=400, detail="decision 必须是 ACCEPTED、REVIEW_REQUIRED 或 REJECTED")
    box = _box(accepted_bbox)
    if decision in ACCEPTED_STATUSES and box is None:
        raise HTTPException(status_code=400, detail={"error": "ACCEPTED_BBOX_REQUIRED", "reason": "通过前必须确认 accepted_bbox"})
    row = _review_row(db, image)
    presence = db.scalar(select(FishPresenceResult).where(FishPresenceResult.image_asset_id == image.id))
    candidate = _candidate(row, presence)
    if row is None:
        row = BatchCropReview(batch_id=image.batch_id, image_asset_id=image.id, image_id=image.image_id)
        db.add(row)
        db.flush()
    if candidate:
        row.candidate_bbox_json = json.dumps(candidate, separators=(",", ":"))
    if presence:
        row.detector_version = presence.model_version
    row.accepted_bbox_json = json.dumps(box, separators=(",", ":")) if box else None
    row.species_name = (image.truth_species or "").strip() or None
    row.status = decision
    row.reviewer = reviewer.strip() or "dataset-accepted-bbox"
    row.reviewed_at = _now()
    row.notes = notes
    row.updated_at = _now()
    db.add(
        ReviewEvent(
            image_asset_id=image.id,
            action="accepted_bbox_backfill",
            reviewer=row.reviewer,
            before_json=None,
            after_json=json.dumps(
                {"status": row.status, "accepted_bbox_json": row.accepted_bbox_json, "species_name": row.species_name},
                ensure_ascii=False,
            ),
        )
    )
    return row


def _rows(db: Session) -> tuple[list[ImageAsset], dict[int, BatchCropReview], dict[int, FishPresenceResult]]:
    images = db.scalars(select(ImageAsset).where(ImageAsset.review_status == "approved").order_by(ImageAsset.batch_id, ImageAsset.id)).all()
    ids = [image.id for image in images]
    reviews = (
        {
            row.image_asset_id: row
            for row in db.scalars(select(BatchCropReview).where(BatchCropReview.image_asset_id.in_(ids))).all()
        }
        if ids
        else {}
    )
    presences = (
        {
            row.image_asset_id: row
            for row in db.scalars(select(FishPresenceResult).where(FishPresenceResult.image_asset_id.in_(ids))).all()
        }
        if ids
        else {}
    )
    return images, reviews, presences


@router.get("/datasets/accepted-bbox", response_class=HTMLResponse)
def accepted_bbox_page(request: Request):
    return templates.TemplateResponse(request=request, name="accepted_bbox_review.html", context={})


@router.get("/api/dataset-accepted-bbox/summary")
def accepted_bbox_summary(db: Session = Depends(get_db)) -> dict[str, int]:
    images, reviews, presences = _rows(db)
    accepted = sum(1 for image in images if _reviewed(reviews.get(image.id)))
    candidates = sum(1 for image in images if _candidate(reviews.get(image.id), presences.get(image.id)))
    return {
        "approved_total": len(images),
        "accepted_bbox_pool": accepted,
        "accepted_bbox_missing": max(len(images) - accepted, 0),
        "candidate_bbox_count": candidates,
    }


@router.get("/api/dataset-accepted-bbox/items")
def accepted_bbox_items(
    status: str = Query(default="MISSING"),
    q: str | None = Query(default=None),
    species: str | None = Query(default=None),
    limit: int = Query(default=24, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    images, reviews, presences = _rows(db)
    normalized = (status or "MISSING").strip().upper()
    if normalized not in {"MISSING", "ACCEPTED", "ALL"}:
        raise HTTPException(status_code=400, detail="status 必须是 MISSING、ACCEPTED 或 ALL")
    query = (q or "").strip().lower()
    species_query = (species or "").strip().lower()
    selected = []
    for image in images:
        if query and query not in f"{image.image_id} {image.file_name} {image.batch_id} {image.source_url or ''}".lower():
            continue
        review = reviews.get(image.id)
        resolved_species = str((review.species_name if review else None) or image.truth_species or "").strip().lower()
        if species_query and resolved_species != species_query:
            continue
        confirmed = _reviewed(review)
        if normalized == "MISSING" and confirmed:
            continue
        if normalized == "ACCEPTED" and not confirmed:
            continue
        selected.append(_item(image, review, presences.get(image.id)))
    return {"total": len(selected), "offset": offset, "limit": limit, "items": selected[offset : offset + limit]}


@router.post("/api/dataset-accepted-bbox/{batch_id}/{image_id}/reidentify")
def reidentify_bbox(batch_id: str, image_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    image = _find_image(db, batch_id, image_id)
    try:
        source, _ = _read_uri(image.gcs_uri)
        with Image.open(BytesIO(source)) as opened:
            detector_image = normalize_android_source(opened)
        try:
            run = detect(detector_image)
        finally:
            detector_image.close()
        primary = run.detections[0] if run.detections else None
        candidate = None
        if primary is not None:
            box = primary.box.normalized()
            candidate = [round(value, 6) for value in (box.x1, box.y1, box.width, box.height)]
        row = _review_row(db, image)
        if row is None:
            row = BatchCropReview(batch_id=image.batch_id, image_asset_id=image.id, image_id=image.image_id)
            db.add(row)
        row.candidate_bbox_json = json.dumps(candidate, separators=(",", ":")) if candidate else None
        row.detector_version = run.model_version
        if row.status not in ACCEPTED_STATUSES:
            row.status = "REVIEW_REQUIRED"
        row.updated_at = _now()
        db.commit()
        db.refresh(row)
        presence = db.scalar(select(FishPresenceResult).where(FishPresenceResult.image_asset_id == image.id))
        return _item(image, row, presence)
    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"重新识别失败：{exc}") from exc


@router.patch("/api/dataset-accepted-bbox/{batch_id}/{image_id}")
def update_accepted_bbox(
    batch_id: str,
    image_id: str,
    payload: AcceptedBBoxUpdate,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    image = _find_image(db, batch_id, image_id)
    try:
        row = _apply(
            db,
            image,
            decision=payload.decision,
            accepted_bbox=payload.accepted_bbox,
            reviewer=payload.reviewer,
            notes=payload.notes,
        )
        db.commit()
        db.refresh(row)
        presence = db.scalar(select(FishPresenceResult).where(FishPresenceResult.image_asset_id == image.id))
        result = _item(image, row, presence)
        if row.status in ACCEPTED_STATUSES:
            from app.accepted_pool import enqueue_accepted_pool_sync

            pool_job = enqueue_accepted_pool_sync(db)
            if pool_job:
                result["accepted_pool_sync"] = {
                    "job_id": pool_job.get("job_id"),
                    "status": pool_job.get("status"),
                    "pending_count": pool_job.get("pending_count", 0),
                }
        return result
    except HTTPException:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        raise


@router.post("/api/dataset-accepted-bbox/bulk")
def bulk_accepted_bbox(payload: AcceptedBBoxBulk, db: Session = Depends(get_db)) -> dict[str, int]:
    try:
        for item in payload.items:
            image = _find_image(db, item.batch_id, item.image_id)
            _apply(
                db,
                image,
                decision=item.decision,
                accepted_bbox=item.accepted_bbox,
                reviewer=item.reviewer,
                notes=item.notes,
            )
        db.commit()
        result: dict[str, Any] = {"updated": len(payload.items)}
        if any(item.decision.strip().upper() in ACCEPTED_STATUSES for item in payload.items):
            from app.accepted_pool import enqueue_accepted_pool_sync

            pool_job = enqueue_accepted_pool_sync(db)
            if pool_job:
                result["accepted_pool_sync"] = {
                    "job_id": pool_job.get("job_id"),
                    "status": pool_job.get("status"),
                    "pending_count": pool_job.get("pending_count", 0),
                }
        return result
    except HTTPException:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        raise


__all__ = [
    "AcceptedBBoxBulk",
    "AcceptedBBoxBulkItem",
    "AcceptedBBoxUpdate",
    "accepted_bbox_items",
    "accepted_bbox_page",
    "router",
    "templates",
]
