from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session
from starlette.requests import Request

from app.data_policy import review_group_clause, valid_truth_for_image
from app.db import get_db
from app.dedupe import ImageFingerprint
from app.flywheel import species_names
from app.models import Batch, BatchCropReview, FeedbackEvent, ImageAsset, ReviewEvent, SpeciesCatalog
from app.presence import FishPresenceResult, effective_status

router = APIRouter(tags=["bulk-review"])
templates = Jinja2Templates(directory="app/templates")
logger = logging.getLogger(__name__)

PENDING_STATUSES = {"pending", "needs_review", "hard_case"}
PUBLIC_REVIEW_STATUSES = {"approved", "rejected", "pending", "needs_review", "hard_case"}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class BulkReviewItem(BaseModel):
    batch_id: str | None = None
    image_id: str
    review_status: str
    truth_species: str | None = None
    accepted_bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    notes: str | None = None


class BulkReviewApply(BaseModel):
    batch_id: str | None = None
    items: list[BulkReviewItem] = Field(min_length=1, max_length=100)


def _status_filter(status: str | None) -> set[str] | None:
    if not status:
        return None
    if status == "pending":
        return PENDING_STATUSES
    if status in {"approved", "rejected"}:
        return {status}
    raise ValueError("invalid status")


def _presence_dict(row: FishPresenceResult | None) -> dict:
    if not row:
        return {"status": "not_scanned", "fish_count": 0, "fish_score": 0.0}
    return {
        "status": effective_status(row),
        "fish_count": row.fish_count or 0,
        "fish_score": row.fish_score or 0.0,
    }


def _bbox(value) -> list[float] | None:
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


def _duplicate_dict(row: ImageFingerprint | None) -> dict:
    if not row or not row.duplicate_group:
        return {"group": None, "is_duplicate": False, "is_representative": True, "kind": None}
    return {
        "group": row.duplicate_group,
        "is_duplicate": not bool(row.is_representative),
        "is_representative": bool(row.is_representative),
        "kind": row.duplicate_kind,
    }


@router.get("/review/bulk", response_class=HTMLResponse)
def bulk_review_page(request: Request):
    return templates.TemplateResponse(request=request, name="bulk_review.html", context={})


@router.get("/api/bulk-review/species")
def api_bulk_species(batch_id: str | None = Query(default=None), status: str = Query(default="pending"), db: Session = Depends(get_db)):
    if batch_id and not db.get(Batch, batch_id):
        raise HTTPException(status_code=404, detail="batch not found")
    try:
        statuses = _status_filter(status)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    truth = func.nullif(func.trim(ImageAsset.truth_species), "")
    claimed = func.nullif(func.trim(ImageAsset.claimed_species), "")
    group_name = func.coalesce(truth, claimed, "未标注")
    stmt = select(group_name, func.count()).where(ImageAsset.review_status.in_(statuses or PENDING_STATUSES))
    if batch_id:
        stmt = stmt.where(ImageAsset.batch_id == batch_id)
    if statuses:
        stmt = stmt.where(ImageAsset.review_status.in_(statuses))
    rows = db.execute(stmt.group_by(group_name)).all()
    counts = {str(name): int(count) for name, count in rows}
    catalog_order = {name: idx for idx, name in enumerate(species_names(db, include_candidates=True))}
    ordered = sorted(counts.items(), key=lambda x: (catalog_order.get(x[0], 9999), x[0]))
    return [{"species": name, "count": count} for name, count in ordered]


@router.get("/api/bulk-review/images")
def api_bulk_images(
    batch_id: str | None = Query(default=None),
    species: str = Query(default=""),
    status: str = Query(default="pending"),
    presence: str | None = Query(default=None),
    limit: int = Query(default=24, ge=1, le=60),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    if batch_id and not db.get(Batch, batch_id):
        raise HTTPException(status_code=404, detail="batch not found")
    try:
        statuses = _status_filter(status)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    presence_status = FishPresenceResult.status
    presence_filter = {
        "single_fish": and_(presence_status == "fish_present", FishPresenceResult.fish_count == 1),
        "multi_fish": and_(presence_status == "fish_present", FishPresenceResult.fish_count >= 2),
        "no_fish": presence_status == "no_fish",
        "uncertain": or_(presence_status == "uncertain", and_(presence_status == "fish_present", FishPresenceResult.fish_count == 0)),
        "not_scanned": FishPresenceResult.id.is_(None),
    }
    if presence and presence not in presence_filter:
        raise HTTPException(status_code=400, detail="invalid presence")

    joins = (
        ImageAsset.__table__
        .outerjoin(FishPresenceResult, FishPresenceResult.image_asset_id == ImageAsset.id)
        .outerjoin(ImageFingerprint, ImageFingerprint.image_asset_id == ImageAsset.id)
        .outerjoin(BatchCropReview, BatchCropReview.image_asset_id == ImageAsset.id)
    )
    criteria = []
    if statuses:
        criteria.append(ImageAsset.review_status.in_(statuses))
    if batch_id:
        criteria.append(ImageAsset.batch_id == batch_id)
    if species:
        criteria.append(review_group_clause(species))
    if presence:
        criteria.append(presence_filter[presence])
    stmt = (
        select(ImageAsset, FishPresenceResult, ImageFingerprint, BatchCropReview)
        .select_from(joins)
        .where(*criteria)
    )
    count_stmt = select(func.count()).select_from(joins).where(*criteria)
    total = int(db.scalar(count_stmt) or 0)
    rows = db.execute(stmt.order_by(ImageAsset.id).offset(offset).limit(limit)).all()

    page = []
    for image, presence_row, duplicate_row, crop in rows:
        p = _presence_dict(presence_row)
        d = _duplicate_dict(duplicate_row)
        candidate = _bbox(crop.candidate_bbox_json) if crop else None
        accepted = _bbox(crop.accepted_bbox_json) if crop else None
        bbox_confirmed = bool(crop and crop.status in {"ACCEPTED", "TRAINING_READY"} and accepted)
        page.append(
            {
                "batch_id": image.batch_id,
                "image_id": image.image_id,
                "media_url": f"/media/{image.batch_id}/{image.image_id}",
                "claimed_species": image.claimed_species,
                "truth_species": image.truth_species,
                "review_status": image.review_status,
                "notes": image.notes or "",
                "presence": p,
                "duplicate": d,
                "candidate_bbox": candidate,
                "accepted_bbox": accepted,
                "bbox_status": "ACCEPTED" if bbox_confirmed else ("CANDIDATE" if candidate else "MISSING"),
            }
        )
    return {"total": total, "offset": offset, "limit": limit, "items": page}


@router.post("/api/bulk-review/apply")
def api_bulk_apply(payload: BulkReviewApply, db: Session = Depends(get_db)):
    started_at = time.perf_counter()
    batch_ids = {item.batch_id or payload.batch_id for item in payload.items}
    if None in batch_ids:
        raise HTTPException(status_code=400, detail="batch_id is required")
    batch_ids = {str(batch_id) for batch_id in batch_ids}
    existing_batches = set(db.scalars(select(Batch.batch_id).where(Batch.batch_id.in_(batch_ids))).all())
    missing_batches = batch_ids - existing_batches
    if missing_batches:
        raise HTTPException(status_code=404, detail=f"batch not found: {sorted(missing_batches)[0]}")

    image_keys = [(item.batch_id or payload.batch_id, item.image_id) for item in payload.items]
    image_ids = {image_id for _, image_id in image_keys}
    images = db.scalars(
        select(ImageAsset).where(ImageAsset.batch_id.in_(batch_ids), ImageAsset.image_id.in_(image_ids))
    ).all()
    image_by_key = {(image.batch_id, image.image_id): image for image in images}
    missing_images = [key for key in image_keys if key not in image_by_key]
    if missing_images:
        raise HTTPException(status_code=404, detail=f"image not found: {missing_images[0][1]}")
    image_rows = list(image_by_key.values())
    asset_ids = [image.id for image in image_rows]
    crop_by_asset = {
        row.image_asset_id: row
        for row in db.scalars(select(BatchCropReview).where(BatchCropReview.image_asset_id.in_(asset_ids))).all()
    }
    feedback_rows = db.scalars(
        select(FeedbackEvent).where(
            or_(*[and_(FeedbackEvent.materialized_batch_id == batch_id, FeedbackEvent.materialized_image_id == image_id) for batch_id, image_id in image_keys])
        )
    ).all()
    feedback_by_key = {(row.materialized_batch_id, row.materialized_image_id): row for row in feedback_rows}
    species_by_name = {
        row.common_name_zh: row
        for row in db.scalars(select(SpeciesCatalog)).all()
    }
    prefetch_ms = (time.perf_counter() - started_at) * 1000
    db_started = time.perf_counter()
    changed = 0
    enqueue_needed = False
    for item in payload.items:
        if item.review_status not in PUBLIC_REVIEW_STATUSES:
            raise HTTPException(status_code=400, detail=f"invalid review_status: {item.review_status}")
        batch_id = item.batch_id or payload.batch_id
        image = image_by_key[(batch_id, item.image_id)]

        if "truth_species" in item.model_fields_set:
            truth = (item.truth_species or "").strip()
        else:
            truth = (image.truth_species or "").strip()
        if truth and not valid_truth_for_image(db, image, truth, catalog_by_name=species_by_name):
            raise HTTPException(status_code=400, detail=f"不可分配真实鱼种: {truth}")
        if item.review_status == "approved" and not truth:
            raise HTTPException(status_code=400, detail=f"{item.image_id}: 通过前必须确认真实鱼种")
        crop = crop_by_asset.get(image.id)
        existing_bbox = _bbox(crop.accepted_bbox_json) if crop else None
        accepted_bbox = _bbox(item.accepted_bbox) if "accepted_bbox" in item.model_fields_set else existing_bbox
        if item.review_status == "approved" and accepted_bbox is None:
            raise HTTPException(status_code=400, detail={"error": "ACCEPTED_BBOX_REQUIRED", "reason": f"{item.image_id}: 通过前必须确认 accepted_bbox"})

        before = {
            "review_status": image.review_status,
            "truth_species": image.truth_species,
            "truth_status": image.truth_status,
            "notes": image.notes,
        }
        image.review_status = item.review_status
        image.truth_species = truth or None
        image.truth_status = "LIKELY_CORRECT" if item.review_status == "approved" and truth else ("UNCERTAIN" if not truth else image.truth_status)
        if item.notes is not None:
            image.notes = item.notes
        image.reviewed_by = "批量审核"
        image.reviewed_at = utcnow()
        feedback = feedback_by_key.get((image.batch_id, image.image_id))
        if image.review_status in {"approved", "rejected"} and feedback and feedback.pipeline_status == "BATCHED":
            feedback.pipeline_status = "REVIEWED"
        if item.review_status == "approved":
            if crop is None:
                crop = BatchCropReview(batch_id=image.batch_id, image_asset_id=image.id, image_id=image.image_id)
                db.add(crop)
                db.flush()
            crop.accepted_bbox_json = json.dumps(accepted_bbox, separators=(",", ":"))
            crop.species_name = truth
            crop.status = "ACCEPTED"
            crop.reviewer = "批量审核"
            crop.reviewed_at = utcnow()
            crop.notes = item.notes
            enqueue_needed = True
        db.add(
            ReviewEvent(
                image_asset_id=image.id,
                action="bulk_review_update",
                reviewer="批量审核",
                before_json=json.dumps(before, ensure_ascii=False),
                after_json=json.dumps(
                    {
                        "review_status": image.review_status,
                        "truth_species": image.truth_species,
                        "truth_status": image.truth_status,
                        "notes": image.notes,
                        "accepted_bbox": accepted_bbox if item.review_status == "approved" else None,
                    },
                    ensure_ascii=False,
                ),
            )
        )
        changed += 1
    if enqueue_needed:
        from app.accepted_pool import enqueue_accepted_pool_sync

        enqueue_started = time.perf_counter()
        pool_signal = enqueue_accepted_pool_sync(db)
        enqueue_ms = (time.perf_counter() - enqueue_started) * 1000
    else:
        pool_signal = None
        enqueue_ms = 0.0
    db.commit()
    db_ms = (time.perf_counter() - db_started) * 1000
    total_ms = (time.perf_counter() - started_at) * 1000
    logger.info(
        "bulk_review_apply item_count=%d batch_count=%d prefetch_ms=%.2f db_ms=%.2f enqueue_ms=%.2f total_ms=%.2f",
        len(payload.items), len(batch_ids), prefetch_ms, db_ms, enqueue_ms, total_ms,
    )
    result = {"batch_id": payload.batch_id, "updated": changed}
    if pool_signal:
        result["accepted_pool_sync"] = pool_signal
    return result
