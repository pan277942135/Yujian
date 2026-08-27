from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session
from starlette.requests import Request

from app.db import get_db
from app.dedupe import ImageFingerprint
from app.flywheel import species_names
from app.models import Batch, ImageAsset, ReviewEvent
from app.presence import FishPresenceResult, effective_status

router = APIRouter(tags=["bulk-review"])
templates = Jinja2Templates(directory="app/templates")

PENDING_STATUSES = {"pending", "needs_review", "hard_case"}
PUBLIC_REVIEW_STATUSES = {"approved", "rejected", "pending"}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class BulkReviewItem(BaseModel):
    image_id: str
    review_status: str
    truth_species: str | None = None
    notes: str | None = None


class BulkReviewApply(BaseModel):
    batch_id: str
    items: list[BulkReviewItem] = Field(min_length=1, max_length=100)


def _status_filter(status: str | None) -> set[str] | None:
    if not status:
        return None
    if status == "pending":
        return PENDING_STATUSES
    if status in {"approved", "rejected"}:
        return {status}
    raise ValueError("invalid status")


def _image_species(image: ImageAsset) -> str:
    return (image.truth_species or image.claimed_species or "未标注").strip() or "未标注"


def _presence_dict(row: FishPresenceResult | None) -> dict:
    if not row:
        return {"status": "not_scanned", "fish_count": 0, "fish_score": 0.0}
    return {
        "status": effective_status(row),
        "fish_count": row.fish_count or 0,
        "fish_score": row.fish_score or 0.0,
    }


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
def api_bulk_species(batch_id: str, status: str = Query(default="pending"), db: Session = Depends(get_db)):
    if not db.get(Batch, batch_id):
        raise HTTPException(status_code=404, detail="batch not found")
    try:
        statuses = _status_filter(status)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    stmt = select(ImageAsset).where(ImageAsset.batch_id == batch_id)
    if statuses:
        stmt = stmt.where(ImageAsset.review_status.in_(statuses))
    rows = db.scalars(stmt.order_by(ImageAsset.id)).all()
    counts: dict[str, int] = {}
    for image in rows:
        name = _image_species(image)
        counts[name] = counts.get(name, 0) + 1
    catalog_order = {name: idx for idx, name in enumerate(species_names(db, include_candidates=True))}
    ordered = sorted(counts.items(), key=lambda x: (catalog_order.get(x[0], 9999), x[0]))
    return [{"species": name, "count": count} for name, count in ordered]


@router.get("/api/bulk-review/images")
def api_bulk_images(
    batch_id: str,
    species: str,
    status: str = Query(default="pending"),
    presence: str | None = Query(default=None),
    limit: int = Query(default=24, ge=1, le=60),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    if not db.get(Batch, batch_id):
        raise HTTPException(status_code=404, detail="batch not found")
    try:
        statuses = _status_filter(status)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    stmt = select(ImageAsset).where(ImageAsset.batch_id == batch_id)
    if statuses:
        stmt = stmt.where(ImageAsset.review_status.in_(statuses))
    if species:
        stmt = stmt.where(or_(ImageAsset.truth_species == species, ImageAsset.claimed_species == species))
    images = db.scalars(stmt.order_by(ImageAsset.id)).all()

    image_ids = [x.id for x in images]
    presence_rows = {}
    duplicate_rows = {}
    if image_ids:
        presence_rows = {
            row.image_asset_id: row
            for row in db.scalars(select(FishPresenceResult).where(FishPresenceResult.image_asset_id.in_(image_ids))).all()
        }
        duplicate_rows = {
            row.image_asset_id: row
            for row in db.scalars(select(ImageFingerprint).where(ImageFingerprint.image_asset_id.in_(image_ids))).all()
        }

    filtered = []
    for image in images:
        p = _presence_dict(presence_rows.get(image.id))
        if presence and p["status"] != presence:
            continue
        d = _duplicate_dict(duplicate_rows.get(image.id))
        filtered.append(
            {
                "image_id": image.image_id,
                "media_url": f"/media/{image.batch_id}/{image.image_id}",
                "claimed_species": image.claimed_species,
                "truth_species": image.truth_species or image.claimed_species,
                "review_status": image.review_status,
                "notes": image.notes or "",
                "presence": p,
                "duplicate": d,
            }
        )
    total = len(filtered)
    page = filtered[offset : offset + limit]
    return {"total": total, "offset": offset, "limit": limit, "items": page}


@router.post("/api/bulk-review/apply")
def api_bulk_apply(payload: BulkReviewApply, db: Session = Depends(get_db)):
    if not db.get(Batch, payload.batch_id):
        raise HTTPException(status_code=404, detail="batch not found")
    valid_species = set(species_names(db, include_candidates=True))
    changed = 0
    for item in payload.items:
        if item.review_status not in PUBLIC_REVIEW_STATUSES:
            raise HTTPException(status_code=400, detail=f"invalid review_status: {item.review_status}")
        image = db.scalar(
            select(ImageAsset).where(ImageAsset.batch_id == payload.batch_id, ImageAsset.image_id == item.image_id)
        )
        if not image:
            raise HTTPException(status_code=404, detail=f"image not found: {item.image_id}")
        truth = (item.truth_species or image.truth_species or image.claimed_species or "").strip()
        if truth and truth not in valid_species:
            raise HTTPException(status_code=400, detail=f"unknown species: {truth}")
        before = {
            "review_status": image.review_status,
            "truth_species": image.truth_species,
            "notes": image.notes,
        }
        image.review_status = item.review_status
        image.truth_species = truth or None
        if item.notes is not None:
            image.notes = item.notes
        image.reviewed_by = "批量审核"
        image.reviewed_at = utcnow()
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
                        "notes": image.notes,
                    },
                    ensure_ascii=False,
                ),
            )
        )
        changed += 1
    db.commit()
    return {"batch_id": payload.batch_id, "updated": changed}
