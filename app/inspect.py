from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import or_, select
from sqlalchemy.orm import Session
from starlette.requests import Request

from app.db import get_db
from app.models import ImageAsset
from app.presence import (
    PRESENCE_MODEL_VERSION,
    FishPresenceResult,
    classify_presence,
    effective_status,
)

router = APIRouter(tags=["data-inspect"])
templates = Jinja2Templates(directory="app/templates")

PENDING_STATUSES = {"pending", "needs_review", "hard_case"}
VALID_PRESENCE = {"single_fish", "multi_fish", "no_fish", "uncertain"}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _review_statuses(value: str | None) -> set[str] | None:
    if not value or value == "all":
        return None
    if value == "pending":
        return PENDING_STATUSES
    if value in {"approved", "rejected"}:
        return {value}
    raise ValueError("invalid review status")


def _presence_meta(row: FishPresenceResult | None) -> dict:
    if not row:
        return {
            "status": "not_scanned",
            "machine_status": "not_scanned",
            "human_override": None,
            "fish_count": 0,
            "fish_score": 0.0,
        }
    saved = {}
    if row.evidence_json:
        try:
            saved = json.loads(row.evidence_json)
        except Exception:
            saved = {}
    human_override = saved.get("human_override")
    machine_status = saved.get("machine_status") or saved.get("status")
    if machine_status not in VALID_PRESENCE:
        machine_status = effective_status(row) if not human_override else "unknown"
    return {
        "status": effective_status(row),
        "machine_status": machine_status,
        "human_override": human_override if human_override in VALID_PRESENCE else None,
        "fish_count": row.fish_count or 0,
        "fish_score": row.fish_score or 0.0,
    }


class PresenceOverride(BaseModel):
    status: str | None = None


@router.get("/inspect", response_class=HTMLResponse)
def inspect_page(request: Request):
    return templates.TemplateResponse(request=request, name="inspect.html", context={})


@router.get("/api/inspect/images")
def inspect_images(
    batch_id: str | None = None,
    species: str | None = None,
    review_status: str | None = Query(default="all"),
    presence: str | None = None,
    q: str | None = None,
    limit: int = Query(default=24, ge=1, le=60),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    try:
        statuses = _review_statuses(review_status)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if presence and presence not in VALID_PRESENCE | {"not_scanned", "error"}:
        raise HTTPException(status_code=400, detail="invalid presence status")

    stmt = select(ImageAsset)
    if batch_id:
        stmt = stmt.where(ImageAsset.batch_id == batch_id)
    if statuses:
        stmt = stmt.where(ImageAsset.review_status.in_(statuses))
    if species:
        stmt = stmt.where(or_(ImageAsset.truth_species == species, ImageAsset.claimed_species == species))
    if q:
        like = f"%{q}%"
        stmt = stmt.where(
            or_(
                ImageAsset.image_id.ilike(like),
                ImageAsset.file_name.ilike(like),
                ImageAsset.source_url.ilike(like),
            )
        )
    images = db.scalars(stmt.order_by(ImageAsset.batch_id, ImageAsset.id)).all()
    image_ids = [x.id for x in images]
    presence_map = {}
    if image_ids:
        presence_map = {
            row.image_asset_id: row
            for row in db.scalars(select(FishPresenceResult).where(FishPresenceResult.image_asset_id.in_(image_ids))).all()
        }

    filtered = []
    for image in images:
        p = _presence_meta(presence_map.get(image.id))
        if presence and p["status"] != presence:
            continue
        filtered.append(
            {
                "batch_id": image.batch_id,
                "image_id": image.image_id,
                "media_url": f"/media/{image.batch_id}/{image.image_id}",
                "claimed_species": image.claimed_species,
                "truth_species": image.truth_species or image.claimed_species,
                "review_status": image.review_status,
                "notes": image.notes or "",
                "presence": p,
            }
        )

    total = len(filtered)
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": filtered[offset : offset + limit],
    }


@router.patch("/api/inspect/presence/{batch_id}/{image_id}")
def inspect_presence_override(
    batch_id: str,
    image_id: str,
    payload: PresenceOverride,
    db: Session = Depends(get_db),
):
    image = db.scalar(select(ImageAsset).where(ImageAsset.batch_id == batch_id, ImageAsset.image_id == image_id))
    if not image:
        raise HTTPException(status_code=404, detail="image not found")
    if payload.status is not None and payload.status not in VALID_PRESENCE:
        raise HTTPException(status_code=400, detail="invalid presence override")

    row = db.scalar(select(FishPresenceResult).where(FishPresenceResult.image_asset_id == image.id))
    if not row:
        if payload.status is None:
            return _presence_meta(None)
        evidence = {
            "status": "not_scanned",
            "machine_status": "not_scanned",
            "human_override": payload.status,
            "created_by_override": True,
            "objects": [],
            "labels": [],
        }
        row = FishPresenceResult(
            image_asset_id=image.id,
            batch_id=batch_id,
            status=payload.status,
            fish_score=0.0,
            fish_count=0,
            max_box_area_ratio=0.0,
            provider="human_override",
            model_version=PRESENCE_MODEL_VERSION,
            evidence_json=json.dumps(evidence, ensure_ascii=False),
            updated_at=utcnow(),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return _presence_meta(row)

    saved = {}
    if row.evidence_json:
        try:
            saved = json.loads(row.evidence_json)
        except Exception:
            saved = {}

    if payload.status is None:
        if saved.get("created_by_override") and not (saved.get("objects") or saved.get("labels")):
            db.delete(row)
            db.commit()
            return _presence_meta(None)
        saved.pop("human_override", None)
        objects = saved.get("objects") or []
        labels = saved.get("labels") or []
        machine = classify_presence(objects, labels)
        row.status = machine["status"]
        row.fish_score = machine["fish_score"]
        row.fish_count = machine["fish_count"]
        row.max_box_area_ratio = machine["max_box_area_ratio"]
        saved.update(machine)
        saved["machine_status"] = machine["status"]
        row.provider = "google_vision"
    else:
        machine_status = saved.get("machine_status") or saved.get("status") or effective_status(row)
        saved["machine_status"] = machine_status
        saved["human_override"] = payload.status
        row.status = payload.status
        row.provider = "human_override"

    row.model_version = PRESENCE_MODEL_VERSION
    row.evidence_json = json.dumps(saved, ensure_ascii=False)
    row.updated_at = utcnow()
    db.commit()
    db.refresh(row)
    return _presence_meta(row)
