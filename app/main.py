import json
import mimetypes
import os
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from google.cloud import storage
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session
from starlette.requests import Request

from app.db import get_db, init_db
from app.factory import (
    DOWNLOAD_RETRY,
    TARGET_SPECIES,
    approved_summary,
    audit_incoming_batch,
    freeze_dataset,
    get_bucket_name,
    list_datasets,
    list_incoming_batches,
    promote_incoming_batch,
    sync_batch_registry,
)
from app.models import Batch, DatasetVersion, ImageAsset, ReviewEvent

REVIEW_VALUES = {"approved", "needs_review", "rejected", "hard_case", "pending"}
TRUTH_VALUES = {
    "LIKELY_CORRECT",
    "UNCERTAIN",
    "HARD_PAIR_REVIEW",
    "WRONG_LABEL_SUSPECTED",
    "EXPERT_HOLD",
}

app = FastAPI(title="YuJian AI Model Factory", version="0.1.0")
templates = Jinja2Templates(directory="app/templates")


@app.on_event("startup")
def startup():
    init_db()


class ReviewUpdate(BaseModel):
    review_status: str | None = None
    truth_species: str | None = None
    truth_status: str | None = None
    notes: str | None = None
    reviewer: str = Field(default="web-review")


class BatchAction(BaseModel):
    incoming_prefix: str
    batch_id: str
    source: str


class BatchSync(BaseModel):
    batch_id: str


class DatasetFreeze(BaseModel):
    dataset_version: str
    batch_ids: list[str]
    parent_version: str | None = None
    git_commit: str = Field(default_factory=lambda: os.getenv("APP_GIT_COMMIT", "unknown"))
    seed: int = 20260826
    train: float = 0.70
    val: float = 0.15


def image_dict(image: ImageAsset):
    return {
        "batch_id": image.batch_id,
        "image_id": image.image_id,
        "file_name": image.file_name,
        "source_url": image.source_url,
        "source_platform": image.source_platform,
        "claimed_species": image.claimed_species,
        "truth_species": image.truth_species,
        "truth_status": image.truth_status,
        "review_status": image.review_status,
        "scene": image.scene,
        "lighting": image.lighting,
        "quality": image.quality,
        "group_id": image.group_id,
        "notes": image.notes,
        "reviewed_by": image.reviewed_by,
        "reviewed_at": image.reviewed_at.isoformat() if image.reviewed_at else None,
        "media_url": f"/media/{image.batch_id}/{image.image_id}",
    }


def apply_review_filters(stmt, status=None, batch_id=None, species=None, q=None):
    if status:
        stmt = stmt.where(ImageAsset.review_status == status)
    if batch_id:
        stmt = stmt.where(ImageAsset.batch_id == batch_id)
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
    return stmt


@app.get("/health")
def health():
    return {"status": "ok", "version": app.version}


@app.get("/", response_class=HTMLResponse)
def overview_page(request: Request):
    return templates.TemplateResponse(request=request, name="overview.html", context={})


@app.get("/batches", response_class=HTMLResponse)
def batches_page(request: Request):
    return templates.TemplateResponse(request=request, name="batches.html", context={})


@app.get("/review", response_class=HTMLResponse)
def review_page(request: Request):
    return templates.TemplateResponse(request=request, name="review.html", context={})


@app.get("/datasets", response_class=HTMLResponse)
def datasets_page(request: Request):
    return templates.TemplateResponse(request=request, name="datasets.html", context={})


@app.get("/api/overview")
def overview(db: Session = Depends(get_db)):
    total = db.scalar(select(func.count()).select_from(ImageAsset)) or 0
    status_rows = db.execute(select(ImageAsset.review_status, func.count()).group_by(ImageAsset.review_status)).all()
    status_counts = {status: count for status, count in status_rows}
    species_key = func.coalesce(ImageAsset.truth_species, ImageAsset.claimed_species, "unknown")
    species_rows = db.execute(
        select(species_key.label("species"), func.count()).group_by(species_key).order_by(func.count().desc())
    ).all()
    return {
        "total_images": total,
        "batch_count": db.scalar(select(func.count()).select_from(Batch)) or 0,
        "dataset_count": db.scalar(select(func.count()).select_from(DatasetVersion)) or 0,
        "review": {name: status_counts.get(name, 0) for name in REVIEW_VALUES},
        "species": [{"species": species, "count": count} for species, count in species_rows],
    }


@app.get("/api/incoming")
def incoming():
    try:
        return list_incoming_batches()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/batches")
def batches(db: Session = Depends(get_db)):
    rows = db.scalars(select(Batch).order_by(Batch.created_at.desc())).all()
    result = []
    for batch in rows:
        status_rows = db.execute(
            select(ImageAsset.review_status, func.count())
            .where(ImageAsset.batch_id == batch.batch_id)
            .group_by(ImageAsset.review_status)
        ).all()
        result.append(
            {
                "batch_id": batch.batch_id,
                "source": batch.source,
                "created_at": batch.created_at.isoformat(),
                "image_count": batch.image_count,
                "status": batch.status,
                "manifest_uri": batch.manifest_uri,
                "raw_uri": batch.raw_uri,
                "review": {status: count for status, count in status_rows},
            }
        )
    return result


@app.post("/api/batches/audit")
def batch_audit(payload: BatchAction):
    try:
        return audit_incoming_batch(payload.incoming_prefix, payload.batch_id, payload.source)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/batches/promote")
def batch_promote(payload: BatchAction):
    try:
        return promote_incoming_batch(payload.incoming_prefix, payload.batch_id, payload.source)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/batches/sync")
def batch_sync(payload: BatchSync, db: Session = Depends(get_db)):
    try:
        return sync_batch_registry(db, payload.batch_id)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/review")
def review_queue(
    status: str | None = Query(default="pending"),
    batch_id: str | None = None,
    species: str | None = None,
    q: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    stmt = apply_review_filters(select(ImageAsset), status, batch_id, species, q)
    stmt = stmt.order_by(ImageAsset.batch_id, ImageAsset.id).offset(offset).limit(limit)
    return [image_dict(row) for row in db.scalars(stmt).all()]


@app.get("/api/review/stats")
def review_stats(
    status: str | None = Query(default=None),
    batch_id: str | None = None,
    species: str | None = None,
    q: str | None = None,
    db: Session = Depends(get_db),
):
    filtered = apply_review_filters(select(ImageAsset.id), status, batch_id, species, q).subquery()
    count = db.scalar(select(func.count()).select_from(filtered)) or 0
    all_status = db.execute(
        select(ImageAsset.review_status, func.count())
        .where(ImageAsset.batch_id == batch_id if batch_id else True)
        .group_by(ImageAsset.review_status)
    ).all()
    return {"filtered": count, "status": {key: value for key, value in all_status}}


@app.patch("/api/review/{batch_id}/{image_id}")
def update_review(batch_id: str, image_id: str, payload: ReviewUpdate, db: Session = Depends(get_db)):
    image = db.scalar(select(ImageAsset).where(ImageAsset.batch_id == batch_id, ImageAsset.image_id == image_id))
    if not image:
        raise HTTPException(status_code=404, detail="image not found")
    if payload.review_status is not None and payload.review_status not in REVIEW_VALUES:
        raise HTTPException(status_code=400, detail=f"invalid review_status: {payload.review_status}")
    if payload.truth_status is not None and payload.truth_status not in TRUTH_VALUES:
        raise HTTPException(status_code=400, detail=f"invalid truth_status: {payload.truth_status}")
    if payload.truth_species and payload.truth_species not in TARGET_SPECIES:
        raise HTTPException(status_code=400, detail=f"invalid truth_species: {payload.truth_species}")

    before = image_dict(image)
    if payload.review_status is not None:
        image.review_status = payload.review_status
    if payload.truth_species is not None:
        image.truth_species = payload.truth_species.strip() or None
    if payload.truth_status is not None:
        image.truth_status = payload.truth_status
    if payload.notes is not None:
        image.notes = payload.notes
    image.reviewed_by = payload.reviewer
    image.reviewed_at = datetime.now(timezone.utc)
    after = image_dict(image)
    db.add(
        ReviewEvent(
            image_asset_id=image.id,
            action="review_update",
            reviewer=payload.reviewer,
            before_json=json.dumps(before, ensure_ascii=False),
            after_json=json.dumps(after, ensure_ascii=False),
        )
    )
    db.commit()
    db.refresh(image)
    return image_dict(image)


@app.get("/media/{batch_id}/{image_id}")
def media(batch_id: str, image_id: str, db: Session = Depends(get_db)):
    image = db.scalar(select(ImageAsset).where(ImageAsset.batch_id == batch_id, ImageAsset.image_id == image_id))
    if not image:
        raise HTTPException(status_code=404, detail="image not found")
    bucket_name = get_bucket_name()
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(image.object_name)
    if not blob.exists(client):
        raise HTTPException(status_code=404, detail="GCS object not found")
    content = blob.download_as_bytes(timeout=120, retry=DOWNLOAD_RETRY)
    media_type = mimetypes.guess_type(image.file_name)[0] or "application/octet-stream"
    return Response(content=content, media_type=media_type, headers={"Cache-Control": "private, max-age=300"})


@app.get("/api/datasets/summary")
def dataset_summary(db: Session = Depends(get_db)):
    return approved_summary(db)


@app.get("/api/datasets")
def datasets(db: Session = Depends(get_db)):
    return list_datasets(db)


@app.post("/api/datasets/freeze")
def dataset_freeze(payload: DatasetFreeze, db: Session = Depends(get_db)):
    try:
        return freeze_dataset(
            db=db,
            dataset_version=payload.dataset_version,
            batch_ids=payload.batch_ids,
            git_commit=payload.git_commit,
            seed=payload.seed,
            train=payload.train,
            val=payload.val,
            parent_version=payload.parent_version,
        )
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
