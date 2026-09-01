import json
import mimetypes
import os
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from google.cloud import storage
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session
from starlette.requests import Request

from app.batch_console import audit_with_species_catalog, list_incoming_batches
from app.batch_upload_api import ensure_incoming_manifest
from app.db import SessionLocal, get_db, init_db
from app.factory import DOWNLOAD_RETRY, get_bucket_name, promote_incoming_batch, sync_batch_registry
from app.feedback_pipeline import materialize_feedback_batch
from app.flywheel import (
    create_species_candidate,
    ensure_species_catalog,
    flywheel_summary,
    freeze_cumulative_dataset,
    list_feedback,
    list_species,
    record_feedback,
    set_species_status,
    species_names,
)
from app.data_policy import (
    UNCONFIRMED_TRUTH,
    mark_feedback_reviewed,
    normalized_truth,
    truth_distribution,
    truth_filter_clause,
    valid_truth_for_image,
)
from app.models import Batch, DatasetVersion, ImageAsset, ReviewEvent
from app.secure import install_access_guard
from app.services.manifest_normalizer import ManifestNormalizationError
from app.services.review_prefill import parse_review_signals, trusted_truth_prefill

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
install_access_guard(app)


@app.on_event("startup")
def startup():
    init_db()
    db = SessionLocal()
    try:
        ensure_species_catalog(db)
    finally:
        db.close()


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
    parent_version: str | None = None
    git_commit: str | None = None
    preview_hash: str | None = None
    seed: int = 20260826
    train: float = 0.70
    val: float = 0.15


class SpeciesCreate(BaseModel):
    common_name_zh: str
    species_key: str | None = None
    common_name_en: str | None = None
    scientific_name: str | None = None
    notes: str | None = None


class SpeciesStatusUpdate(BaseModel):
    status: str


class FeedbackCreate(BaseModel):
    source_event_id: str
    feedback_type: str
    source: str = "app"
    image_gcs_uri: str | None = None
    model_version: str | None = None
    predicted_species: str | None = None
    confidence: float | None = None
    corrected_species: str | None = None
    user_note: str | None = None


class FeedbackMaterialize(BaseModel):
    batch_id: str
    limit: int = Field(default=500, ge=1, le=2000)


def image_dict(
    image: ImageAsset,
    *,
    classifier_prediction: str | None = None,
    classifier_confidence: float | None = None,
    species_check: str | None = None,
):
    signals = parse_review_signals(image.notes)
    classifier_prediction = (
        classifier_prediction
        or signals.get("classifier_prediction")
        or signals.get("prediction")
        or getattr(image, "classifier_prediction", None)
    )
    classifier_confidence = (
        classifier_confidence
        if classifier_confidence is not None
        else signals.get(
            "classifier_confidence",
            signals.get("prediction_confidence", getattr(image, "classifier_confidence", None)),
        )
    )
    species_check = species_check or signals.get("species_check") or getattr(image, "species_check", None)
    prefill = trusted_truth_prefill(
        claimed_species=image.claimed_species,
        species_check=species_check,
        classifier_prediction=classifier_prediction,
        classifier_confidence=classifier_confidence,
        detector_confidence=getattr(image, "detector_confidence", None),
    )
    return {
        "batch_id": image.batch_id,
        "image_id": image.image_id,
        "file_name": image.file_name,
        "source_url": image.source_url,
        "source_platform": image.source_platform,
        "claimed_species": image.claimed_species,
        "collected_label": image.claimed_species,
        "species_check": species_check,
        "ai_suggestion": prefill.ai_prediction,
        "ai_confidence": prefill.ai_confidence,
        "truth_prefill": prefill.truth_species,
        "truth_prefill_source": prefill.source,
        "label_conflict": prefill.conflict,
        "label_conflict_message": prefill.message if prefill.conflict else None,
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
        stmt = stmt.where(truth_filter_clause(species))
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


@app.get("/species", response_class=HTMLResponse)
def species_page(request: Request):
    return templates.TemplateResponse(request=request, name="species.html", context={})


@app.get("/feedback", response_class=HTMLResponse)
def feedback_page(request: Request):
    return templates.TemplateResponse(request=request, name="feedback.html", context={})


@app.get("/api/overview")
def overview(db: Session = Depends(get_db)):
    total = db.scalar(select(func.count()).select_from(ImageAsset)) or 0
    status_rows = db.execute(select(ImageAsset.review_status, func.count()).group_by(ImageAsset.review_status)).all()
    status_counts = {status: count for status, count in status_rows}
    species_rows, unconfirmed_truth = truth_distribution(db)
    species = [{"species": name, "count": count} for name, count in species_rows]
    if unconfirmed_truth:
        species.append({"species": UNCONFIRMED_TRUTH, "count": unconfirmed_truth})
    result = {
        "total_images": total,
        "batch_count": db.scalar(select(func.count()).select_from(Batch)) or 0,
        "dataset_count": db.scalar(select(func.count()).select_from(DatasetVersion)) or 0,
        "review": {name: status_counts.get(name, 0) for name in REVIEW_VALUES},
        "species": species,
        "unconfirmed_truth_count": unconfirmed_truth,
    }
    result["flywheel"] = flywheel_summary(db)
    return result


@app.get("/api/flywheel/summary")
def api_flywheel_summary(db: Session = Depends(get_db)):
    return flywheel_summary(db)


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
                "image_count": db.scalar(select(func.count()).select_from(ImageAsset).where(ImageAsset.batch_id == batch.batch_id)) or 0,
                "raw_image_count": batch.image_count,
                "status": batch.status,
                "manifest_uri": batch.manifest_uri,
                "raw_uri": batch.raw_uri,
                "review": {status: count for status, count in status_rows},
            }
        )
    return result


@app.post("/api/batches/audit")
def batch_audit(payload: BatchAction, db: Session = Depends(get_db)):
    try:
        return audit_with_species_catalog(
            db,
            incoming_prefix=payload.incoming_prefix,
            batch_id=payload.batch_id,
            source=payload.source,
        )
    except ManifestNormalizationError as exc:
        return JSONResponse(status_code=400, content=exc.as_dict())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/batches/promote")
def batch_promote(payload: BatchAction):
    try:
        ensure_incoming_manifest(payload.incoming_prefix)
        return promote_incoming_batch(payload.incoming_prefix, payload.batch_id, payload.source)
    except ManifestNormalizationError as exc:
        return JSONResponse(status_code=400, content=exc.as_dict())
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
    rows = db.scalars(stmt).all()
    result = []
    for row in rows:
        event = db.scalar(
            select(FeedbackEvent)
            .where(FeedbackEvent.image_gcs_uri == row.gcs_uri)
            .order_by(FeedbackEvent.created_at.desc())
        )
        result.append(
            image_dict(
                row,
                classifier_prediction=event.predicted_species if event else None,
                classifier_confidence=event.confidence if event else None,
            )
        )
    return result


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
    status_stmt = apply_review_filters(
        select(ImageAsset.review_status, func.count()),
        None,
        batch_id,
        species,
        q,
    ).group_by(ImageAsset.review_status)
    all_status = db.execute(status_stmt).all()
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

    proposed_truth = normalized_truth(image)
    if "truth_species" in payload.model_fields_set:
        proposed_truth = (payload.truth_species or "").strip()
    if proposed_truth and not valid_truth_for_image(db, image, proposed_truth):
        raise HTTPException(status_code=400, detail="真实鱼种不是可用鱼种；已停用鱼种只能保留历史值，不能新分配")

    proposed_status = payload.review_status if payload.review_status is not None else image.review_status
    if proposed_status == "approved" and not proposed_truth:
        raise HTTPException(status_code=400, detail="通过前必须确认真实鱼种；采集标注不能自动作为 Ground Truth")

    before = image_dict(image)
    image.review_status = proposed_status
    image.truth_species = proposed_truth or None
    if payload.truth_status is not None:
        image.truth_status = payload.truth_status
    elif not proposed_truth:
        image.truth_status = "UNCERTAIN"
    elif proposed_status == "approved":
        image.truth_status = "LIKELY_CORRECT"
    if payload.notes is not None:
        image.notes = payload.notes
    image.reviewed_by = payload.reviewer
    image.reviewed_at = datetime.now(timezone.utc)
    mark_feedback_reviewed(db, image)
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


@app.get("/api/species")
def api_species(status: str | None = None, db: Session = Depends(get_db)):
    return list_species(db, status=status)


@app.post("/api/species")
def api_create_species(payload: SpeciesCreate, db: Session = Depends(get_db)):
    try:
        return create_species_candidate(
            db,
            common_name_zh=payload.common_name_zh,
            species_key=payload.species_key,
            common_name_en=payload.common_name_en,
            scientific_name=payload.scientific_name,
            notes=payload.notes,
        )
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.patch("/api/species/{species_key}/status")
def api_species_status(species_key: str, payload: SpeciesStatusUpdate, db: Session = Depends(get_db)):
    try:
        return set_species_status(db, species_key, payload.status)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/feedback")
def api_feedback(status: str | None = None, limit: int = Query(default=100, ge=1, le=500), db: Session = Depends(get_db)):
    return list_feedback(db, status=status, limit=limit)


@app.post("/api/feedback")
def api_record_feedback(payload: FeedbackCreate, db: Session = Depends(get_db)):
    try:
        return record_feedback(db, **payload.model_dump())
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/feedback/materialize")
def api_materialize_feedback(payload: FeedbackMaterialize, db: Session = Depends(get_db)):
    try:
        return materialize_feedback_batch(db, batch_id=payload.batch_id, limit=payload.limit)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/datasets/summary")
def dataset_summary(db: Session = Depends(get_db)):
    return flywheel_summary(db)


@app.get("/api/datasets")
def datasets(db: Session = Depends(get_db)):
    rows = db.scalars(select(DatasetVersion).order_by(DatasetVersion.created_at.desc())).all()
    return [
        {
            "dataset_version": row.dataset_version,
            "parent_version": row.parent_version,
            "created_at": row.created_at.isoformat(),
            "manifest_uri": row.manifest_uri,
            "class_map_uri": row.class_map_uri,
            "train_count": row.train_count,
            "val_count": row.val_count,
            "test_count": row.test_count,
            "species_count": row.species_count,
            "selection_mode": row.selection_mode,
            "source_cutoff_at": row.source_cutoff_at.isoformat() if row.source_cutoff_at else None,
            "git_commit": row.git_commit,
            "status": row.status,
        }
        for row in rows
    ]


@app.post("/api/datasets/freeze")
def dataset_freeze(payload: DatasetFreeze, db: Session = Depends(get_db)):
    try:
        from app.dataset_api import DatasetFreezePreviewRequest, build_preview

        if not payload.preview_hash:
            raise ValueError("请先生成冻结预览；Freeze 必须携带 preview_hash")
        preview = build_preview(
            db,
            DatasetFreezePreviewRequest(
                dataset_version=payload.dataset_version,
                parent_version=payload.parent_version,
                seed=payload.seed,
                train=payload.train,
                val=payload.val,
            ),
        )
        if preview.get("selection_hash") != payload.preview_hash:
            raise ValueError("冻结预览已失效：数据、鱼种状态或父版本发生变化，请重新预览")
        deployed_git = (os.getenv("APP_GIT_COMMIT") or "unknown").strip() or "unknown"
        return freeze_cumulative_dataset(
            db,
            dataset_version=payload.dataset_version,
            parent_version=preview.get("parent_version"),
            git_commit=deployed_git,
            seed=payload.seed,
            train=payload.train,
            val=payload.val,
        )
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
