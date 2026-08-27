from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from google.cloud import storage, vision
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func, select
from sqlalchemy.orm import Session

from app.db import Base, get_db
from app.factory import DOWNLOAD_RETRY, get_bucket_name
from app.models import Batch, ImageAsset, ReviewEvent

router = APIRouter(prefix="/api/presence", tags=["fish-presence"])

PRESENCE_MODEL_VERSION = "google-vision-presence-v0.1"
FISH_OBJECT_THRESHOLD = 0.45
FISH_LABEL_THRESHOLD = 0.65
UNCERTAIN_FISH_THRESHOLD = 0.20
STRONG_CONTEXT_THRESHOLD = 0.75
MIN_CONTEXT_LABELS_FOR_NO_FISH = 4

# Cloud Vision labels are English. Keep the matching conservative: an explicit
# fish label is required for positive evidence; generic "animal" is not enough.
FISH_TERMS = {
    "fish",
    "freshwater fish",
    "ray-finned fish",
    "bony fish",
    "marine fish",
    "game fish",
    "fishery",
    "fishing",
}

ELIGIBLE_REVIEW_STATUSES = {"pending", "needs_review", "hard_case"}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class FishPresenceResult(Base):
    """Pre-review fish-body presence result.

    Kept in a separate table so the deployed Registry can add this capability
    through SQLAlchemy create_all without altering the existing image_assets table.
    """

    __tablename__ = "fish_presence_results"
    __table_args__ = (UniqueConstraint("image_asset_id", name="uq_presence_image_asset"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_asset_id = Column(Integer, ForeignKey("image_assets.id"), nullable=False, index=True)
    batch_id = Column(String(128), nullable=False, index=True)
    status = Column(String(32), nullable=False, index=True)  # fish_present/no_fish/uncertain/error
    fish_score = Column(Float, nullable=False, default=0.0)
    fish_count = Column(Integer, nullable=False, default=0)
    max_box_area_ratio = Column(Float, nullable=False, default=0.0)
    provider = Column(String(64), nullable=False, default="google_vision")
    model_version = Column(String(128), nullable=False, default=PRESENCE_MODEL_VERSION)
    evidence_json = Column(Text)
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)


class PresenceScanRequest(BaseModel):
    batch_id: str
    limit: int = Field(default=40, ge=1, le=100)
    rescan: bool = False


class PresenceRejectRequest(BaseModel):
    batch_id: str


def _norm(value: str | None) -> str:
    return (value or "").strip().lower()


def _is_fish_term(name: str | None) -> bool:
    value = _norm(name)
    if not value:
        return False
    if value in FISH_TERMS:
        return True
    return "fish" in value and "fishing rod" not in value and "fishing reel" not in value


def _box_area(vertices: list[dict]) -> float:
    xs = [float(v.get("x", 0.0) or 0.0) for v in vertices]
    ys = [float(v.get("y", 0.0) or 0.0) for v in vertices]
    if not xs or not ys:
        return 0.0
    width = max(0.0, min(1.0, max(xs)) - max(0.0, min(xs)))
    height = max(0.0, min(1.0, max(ys)) - max(0.0, min(ys)))
    return min(1.0, width * height)


def classify_presence(objects: list[dict], labels: list[dict]) -> dict:
    """Turn generic Vision evidence into a conservative three-way decision.

    `no_fish` is only emitted when there is no fish evidence and the label model
    still returned enough high-confidence context to make the absence meaningful.
    Everything else falls back to `uncertain` rather than being silently rejected.
    """

    fish_objects = [x for x in objects if _is_fish_term(x.get("name"))]
    fish_labels = [x for x in labels if _is_fish_term(x.get("name"))]

    object_scores = [float(x.get("score", 0.0) or 0.0) for x in fish_objects]
    label_scores = [float(x.get("score", 0.0) or 0.0) for x in fish_labels]
    max_object_score = max(object_scores, default=0.0)
    max_label_score = max(label_scores, default=0.0)
    fish_score = max(max_object_score, max_label_score)

    strong_fish_objects = [x for x in fish_objects if float(x.get("score", 0.0) or 0.0) >= FISH_OBJECT_THRESHOLD]
    areas = [_box_area(x.get("vertices") or []) for x in strong_fish_objects]
    max_area = max(areas, default=0.0)

    if strong_fish_objects or max_label_score >= FISH_LABEL_THRESHOLD:
        status = "fish_present"
    elif fish_score >= UNCERTAIN_FISH_THRESHOLD:
        status = "uncertain"
    else:
        strong_context = [x for x in labels if float(x.get("score", 0.0) or 0.0) >= STRONG_CONTEXT_THRESHOLD]
        status = "no_fish" if len(strong_context) >= MIN_CONTEXT_LABELS_FOR_NO_FISH else "uncertain"

    return {
        "status": status,
        "fish_score": round(fish_score, 6),
        "fish_count": len(strong_fish_objects),
        "max_box_area_ratio": round(max_area, 6),
        "objects": objects,
        "labels": labels,
    }


def _vision_evidence(client: vision.ImageAnnotatorClient, content: bytes) -> dict:
    image = vision.Image(content=content)
    features = [
        vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION, max_results=20),
        vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION, max_results=20),
    ]
    response = client.annotate_image(request={"image": image, "features": features})
    if response.error.message:
        raise RuntimeError(response.error.message)

    objects = []
    for item in response.localized_object_annotations:
        vertices = [
            {"x": float(v.x or 0.0), "y": float(v.y or 0.0)}
            for v in item.bounding_poly.normalized_vertices
        ]
        objects.append({"name": item.name, "score": float(item.score), "vertices": vertices})

    labels = [
        {"name": item.description, "score": float(item.score)}
        for item in response.label_annotations
    ]
    return classify_presence(objects, labels)


def _result_dict(row: FishPresenceResult) -> dict:
    return {
        "image_asset_id": row.image_asset_id,
        "batch_id": row.batch_id,
        "status": row.status,
        "fish_score": row.fish_score,
        "fish_count": row.fish_count,
        "max_box_area_ratio": row.max_box_area_ratio,
        "provider": row.provider,
        "model_version": row.model_version,
        "error_message": row.error_message,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def presence_summary(db: Session, batch_id: str) -> dict:
    batch = db.get(Batch, batch_id)
    if not batch:
        raise ValueError("batch not found")

    eligible = db.scalar(
        select(func.count())
        .select_from(ImageAsset)
        .where(ImageAsset.batch_id == batch_id, ImageAsset.review_status.in_(ELIGIBLE_REVIEW_STATUSES))
    ) or 0
    rows = db.execute(
        select(FishPresenceResult.status, func.count())
        .where(FishPresenceResult.batch_id == batch_id)
        .group_by(FishPresenceResult.status)
    ).all()
    counts = {status: count for status, count in rows}
    scanned = sum(counts.values())
    return {
        "batch_id": batch_id,
        "eligible": eligible,
        "scanned": scanned,
        "fish_present": counts.get("fish_present", 0),
        "no_fish": counts.get("no_fish", 0),
        "uncertain": counts.get("uncertain", 0),
        "error": counts.get("error", 0),
        "remaining": max(0, eligible - scanned),
    }


def scan_batch(db: Session, batch_id: str, limit: int = 40, rescan: bool = False) -> dict:
    batch = db.get(Batch, batch_id)
    if not batch:
        raise ValueError("batch not found")

    stmt = (
        select(ImageAsset)
        .where(ImageAsset.batch_id == batch_id, ImageAsset.review_status.in_(ELIGIBLE_REVIEW_STATUSES))
        .order_by(ImageAsset.id)
    )
    if not rescan:
        stmt = (
            stmt.outerjoin(FishPresenceResult, FishPresenceResult.image_asset_id == ImageAsset.id)
            .where(FishPresenceResult.id.is_(None))
        )
    images = db.scalars(stmt.limit(limit)).all()
    if not images:
        summary = presence_summary(db, batch_id)
        summary["processed"] = 0
        return summary

    storage_client = storage.Client()
    vision_client = vision.ImageAnnotatorClient()
    bucket = storage_client.bucket(get_bucket_name())
    processed = 0

    for image in images:
        row = db.scalar(select(FishPresenceResult).where(FishPresenceResult.image_asset_id == image.id))
        if not row:
            row = FishPresenceResult(image_asset_id=image.id, batch_id=batch_id, status="uncertain")
            db.add(row)

        try:
            blob = bucket.blob(image.object_name)
            content = blob.download_as_bytes(timeout=120, retry=DOWNLOAD_RETRY)
            evidence = _vision_evidence(vision_client, content)
            row.status = evidence["status"]
            row.fish_score = evidence["fish_score"]
            row.fish_count = evidence["fish_count"]
            row.max_box_area_ratio = evidence["max_box_area_ratio"]
            row.provider = "google_vision"
            row.model_version = PRESENCE_MODEL_VERSION
            row.evidence_json = json.dumps(evidence, ensure_ascii=False)
            row.error_message = None
        except Exception as exc:  # persist failure and continue with the batch
            row.status = "error"
            row.fish_score = 0.0
            row.fish_count = 0
            row.max_box_area_ratio = 0.0
            row.provider = "google_vision"
            row.model_version = PRESENCE_MODEL_VERSION
            row.evidence_json = None
            row.error_message = str(exc)[:2000]

        row.updated_at = utcnow()
        db.commit()  # every image is resumable if browser/Cloud Run request stops
        processed += 1

    summary = presence_summary(db, batch_id)
    summary["processed"] = processed
    return summary


def reject_no_fish(db: Session, batch_id: str) -> dict:
    if not db.get(Batch, batch_id):
        raise ValueError("batch not found")

    images = db.scalars(
        select(ImageAsset)
        .join(FishPresenceResult, FishPresenceResult.image_asset_id == ImageAsset.id)
        .where(
            ImageAsset.batch_id == batch_id,
            ImageAsset.review_status.in_(ELIGIBLE_REVIEW_STATUSES),
            FishPresenceResult.status == "no_fish",
        )
        .order_by(ImageAsset.id)
    ).all()

    changed = 0
    for image in images:
        before = {
            "review_status": image.review_status,
            "truth_species": image.truth_species,
            "notes": image.notes,
        }
        image.review_status = "rejected"
        note = "[鱼体检测] 未发现可靠鱼体证据，已批量标记为不通过。"
        if note not in (image.notes or ""):
            image.notes = f"{image.notes or ''}\n{note}".strip()
        image.reviewed_by = "鱼体检测"
        image.reviewed_at = utcnow()
        after = {
            "review_status": image.review_status,
            "truth_species": image.truth_species,
            "notes": image.notes,
        }
        db.add(
            ReviewEvent(
                image_asset_id=image.id,
                action="fish_presence_reject",
                reviewer="鱼体检测",
                before_json=json.dumps(before, ensure_ascii=False),
                after_json=json.dumps(after, ensure_ascii=False),
            )
        )
        changed += 1

    db.commit()
    return {"batch_id": batch_id, "rejected": changed, "summary": presence_summary(db, batch_id)}


@router.get("/batches")
def api_presence_batches(db: Session = Depends(get_db)):
    batch_ids = db.scalars(select(Batch.batch_id).order_by(Batch.created_at.desc())).all()
    return [presence_summary(db, batch_id) for batch_id in batch_ids]


@router.get("/batch/{batch_id}")
def api_presence_batch(batch_id: str, db: Session = Depends(get_db)):
    try:
        return presence_summary(db, batch_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/scan")
def api_presence_scan(payload: PresenceScanRequest, db: Session = Depends(get_db)):
    try:
        return scan_batch(db, payload.batch_id, limit=payload.limit, rescan=payload.rescan)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/reject-no-fish")
def api_reject_no_fish(payload: PresenceRejectRequest, db: Session = Depends(get_db)):
    try:
        return reject_no_fish(db, payload.batch_id)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/image/{batch_id}/{image_id}")
def api_presence_image(batch_id: str, image_id: str, db: Session = Depends(get_db)):
    image = db.scalar(select(ImageAsset).where(ImageAsset.batch_id == batch_id, ImageAsset.image_id == image_id))
    if not image:
        raise HTTPException(status_code=404, detail="image not found")
    row = db.scalar(select(FishPresenceResult).where(FishPresenceResult.image_asset_id == image.id))
    if not row:
        return {"status": "not_scanned"}
    return _result_dict(row)


@router.get("/images")
def api_presence_images(
    batch_id: str,
    status: str = Query(default="no_fish"),
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    rows = db.execute(
        select(ImageAsset, FishPresenceResult)
        .join(FishPresenceResult, FishPresenceResult.image_asset_id == ImageAsset.id)
        .where(ImageAsset.batch_id == batch_id, FishPresenceResult.status == status)
        .order_by(ImageAsset.id)
        .limit(limit)
    ).all()
    return [
        {
            "batch_id": image.batch_id,
            "image_id": image.image_id,
            "review_status": image.review_status,
            "media_url": f"/media/{image.batch_id}/{image.image_id}",
            **_result_dict(result),
        }
        for image, result in rows
    ]
