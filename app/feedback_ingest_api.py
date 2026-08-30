from __future__ import annotations

import hashlib
import mimetypes
import re
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from google.cloud import storage
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.factory import get_bucket_name
from app.flywheel import VALID_FEEDBACK_TYPES, feedback_dict, record_feedback
from app.models import FeedbackEvent

router = APIRouter(tags=["feedback-ingest"])

MAX_FEEDBACK_IMAGE_BYTES = 15 * 1024 * 1024
_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
_CONTENT_TYPE_SUFFIX = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


def _safe_event_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())[:96]
    return cleaned or "event"


@router.post("/api/feedback/ingest")
async def ingest_app_feedback(
    source_event_id: str = Form(...),
    feedback_type: str = Form(...),
    source: str = Form(default="app"),
    model_version: str | None = Form(default=None),
    predicted_species: str | None = Form(default=None),
    confidence: float | None = Form(default=None),
    corrected_species: str | None = Form(default=None),
    user_note: str | None = Form(default=None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Ingest one App feedback event plus its user-provided catch image.

    This endpoint has write-only feedback semantics. The image is persisted in GCS,
    while record_feedback keeps the same idempotent source_event_id contract as
    /api/feedback. Feedback is never promoted directly to training truth; the
    existing materialize -> Review pipeline remains mandatory.
    """
    event_id = source_event_id.strip()
    if not event_id:
        raise HTTPException(status_code=400, detail="source_event_id is required")
    feedback_kind = feedback_type.strip()
    if feedback_kind not in VALID_FEEDBACK_TYPES:
        raise HTTPException(status_code=400, detail=f"invalid feedback_type: {feedback_kind}")

    existing = db.scalar(select(FeedbackEvent).where(FeedbackEvent.source_event_id == event_id))
    if existing:
        return feedback_dict(existing)

    content_type = (file.content_type or "").lower().strip()
    if content_type not in _ALLOWED_CONTENT_TYPES:
        guessed = mimetypes.guess_type(file.filename or "")[0]
        if guessed in _ALLOWED_CONTENT_TYPES:
            content_type = guessed or ""
    if content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="feedback image must be JPEG, PNG, or WEBP")

    data = await file.read(MAX_FEEDBACK_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="feedback image is empty")
    if len(data) > MAX_FEEDBACK_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="feedback image exceeds 15 MB")

    digest = hashlib.sha256(data).hexdigest()
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    suffix = _CONTENT_TYPE_SUFFIX[content_type]
    object_name = f"feedback/app/{day}/{_safe_event_component(event_id)}_{digest[:16]}{suffix}"
    bucket_name = get_bucket_name()
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(object_name)

    try:
        if not blob.exists(client):
            blob.upload_from_string(data, content_type=content_type, if_generation_match=0)
        image_gcs_uri = f"gs://{bucket_name}/{object_name}"
        return record_feedback(
            db,
            source_event_id=event_id,
            feedback_type=feedback_kind,
            source=source,
            image_gcs_uri=image_gcs_uri,
            model_version=model_version,
            predicted_species=predicted_species,
            confidence=confidence,
            corrected_species=corrected_species,
            user_note=user_note,
        )
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"feedback ingest failed: {exc}") from exc
