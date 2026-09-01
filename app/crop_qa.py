"""Read-only Crop QA queue for human inspection of reviewed boxes."""

from __future__ import annotations

import json
import mimetypes
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from google.cloud import storage
from sqlalchemy import or_, select
from sqlalchemy.orm import Session
from starlette.requests import Request

from app.db import get_db
from app.models import InferenceAsset


router = APIRouter(tags=["crop-qa"])
templates = Jinja2Templates(directory="app/templates")

REVIEWED_STATUSES = {"ACCEPTED", "TRAINING_READY"}
DEFAULT_EXPAND_RATIO = 0.15


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://") or "/" not in uri[5:]:
        raise ValueError("invalid GCS URI")
    return tuple(uri[5:].split("/", 1))  # type: ignore[return-value]


def _parse_bbox(value: Any) -> list[float] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bbox = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(0.0 <= item <= 1.0 for item in bbox):
        return None
    if bbox[2] <= 0 or bbox[3] <= 0 or bbox[0] + bbox[2] > 1.00001 or bbox[1] + bbox[3] > 1.00001:
        return None
    return bbox


def _record_document(asset: InferenceAsset, storage_client: Any = None) -> dict[str, Any]:
    """Best-effort metadata read; QA remains useful when the record is offline."""

    uri = str(getattr(asset, "record_gcs_uri", "") or "")
    if not uri:
        return {}
    try:
        if uri.startswith("gs://"):
            bucket_name, object_name = _parse_gs_uri(uri)
            client = storage_client or storage.Client()
            text = client.bucket(bucket_name).blob(object_name).download_as_text(encoding="utf-8")
            document = json.loads(text)
        else:
            from pathlib import Path

            document = json.loads(Path(uri).read_text(encoding="utf-8"))
        return document if isinstance(document, dict) else {}
    except Exception:
        return {}


def crop_qa_item(asset: InferenceAsset, *, storage_client: Any = None) -> dict[str, Any] | None:
    """Serialize one accepted inference asset for the QA page/API."""

    status = str(getattr(asset, "status", "") or "").upper()
    if status not in REVIEWED_STATUSES:
        return None
    bbox = _parse_bbox(getattr(asset, "accepted_bbox_json", None))
    if bbox is None:
        # An accepted state without a valid human box is not QA/training-ready.
        return None
    document = _record_document(asset, storage_client)
    crop = document.get("crop") if isinstance(document.get("crop"), dict) else {}
    image_id = str(getattr(asset, "image_id", "") or "")
    source_image_id = str(crop.get("source_image_id") or document.get("image_id") or image_id)
    crop_uri = str(getattr(asset, "crop_gcs_uri", "") or "").strip() or None
    return {
        "image_id": image_id,
        "source_image_id": source_image_id,
        "status": status,
        "accepted_species": getattr(asset, "accepted_species", None),
        "accepted_bbox": bbox,
        "bbox_source": "human_review",
        "crop_available": bool(crop_uri),
        "original_url": f"/media/inference/{quote(image_id, safe='')}/original",
        "crop_url": f"/media/inference/{quote(image_id, safe='')}/crop" if crop_uri else None,
        "crop_metadata": {
            "source_image_id": source_image_id,
            "crop_gcs_uri": crop_uri,
            "expand_ratio": crop.get("expand_ratio", DEFAULT_EXPAND_RATIO),
            "crop_width": crop.get("crop_width"),
            "crop_height": crop.get("crop_height"),
            "crop_path": crop.get("crop_path") or crop.get("crop_uri"),
        },
    }


@router.get("/crop-qa", response_class=HTMLResponse)
def crop_qa_page(request: Request):
    return templates.TemplateResponse(request=request, name="crop_qa.html", context={})


@router.get("/api/crop-qa")
def crop_qa_items(
    status: str = Query(default="accepted", max_length=32),
    q: str | None = Query(default=None, max_length=256),
    limit: int = Query(default=24, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    normalized = status.strip().upper()
    if normalized == "ACCEPTED":
        statuses = REVIEWED_STATUSES
    elif normalized == "TRAINING_READY":
        statuses = {"TRAINING_READY"}
    elif normalized == "ALL":
        statuses = REVIEWED_STATUSES
    else:
        raise HTTPException(status_code=400, detail="status must be accepted, training_ready, or all")

    stmt = select(InferenceAsset).where(InferenceAsset.status.in_(statuses))
    if q:
        like = f"%{q.strip()}%"
        stmt = stmt.where(or_(InferenceAsset.image_id.ilike(like), InferenceAsset.accepted_species.ilike(like)))
    assets = db.scalars(stmt.order_by(InferenceAsset.created_at.desc())).all()
    items = [item for asset in assets if (item := crop_qa_item(asset)) is not None]
    total = len(items)
    return {"status": normalized, "total": total, "offset": offset, "limit": limit, "items": items[offset : offset + limit]}


@router.get("/media/inference/{image_id}/{kind}")
def inference_media(image_id: str, kind: str, db: Session = Depends(get_db)):
    if kind not in {"original", "crop"}:
        raise HTTPException(status_code=400, detail="kind must be original or crop")
    asset = db.get(InferenceAsset, image_id)
    if not asset or str(asset.status or "").upper() not in REVIEWED_STATUSES or _parse_bbox(asset.accepted_bbox_json) is None:
        raise HTTPException(status_code=404, detail="reviewed inference asset not found")
    uri = asset.image_gcs_uri if kind == "original" else asset.crop_gcs_uri
    if not uri:
        raise HTTPException(status_code=404, detail="crop is not available")
    try:
        bucket_name, object_name = _parse_gs_uri(uri)
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(object_name)
        try:
            exists = bool(blob.exists(client))
        except TypeError:  # lightweight fakes and older storage clients
            exists = bool(blob.exists())
        if not exists:
            raise HTTPException(status_code=404, detail="GCS object not found")
        content = blob.download_as_bytes(timeout=120)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail="unable to read inference image") from exc
    media_type = mimetypes.guess_type(object_name)[0] or "image/jpeg"
    return Response(content=content, media_type=media_type, headers={"Cache-Control": "private, max-age=300"})


__all__ = ["crop_qa_item", "crop_qa_items", "inference_media", "router", "templates"]
