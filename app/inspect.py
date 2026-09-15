from __future__ import annotations

import json
import mimetypes
from datetime import datetime, timezone
from io import BytesIO
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from PIL import Image
from pydantic import BaseModel
from sqlalchemy import or_, select
from sqlalchemy.orm import Session
from starlette.requests import Request

from app.data_policy import truth_filter_clause
from app.db import get_db
from app.dedupe import ImageFingerprint
from app.models import DatasetVersion, ImageAsset, ReviewEvent
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


def _manifest_bbox(value) -> list[float] | None:
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
    if not all(0.0 <= item <= 1.0 for item in result):
        return None
    if result[2] <= 0 or result[3] <= 0 or result[0] + result[2] > 1.00001 or result[1] + result[3] > 1.00001:
        return None
    return [round(item, 6) for item in result]


def _accepted_pool_media_url(pool_key: str, *, variant: str | None = None) -> str:
    query = f"pool_key={quote(str(pool_key), safe='')}"
    if variant:
        query += f"&variant={quote(variant, safe='')}"
    return f"/api/inspect/accepted-pool/media?{query}"


def _accepted_pool_items(
    *,
    species: str | None,
    batch_id: str | None,
    q: str | None,
    offset: int,
    limit: int,
    db: Session,
) -> dict:
    """Build the Accepted Pool inspection view from the materialised crop index."""

    from app.accepted_pool import list_accepted_pool_manifest_rows

    rows, total = list_accepted_pool_manifest_rows(species=species, batch_id=batch_id, q=q)
    page_rows = rows[offset : offset + limit]
    pairs = [
        (
            str(row.get("source_batch") or row.get("batch_id") or "").strip(),
            str(row.get("image_id") or "").strip(),
        )
        for row in page_rows
    ]
    pairs = [(batch, image_id) for batch, image_id in pairs if batch and image_id]
    image_map = {}
    if pairs:
        image_map = {
            (str(image.batch_id), str(image.image_id)): image
            for image in db.scalars(
                select(ImageAsset).where(
                    or_(*((ImageAsset.batch_id == batch) & (ImageAsset.image_id == image_id) for batch, image_id in pairs))
                )
            ).all()
        }
    image_ids = [image.id for image in image_map.values()]
    presence_map = {}
    duplicate_map = {}
    if image_ids:
        presence_map = {
            row.image_asset_id: row
            for row in db.scalars(select(FishPresenceResult).where(FishPresenceResult.image_asset_id.in_(image_ids))).all()
        }
        duplicate_map = {
            row.image_asset_id: row
            for row in db.scalars(select(ImageFingerprint).where(ImageFingerprint.image_asset_id.in_(image_ids))).all()
        }

    items = []
    for row in page_rows:
        batch = str(row.get("source_batch") or row.get("batch_id") or "").strip()
        image_id = str(row.get("image_id") or "").strip()
        image = image_map.get((batch, image_id))
        presence = _presence_meta(presence_map.get(image.id)) if image else _presence_meta(None)
        fingerprint = duplicate_map.get(image.id) if image else None
        pool_key = str(row.get("pool_key") or f"{batch}:{image_id}").strip()
        crop_url = _accepted_pool_media_url(pool_key)
        items.append(
            {
                "batch_id": batch,
                "image_id": image_id,
                # Accepted Pool inspection must show the durable crop.  The
                # original remains metadata-only for traceability and is never
                # used as the primary image in this view.
                "media_url": crop_url if str(row.get("crop_path") or "").strip() else "",
                "thumbnail_url": _accepted_pool_media_url(pool_key, variant="thumbnail")
                if str(row.get("crop_path") or "").strip()
                else "",
                "source_image_url": (
                    f"/media/{quote(batch, safe='')}/{quote(image_id, safe='')}" if batch and image_id else ""
                ),
                "claimed_species": image.claimed_species if image else None,
                "truth_species": str(row.get("species_name") or row.get("species") or "").strip()
                or (image.truth_species if image else None),
                "review_status": "approved",
                "notes": image.notes if image else "",
                "accepted_pool": True,
                "media_source": "accepted_pool_crop",
                "accepted_bbox": _manifest_bbox(row.get("accepted_bbox") or row.get("bbox")),
                "presence": presence,
                "duplicate": {
                    "group": fingerprint.duplicate_group if fingerprint else None,
                    "is_duplicate": bool(
                        fingerprint and fingerprint.duplicate_group and not fingerprint.is_representative
                    ),
                    "kind": fingerprint.duplicate_kind if fingerprint else None,
                },
            }
        )
    return {"source": "accepted_pool", "total": total, "offset": offset, "limit": limit, "items": items}


@router.get("/inspect", response_class=HTMLResponse)
def inspect_page(request: Request):
    return templates.TemplateResponse(request=request, name="inspect.html", context={})


@router.get("/api/inspect/images")
def inspect_images(
    batch_id: str | None = None,
    species: str | None = None,
    review_status: str | None = Query(default="all"),
    presence: str | None = None,
    source: str = Query(default="original", max_length=32),
    new_since_latest: bool = Query(default=False),
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
    source = str(source or "original").strip().lower()
    if source not in {"original", "accepted_pool"}:
        raise HTTPException(status_code=400, detail="invalid inspect source")
    if source == "accepted_pool":
        if statuses and "approved" not in statuses:
            return {"source": source, "total": 0, "offset": offset, "limit": limit, "items": []}
        result = _accepted_pool_items(
            species=species,
            batch_id=batch_id,
            q=q,
            offset=offset,
            limit=limit,
            db=db,
        )
        if presence:
            result["items"] = [item for item in result["items"] if item["presence"]["status"] == presence]
            # Presence is an optional enrichment for Accepted Pool rows.  Keep
            # pagination semantics honest after applying this legacy filter.
            result["total"] = len(result["items"])
        return result

    stmt = select(ImageAsset)
    if batch_id:
        stmt = stmt.where(ImageAsset.batch_id == batch_id)
    if statuses:
        stmt = stmt.where(ImageAsset.review_status.in_(statuses))
    if species:
        stmt = stmt.where(truth_filter_clause(species))
    if new_since_latest:
        latest = db.scalar(select(DatasetVersion).order_by(DatasetVersion.created_at.desc()).limit(1))
        if latest and latest.source_cutoff_at:
            stmt = stmt.where(ImageAsset.updated_at > latest.source_cutoff_at)
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
    duplicate_map = {}
    if image_ids:
        presence_map = {
            row.image_asset_id: row
            for row in db.scalars(select(FishPresenceResult).where(FishPresenceResult.image_asset_id.in_(image_ids))).all()
        }
        duplicate_map = {
            row.image_asset_id: row
            for row in db.scalars(select(ImageFingerprint).where(ImageFingerprint.image_asset_id.in_(image_ids))).all()
        }

    filtered = []
    for image in images:
        p = _presence_meta(presence_map.get(image.id))
        if presence and p["status"] != presence:
            continue
        fp = duplicate_map.get(image.id)
        filtered.append(
            {
                "batch_id": image.batch_id,
                "image_id": image.image_id,
                "media_url": f"/media/{image.batch_id}/{image.image_id}",
                "thumbnail_url": f"/media/{image.batch_id}/{image.image_id}?variant=thumbnail",
                "claimed_species": image.claimed_species,
                "truth_species": image.truth_species,
                "review_status": image.review_status,
                "notes": image.notes or "",
                "presence": p,
                "duplicate": {
                    "group": fp.duplicate_group if fp else None,
                    "is_duplicate": bool(fp and fp.duplicate_group and not fp.is_representative),
                    "kind": fp.duplicate_kind if fp else None,
                },
            }
        )

    total = len(filtered)
    return {
        "source": "original",
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": filtered[offset : offset + limit],
    }


@router.get("/api/inspect/accepted-pool/media")
def inspect_accepted_pool_media(
    pool_key: str = Query(..., min_length=1, max_length=512),
    variant: str | None = Query(default=None, max_length=16),
) -> Response:
    """Serve only the materialised Accepted Pool crop for an inspect card."""

    from app.accepted_pool import accepted_pool_crop_object_name, find_accepted_pool_manifest_row
    from app.platform.services import crop_dataset

    if variant not in {None, "thumbnail"}:
        raise HTTPException(status_code=400, detail="invalid media variant")
    try:
        row = find_accepted_pool_manifest_row(pool_key)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Accepted Pool 清单暂不可用") from exc
    if row is None:
        raise HTTPException(status_code=404, detail="Accepted Pool 图片不存在")
    try:
        object_name = accepted_pool_crop_object_name(row)
        client, bucket = crop_dataset._storage()
        blob = bucket.blob(object_name)
        if not blob.exists(client):
            raise HTTPException(status_code=404, detail="Accepted Pool Crop 尚未生成")
        content = blob.download_as_bytes(timeout=120)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Accepted Pool Crop 读取失败") from exc

    if variant == "thumbnail":
        try:
            with Image.open(BytesIO(content)) as source:
                preview = source.convert("RGB")
                preview.thumbnail((320, 320), Image.Resampling.LANCZOS)
                output = BytesIO()
                preview.save(output, format="WEBP", quality=78, method=4)
            return Response(
                content=output.getvalue(),
                media_type="image/webp",
                headers={"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"},
            )
        except Exception:
            # The full crop remains the controlled fallback for an unusual
            # image encoding; it is still the Accepted Pool crop, never source.
            pass
    media_type = mimetypes.guess_type(str(row.get("crop_path") or ""))[0] or "image/jpeg"
    return Response(
        content=content,
        media_type=media_type,
        headers={"Cache-Control": "private, max-age=300", "X-Content-Type-Options": "nosniff"},
    )


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
    before = _presence_meta(row)
    if not row:
        if payload.status is None:
            return before
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
        db.flush()
    else:
        saved = {}
        if row.evidence_json:
            try:
                saved = json.loads(row.evidence_json)
            except Exception:
                saved = {}

        if payload.status is None:
            if saved.get("created_by_override") and not (saved.get("objects") or saved.get("labels")):
                db.delete(row)
                db.add(
                    ReviewEvent(
                        image_asset_id=image.id,
                        action="presence_override_clear",
                        reviewer="数据检查",
                        before_json=json.dumps(before, ensure_ascii=False),
                        after_json=json.dumps(_presence_meta(None), ensure_ascii=False),
                    )
                )
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
        db.flush()

    after = _presence_meta(row)
    db.add(
        ReviewEvent(
            image_asset_id=image.id,
            action="presence_override_update",
            reviewer="数据检查",
            before_json=json.dumps(before, ensure_ascii=False),
            after_json=json.dumps(after, ensure_ascii=False),
        )
    )
    db.commit()
    if row in db:
        db.refresh(row)
        return _presence_meta(row)
    return after
