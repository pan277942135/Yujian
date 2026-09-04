"""Authenticated user fish-catch archive APIs for the YuJian MVP."""

from __future__ import annotations

import io
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response
from google.cloud import storage
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.auth_api import get_current_user
from app.db import get_db
from app.factory import DOWNLOAD_RETRY, get_bucket_name
from app.models import AppUser, FishCatch, utcnow


router = APIRouter(prefix="/api/v1/catches", tags=["user-catches"])
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
IMAGE_CONTENT_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
UPLOAD_URL_PATTERN = re.compile(r"/api/v1/catches/uploads/([0-9a-f-]{36})/media$")


class CatchCreate(BaseModel):
    # image_upload_id is the normal Android path. image_url is accepted as a
    # compatibility alias for clients following the initial MVP request shape.
    image_upload_id: str | None = None
    image_url: str | None = None
    species_id: str = Field(min_length=1, max_length=128)
    species_name: str = Field(min_length=1, max_length=128)
    confidence: float = Field(ge=0, le=1)
    model_version: str = Field(min_length=1, max_length=128)
    detector_result: dict[str, Any] | None = None
    classifier_result: dict[str, Any] | None = None
    captured_at: datetime | None = None

    @field_validator("species_id", "species_name", "model_version")
    @classmethod
    def non_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class CatchOut(BaseModel):
    id: str
    image_url: str
    species_id: str
    species_name: str
    confidence: float
    model_version: str
    captured_at: datetime
    created_at: datetime


class CatchCreateResponse(BaseModel):
    catch_id: str
    saved: bool
    catch: CatchOut


class UploadedImageOut(BaseModel):
    image_upload_id: str
    image_url: str


class SpeciesCount(BaseModel):
    species_id: str
    species: str
    count: int


class CatchStatisticsOut(BaseModel):
    total_catches: int
    species_count: int
    top_species: list[SpeciesCount]
    recent_species: str | None


def _upload_object_name(user_id: str, upload_id: str, suffix: str) -> str:
    return f"user_catches/{user_id}/uploads/{upload_id}{suffix}"


def _upload_media_url(upload_id: str) -> str:
    return f"/api/v1/catches/uploads/{upload_id}/media"


def _catch_media_url(catch_id: str) -> str:
    return f"/api/v1/catches/{catch_id}/media"


def _safe_upload_id(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (ValueError, AttributeError) as exc:
        raise HTTPException(status_code=422, detail="图片上传标识无效") from exc


async def _read_image(file: UploadFile) -> tuple[bytes, str, str]:
    data = await file.read(MAX_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="图片不能为空")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="图片不能超过 25MB")
    try:
        with Image.open(io.BytesIO(data)) as image:
            oriented = ImageOps.exif_transpose(image)
            width, height = oriented.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=400, detail="图片尺寸无效或过大")
            detected_format = (image.format or "").upper()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="仅支持 JPEG、PNG、WEBP 图片") from exc
    media_type = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}.get(detected_format)
    if media_type is None:
        raise HTTPException(status_code=400, detail="仅支持 JPEG、PNG、WEBP 图片")
    return data, media_type, IMAGE_CONTENT_TYPES[media_type]


def _find_uploaded_blob(user: AppUser, upload_id: str):
    bucket_name = get_bucket_name()
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for suffix in IMAGE_CONTENT_TYPES.values():
        blob = bucket.blob(_upload_object_name(user.id, upload_id, suffix))
        if blob.exists(client):
            return client, blob
    raise HTTPException(status_code=404, detail="上传图片不存在或不属于当前用户")


def _content_type_for_name(name: str) -> str:
    return {".jpg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(PurePosixPath(name).suffix.lower(), "application/octet-stream")


def _catch_out(row: FishCatch) -> CatchOut:
    return CatchOut(
        id=row.id,
        image_url=_catch_media_url(row.id),
        species_id=row.species_id,
        species_name=row.species_name,
        confidence=row.confidence,
        model_version=row.model_version,
        captured_at=row.captured_at,
        created_at=row.created_at,
    )


def _resolve_upload_id(payload: CatchCreate) -> str:
    if payload.image_upload_id:
        return _safe_upload_id(payload.image_upload_id)
    if payload.image_url:
        matched = UPLOAD_URL_PATTERN.search(payload.image_url.strip())
        if matched:
            return _safe_upload_id(matched.group(1))
    raise HTTPException(status_code=422, detail="请先上传鱼获图片")


@router.post("/upload-image", response_model=UploadedImageOut)
async def upload_catch_image(
    image: UploadFile = File(...),
    user: AppUser = Depends(get_current_user),
) -> UploadedImageOut:
    data, media_type, suffix = await _read_image(image)
    upload_id = str(uuid.uuid4())
    try:
        bucket_name = get_bucket_name()
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(_upload_object_name(user.id, upload_id, suffix))
        blob.upload_from_string(data, content_type=media_type, if_generation_match=0)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail="鱼获图片上传失败") from exc
    return UploadedImageOut(image_upload_id=upload_id, image_url=_upload_media_url(upload_id))


@router.get("/uploads/{upload_id}/media")
def get_uploaded_catch_image(upload_id: str, user: AppUser = Depends(get_current_user)) -> Response:
    upload_id = _safe_upload_id(upload_id)
    try:
        _client, blob = _find_uploaded_blob(user, upload_id)
        content = blob.download_as_bytes(timeout=120, retry=DOWNLOAD_RETRY)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail="鱼获图片暂时无法读取") from exc
    return Response(
        content=content,
        media_type=_content_type_for_name(blob.name),
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
    )


@router.post("", response_model=CatchCreateResponse)
def create_catch(
    payload: CatchCreate,
    user: AppUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CatchCreateResponse:
    upload_id = _resolve_upload_id(payload)
    try:
        _client, blob = _find_uploaded_blob(user, upload_id)
    except HTTPException:
        raise
    catch_id = str(uuid.uuid4())
    captured_at = payload.captured_at or utcnow()
    if captured_at.tzinfo is None:
        captured_at = captured_at.replace(tzinfo=timezone.utc)
    row = FishCatch(
        id=catch_id,
        user_id=user.id,
        image_url=_catch_media_url(catch_id),
        image_object_name=blob.name,
        species_id=payload.species_id,
        species_name=payload.species_name,
        confidence=payload.confidence,
        model_version=payload.model_version,
        detector_result_json=json.dumps(payload.detector_result, ensure_ascii=False, separators=(",", ":")) if payload.detector_result else None,
        classifier_result_json=json.dumps(payload.classifier_result, ensure_ascii=False, separators=(",", ":")) if payload.classifier_result else None,
        captured_at=captured_at,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return CatchCreateResponse(catch_id=row.id, saved=True, catch=_catch_out(row))


@router.get("", response_model=list[CatchOut])
def list_catches(
    limit: int = 50,
    user: AppUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[CatchOut]:
    safe_limit = min(max(limit, 1), 100)
    rows = db.scalars(
        select(FishCatch)
        .where(FishCatch.user_id == user.id)
        .order_by(desc(FishCatch.captured_at), desc(FishCatch.created_at))
        .limit(safe_limit)
    ).all()
    return [_catch_out(row) for row in rows]


@router.get("/statistics", response_model=CatchStatisticsOut)
def catch_statistics(
    user: AppUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CatchStatisticsOut:
    total = db.scalar(select(func.count()).select_from(FishCatch).where(FishCatch.user_id == user.id)) or 0
    species_count = db.scalar(
        select(func.count(func.distinct(FishCatch.species_id))).where(FishCatch.user_id == user.id)
    ) or 0
    top_rows = db.execute(
        select(FishCatch.species_id, FishCatch.species_name, func.count().label("count"))
        .where(FishCatch.user_id == user.id)
        .group_by(FishCatch.species_id, FishCatch.species_name)
        .order_by(desc(func.count()), FishCatch.species_name)
        .limit(3)
    ).all()
    recent = db.scalar(
        select(FishCatch.species_name)
        .where(FishCatch.user_id == user.id)
        .order_by(desc(FishCatch.captured_at), desc(FishCatch.created_at))
        .limit(1)
    )
    return CatchStatisticsOut(
        total_catches=int(total),
        species_count=int(species_count),
        top_species=[SpeciesCount(species_id=row.species_id, species=row.species_name, count=int(row.count)) for row in top_rows],
        recent_species=recent,
    )


@router.get("/{catch_id}/media")
def get_catch_image(
    catch_id: str,
    user: AppUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    row = db.get(FishCatch, catch_id)
    if row is None or row.user_id != user.id:
        raise HTTPException(status_code=404, detail="鱼获图片不存在")
    try:
        client = storage.Client()
        blob = client.bucket(get_bucket_name()).blob(row.image_object_name)
        if not blob.exists(client):
            raise HTTPException(status_code=404, detail="鱼获图片不存在")
        content = blob.download_as_bytes(timeout=120, retry=DOWNLOAD_RETRY)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail="鱼获图片暂时无法读取") from exc
    return Response(
        content=content,
        media_type=_content_type_for_name(row.image_object_name),
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
    )
