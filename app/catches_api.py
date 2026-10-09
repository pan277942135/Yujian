"""Authenticated user fish-catch archive APIs for the YuJian MVP."""

from __future__ import annotations

import io
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response
from google.cloud import storage
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import desc, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth_api import get_current_user
from app.db import get_db
from app.factory import DOWNLOAD_RETRY, get_bucket_name
from app.models import AppUser, FishBsideJob, FishCatch, utcnow
from app.platform.models import PlatformOperationLog
from app.services.fish_bside_jobs import PENDING, PROCESSING, enqueue_fish_bside_job


router = APIRouter(prefix="/api/v1/catches", tags=["user-catches"])
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
IMAGE_CONTENT_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
UPLOAD_URL_PATTERN = re.compile(r"/api/v1/catches/uploads/([0-9a-f-]{36})/media$")


class CatchCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    length_cm: float | None = Field(default=None, gt=0, le=1000, allow_inf_nan=False)
    weight_kg: float | None = Field(default=None, gt=0, le=1000, allow_inf_nan=False)
    location: str | None = Field(default=None, max_length=512)
    story: str | None = Field(default=None, max_length=4096)
    client_record_id: str | None = Field(default=None, max_length=128)

    @field_validator("species_id", "species_name", "model_version")
    @classmethod
    def non_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("location", "story", "client_record_id")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        value = value.strip() if value is not None else None
        return value or None


class CatchOut(BaseModel):
    id: str
    image_url: str
    species_id: str
    species_name: str
    confidence: float
    model_version: str
    length_cm: float | None = None
    weight_kg: float | None = None
    location: str | None = None
    story: str | None = None
    client_record_id: str | None = None
    captured_at: datetime
    created_at: datetime
    bside_status: str = "NONE"
    bside_uri: str | None = None


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


class BsideJobOut(BaseModel):
    job_id: str | None = None
    status: str
    result_uri: str | None = None


class CatchStatisticsOut(BaseModel):
    total_catches: int
    species_count: int
    top_species: list[SpeciesCount]
    recent_species: str | None


class CatchCapabilitiesOut(BaseModel):
    metadata_version: int = 1
    idempotency_keys: bool = True
    lookup_by_client_record_id: bool = True


def _upload_object_name(user_id: str, upload_id: str, suffix: str) -> str:
    return f"user_catches/{user_id}/uploads/{upload_id}{suffix}"


def _upload_media_url(upload_id: str) -> str:
    return f"/api/v1/catches/uploads/{upload_id}/media"


def _catch_media_url(catch_id: str) -> str:
    return f"/api/v1/catches/{catch_id}/media"


def _bside_media_url(catch_id: str) -> str:
    return f"/api/v1/catches/{catch_id}/bside-media"


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


def _legacy_classifier_result(row: FishCatch) -> dict[str, Any]:
    """Return the old classifier payload only for rows marked legacy."""
    if int(row.metadata_version or 0) > 0 or not row.classifier_result_json:
        return {}
    try:
        value = json.loads(row.classifier_result_json)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _legacy_numeric(value: Any, maximum: float) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if 0 < number <= maximum and number != float("inf") and number == number else None


def _same_idempotent_request(
    row: FishCatch,
    payload: CatchCreate,
    length_cm: float | None,
    weight_kg: float | None,
    location: str | None,
    story: str | None,
) -> bool:
    if not (
        row.species_id == payload.species_id
        and row.species_name == payload.species_name
        and row.model_version == payload.model_version
        and row.confidence == payload.confidence
        and row.length_cm == length_cm
        and row.weight_kg == weight_kg
        and row.location == (location or None)
        and row.story == (story or None)
    ):
        return False
    if payload.captured_at is None:
        return True
    actual = row.captured_at
    expected = payload.captured_at
    if actual.tzinfo is None and expected.tzinfo is not None:
        actual = actual.replace(tzinfo=expected.tzinfo)
    elif expected.tzinfo is None and actual.tzinfo is not None:
        expected = expected.replace(tzinfo=actual.tzinfo)
    elif actual.tzinfo is None and expected.tzinfo is None:
        actual = actual.replace(tzinfo=timezone.utc)
        expected = expected.replace(tzinfo=timezone.utc)
    return actual == expected


def _catch_out(row: FishCatch) -> CatchOut:
    # Formal business columns always win. The JSON compatibility path is
    # limited to the same legacy row and only fills columns that are NULL.
    legacy = _legacy_classifier_result(row)
    length_cm = row.length_cm if row.length_cm is not None else _legacy_numeric(
        legacy.get("length_cm", legacy.get("length")), 1000
    )
    weight_kg = row.weight_kg if row.weight_kg is not None else _legacy_numeric(
        legacy.get("weight_kg", legacy.get("weight")), 1000
    )
    location = row.location if row.location is not None else legacy.get("location", legacy.get("location_name"))
    story = row.story if row.story is not None else legacy.get("story")
    if not isinstance(location, str):
        location = None
    if not isinstance(story, str):
        story = None
    return CatchOut(
        id=row.id,
        image_url=_catch_media_url(row.id),
        species_id=row.species_id,
        species_name=row.species_name,
        confidence=row.confidence,
        model_version=row.model_version,
        length_cm=length_cm,
        weight_kg=weight_kg,
        location=location or None,
        story=story or None,
        client_record_id=row.client_record_id,
        captured_at=row.captured_at,
        created_at=row.created_at,
        bside_status=str(row.bside_status or "NONE"),
        bside_uri=_bside_media_url(row.id) if row.bside_status == "READY" and row.bside_result_object_name else None,
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
    classifier = payload.classifier_result or {}
    # Explicit top-level nulls are authoritative; fall back to the historic
    # classifier envelope only when an older client omitted that top-level key.
    length_cm = payload.length_cm if "length_cm" in payload.model_fields_set else _legacy_numeric(
        classifier.get("length_cm", classifier.get("length")), 1000
    )
    weight_kg = payload.weight_kg if "weight_kg" in payload.model_fields_set else _legacy_numeric(
        classifier.get("weight_kg", classifier.get("weight")), 1000
    )
    location = payload.location if "location" in payload.model_fields_set else classifier.get(
        "location", classifier.get("location_name")
    )
    story = payload.story if "story" in payload.model_fields_set else classifier.get("story")
    if not isinstance(location, str):
        location = None
    if not isinstance(story, str):
        story = None
    if payload.client_record_id:
        existing = db.scalar(
            select(FishCatch).where(
                FishCatch.user_id == user.id,
                FishCatch.client_record_id == payload.client_record_id,
            )
        )
        if existing is not None:
            if not _same_idempotent_request(existing, payload, length_cm, weight_kg, location, story):
                raise HTTPException(status_code=409, detail="该客户端记录标识已用于不同鱼获数据")
            return CatchCreateResponse(catch_id=existing.id, saved=True, catch=_catch_out(existing))

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
        length_cm=length_cm,
        weight_kg=weight_kg,
        location=location or None,
        story=story or None,
        metadata_version=1,
        client_record_id=payload.client_record_id,
        captured_at=captured_at,
    )
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        if payload.client_record_id:
            existing = db.scalar(
                select(FishCatch).where(
                    FishCatch.user_id == user.id,
                    FishCatch.client_record_id == payload.client_record_id,
                )
            )
            if existing is not None and _same_idempotent_request(
                existing, payload, length_cm, weight_kg, location, story
            ):
                return CatchCreateResponse(catch_id=existing.id, saved=True, catch=_catch_out(existing))
        raise
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


@router.get("/capabilities", response_model=CatchCapabilitiesOut)
def catch_capabilities(
    user: AppUser = Depends(get_current_user),
) -> CatchCapabilitiesOut:
    # Authentication prevents exposing unnecessary API surface to anonymous callers.
    del user
    return CatchCapabilitiesOut()


@router.get("/by-client-record/{client_record_id}", response_model=CatchOut)
def get_catch_by_client_record_id(
    client_record_id: str,
    user: AppUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CatchOut:
    """Resolve an idempotent client save/migration key for crash recovery."""
    key = client_record_id.strip()
    if not key or len(key) > 128:
        raise HTTPException(status_code=422, detail="客户端记录标识无效")
    row = db.scalar(
        select(FishCatch).where(FishCatch.user_id == user.id, FishCatch.client_record_id == key)
    )
    if row is None:
        raise HTTPException(status_code=404, detail="鱼获记录不存在")
    return _catch_out(row)


@router.get("/{catch_id}", response_model=CatchOut)
def get_catch(
    catch_id: str,
    user: AppUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CatchOut:
    return _catch_out(_owned_catch_or_404(catch_id, user, db))


def _owned_catch_or_404(catch_id: str, user: AppUser, db: Session) -> FishCatch:
    row = db.get(FishCatch, catch_id)
    if row is None or row.user_id != user.id:
        raise HTTPException(status_code=404, detail="鱼获记录不存在")
    return row


@router.post("/{catch_id}/bside", response_model=BsideJobOut)
def create_bside_job(
    catch_id: str,
    background_tasks: BackgroundTasks,
    user: AppUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> BsideJobOut:
    """Create one durable B-side job, or return the in-flight one on repeats."""

    row = _owned_catch_or_404(catch_id, user, db)
    active = db.scalar(
        select(FishBsideJob)
        .where(
            FishBsideJob.fish_record_id == row.id,
            FishBsideJob.status.in_([PENDING, PROCESSING]),
        )
        .order_by(FishBsideJob.created_at.desc())
    )
    if active is not None:
        return BsideJobOut(job_id=active.id, status="GENERATING")
    if row.bside_status == "READY" and row.bside_job_id:
        return BsideJobOut(job_id=row.bside_job_id, status="READY", result_uri=_bside_media_url(row.id))

    job = FishBsideJob(
        id=str(uuid.uuid4()),
        fish_record_id=row.id,
        user_id=user.id,
        status=PENDING,
    )
    row.bside_status = "GENERATING"
    row.bside_result_uri = None
    row.bside_result_object_name = None
    row.bside_generated_at = None
    row.bside_job_id = job.id
    db.add(job)
    db.add(
        PlatformOperationLog(
            operation_type="BSIDE_JOB_CREATED",
            resource_type="fish_bside_job",
            resource_id=job.id,
            status=PENDING,
            message="用户主动触发渔获 B 面生成",
            actor=f"user:{user.id}",
        )
    )
    db.commit()
    # The task is first committed to the durable DB queue.  Dispatch is
    # intentionally post-commit so duplicate taps cannot create duplicate work.
    background_tasks.add_task(enqueue_fish_bside_job, job.id)
    return BsideJobOut(job_id=job.id, status="GENERATING")


@router.get("/{catch_id}/bside-status", response_model=BsideJobOut)
def get_bside_status(
    catch_id: str,
    user: AppUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> BsideJobOut:
    row = _owned_catch_or_404(catch_id, user, db)
    return BsideJobOut(
        job_id=row.bside_job_id,
        status=str(row.bside_status or "NONE"),
        result_uri=_bside_media_url(row.id) if row.bside_status == "READY" and row.bside_result_object_name else None,
    )


@router.get("/{catch_id}/bside-media")
def get_bside_result(
    catch_id: str,
    user: AppUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    row = _owned_catch_or_404(catch_id, user, db)
    if row.bside_status != "READY" or not row.bside_result_object_name:
        raise HTTPException(status_code=404, detail="B 面结果尚未生成")
    try:
        blob = storage.Client().bucket(get_bucket_name()).blob(row.bside_result_object_name)
        if not blob.exists():
            raise HTTPException(status_code=404, detail="B 面结果不存在")
        content = blob.download_as_bytes(timeout=120, retry=DOWNLOAD_RETRY)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail="B 面结果暂时无法读取") from exc
    return Response(
        content=content,
        media_type="image/png",
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
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
