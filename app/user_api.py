from __future__ import annotations

import io
import json
import re
import uuid
from datetime import date, datetime, timezone
from pathlib import PurePosixPath
from typing import Any

import bcrypt
from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import Response
from google.cloud import storage
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth import create_access_token, decode_access_token
from app.db import get_db
from app.factory import get_bucket_name
from app.models import FishCatch, User


router = APIRouter(prefix="/api/v1", tags=["mvp-user"])

_ALLOWED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
_MAX_IMAGE_BYTES = 15 * 1024 * 1024
_USERNAME_RE = re.compile(r"^[a-z0-9_.-]{3,64}$")
_ASSET_RE = re.compile(r"^[0-9a-f]{32}\.(?:jpg|jpeg|png|webp)$")


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=6, max_length=128)
    nickname: str = Field(min_length=1, max_length=128)


class LoginRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=6, max_length=128)


class CatchCreateRequest(BaseModel):
    image_url: str = Field(default="", max_length=2048)
    species_id: str = Field(min_length=1, max_length=128)
    species_name: str = Field(min_length=1, max_length=128)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    model_version: str = Field(default="", max_length=128)
    detector_result: dict[str, Any] | None = None
    classifier_result: dict[str, Any] | None = None
    captured_at: datetime | None = None


def _username(value: str) -> str:
    normalized = value.strip().lower()
    if not _USERNAME_RE.fullmatch(normalized):
        raise HTTPException(
            status_code=422,
            detail="username must contain 3-64 lowercase letters, numbers, '.', '_' or '-'",
        )
    return normalized


def _non_empty(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise HTTPException(status_code=422, detail=f"{field_name} cannot be empty")
    return normalized


def _reject_data_url(value: str) -> None:
    if value.strip().lower().startswith("data:"):
        raise HTTPException(status_code=422, detail="image_url must reference an uploaded asset, not base64")


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _user_payload(user: User) -> dict[str, str]:
    return {
        "id": user.id,
        "username": user.username,
        "nickname": user.nickname,
    }


def _catch_payload(row: FishCatch) -> dict[str, Any]:
    return {
        "catch_id": row.id,
        "image_url": row.image_url,
        "species_id": row.species_id,
        "species_name": row.species_name,
        "confidence": row.confidence,
        "model_version": row.model_version,
        "captured_at": _iso(row.captured_at),
        "created_at": _iso(row.created_at),
    }


def get_current_user(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=401,
            detail="需要 Bearer 登录令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )
    claims = decode_access_token(authorization[7:].strip())
    if not claims:
        raise HTTPException(
            status_code=401,
            detail="登录令牌无效或已过期",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id = str(claims.get("sub", "")).strip()
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="用户不存在",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


@router.post("/auth/register", status_code=201)
def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    username = _username(payload.username)
    nickname = _non_empty(payload.nickname, "nickname")
    if not payload.password.strip():
        raise HTTPException(status_code=422, detail="password cannot be empty")

    if db.scalar(select(User).where(User.username == username)) is not None:
        raise HTTPException(status_code=409, detail={"error": "USERNAME_TAKEN"})

    user = User(
        id=f"user_{uuid.uuid4().hex}",
        username=username,
        password_hash=bcrypt.hashpw(payload.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8"),
        nickname=nickname,
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail={"error": "USERNAME_TAKEN"}) from exc
    db.refresh(user)
    return {"user_id": user.id, "username": user.username, "nickname": user.nickname}


@router.post("/auth/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    username = _username(payload.username)
    user = db.scalar(select(User).where(User.username == username))
    if user is None or not bcrypt.checkpw(payload.password.encode("utf-8"), user.password_hash.encode("utf-8")):
        raise HTTPException(
            status_code=401,
            detail="账号或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {
        "access_token": create_access_token(user),
        "token_type": "bearer",
        "user": _user_payload(user),
    }


@router.post("/catches/upload-image")
async def upload_catch_image(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    content_type = (file.content_type or "").lower().strip()
    if content_type not in _ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="只支持 JPEG、PNG 或 WEBP 图片")

    data = await file.read(_MAX_IMAGE_BYTES + 1)
    if len(data) > _MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="图片不能超过 15MB")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="上传内容不是有效图片") from exc

    now = datetime.now(timezone.utc)
    token = uuid.uuid4().hex
    suffix = _ALLOWED_IMAGE_TYPES[content_type]
    object_name = f"app_feedback/catches/{user.id}/{now:%Y/%m/%d}/{token}{suffix}"
    try:
        blob = storage.Client().bucket(get_bucket_name()).blob(object_name)
        blob.upload_from_string(data, content_type=content_type)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="鱼获图片存储暂不可用") from exc

    relative_url = f"/api/v1/catches/media/{user.id}/{now:%Y/%m/%d}/{token}{suffix}"
    return {
        "url": relative_url,
        "image_url": relative_url,
        "object_name": object_name,
        "content_type": content_type,
        "size_bytes": len(data),
    }


def _validate_asset_parts(user_id: str, year: str, month: str, day: str, asset_name: str) -> date:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", user_id):
        raise HTTPException(status_code=400, detail="invalid user id")
    if not re.fullmatch(r"\d{4}", year) or not re.fullmatch(r"\d{2}", month) or not re.fullmatch(r"\d{2}", day):
        raise HTTPException(status_code=400, detail="invalid asset date")
    try:
        asset_date = date(int(year), int(month), int(day))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid asset date") from exc
    if "/" in asset_name or "\\" in asset_name or not _ASSET_RE.fullmatch(asset_name):
        raise HTTPException(status_code=400, detail="invalid asset name")
    return asset_date


@router.get("/catches/media/{user_id}/{year}/{month}/{day}/{asset_name}")
def get_catch_media(
    user_id: str,
    year: str,
    month: str,
    day: str,
    asset_name: str,
    user: User = Depends(get_current_user),
) -> Response:
    if user_id != user.id:
        raise HTTPException(status_code=404, detail="asset not found")
    asset_date = _validate_asset_parts(user_id, year, month, day, asset_name)
    object_name = f"app_feedback/catches/{user.id}/{asset_date:%Y/%m/%d}/{asset_name}"
    suffix = PurePosixPath(asset_name).suffix.lower()
    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }[suffix]
    try:
        blob = storage.Client().bucket(get_bucket_name()).blob(object_name)
        if not blob.exists():
            raise HTTPException(status_code=404, detail="asset not found")
        data = blob.download_as_bytes()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail="鱼获图片读取暂不可用") from exc
    return Response(
        content=data,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=86400", "X-Content-Type-Options": "nosniff"},
    )


@router.post("/catches", status_code=201)
def create_catch(
    payload: CatchCreateRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    image_url = payload.image_url.strip()
    _reject_data_url(image_url)
    species_id = _non_empty(payload.species_id, "species_id")
    species_name = _non_empty(payload.species_name, "species_name")
    captured_at = payload.captured_at or datetime.now(timezone.utc)
    if captured_at.tzinfo is None:
        captured_at = captured_at.replace(tzinfo=timezone.utc)

    row = FishCatch(
        id=f"catch_{uuid.uuid4().hex}",
        user_id=user.id,
        image_url=image_url,
        species_id=species_id,
        species_name=species_name,
        confidence=float(payload.confidence),
        model_version=payload.model_version.strip(),
        detector_result_json=json.dumps(payload.detector_result, ensure_ascii=False, separators=(",", ":"))
        if payload.detector_result is not None
        else None,
        classifier_result_json=json.dumps(payload.classifier_result, ensure_ascii=False, separators=(",", ":"))
        if payload.classifier_result is not None
        else None,
        captured_at=captured_at,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return {"catch_id": row.id, "saved": True}


@router.get("/catches")
def list_catches(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[dict[str, Any]]:
    rows = db.scalars(
        select(FishCatch)
        .where(FishCatch.user_id == user.id)
        .order_by(FishCatch.created_at.desc(), FishCatch.id.desc())
        .offset(offset)
        .limit(limit)
    ).all()
    return [_catch_payload(row) for row in rows]


@router.get("/catches/statistics")
def catch_statistics(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    total = int(
        db.scalar(select(func.count(FishCatch.id)).where(FishCatch.user_id == user.id)) or 0
    )
    species_count = int(
        db.scalar(
            select(func.count(func.distinct(FishCatch.species_id))).where(FishCatch.user_id == user.id)
        )
        or 0
    )
    grouped = db.execute(
        select(
            FishCatch.species_id,
            FishCatch.species_name,
            func.count(FishCatch.id).label("catch_count"),
        )
        .where(FishCatch.user_id == user.id)
        .group_by(FishCatch.species_id, FishCatch.species_name)
        .order_by(func.count(FishCatch.id).desc(), FishCatch.species_name.asc())
        .limit(10)
    ).all()
    recent = db.scalars(
        select(FishCatch)
        .where(FishCatch.user_id == user.id)
        .order_by(FishCatch.created_at.desc(), FishCatch.id.desc())
        .limit(1)
    ).first()
    return {
        "total_catches": total,
        "species_count": species_count,
        "top_species": [
            {"species": row.species_name, "species_id": row.species_id, "count": int(row.catch_count)}
            for row in grouped
        ],
        "recent_species": recent.species_name if recent else None,
        "recent_species_id": recent.species_id if recent else None,
    }


@router.get("/catches/{catch_id}")
def get_catch(
    catch_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    row = db.scalar(
        select(FishCatch).where(FishCatch.id == catch_id, FishCatch.user_id == user.id)
    )
    if row is None:
        raise HTTPException(status_code=404, detail="catch not found")
    return _catch_payload(row)
