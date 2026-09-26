"""Password + JWT authentication for the YuJian consumer App.

This module is intentionally independent from the password-protected Model
Factory console.  Console cookies never grant access to a user's fish archive.
"""

from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import PurePosixPath

import bcrypt
import jwt
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from google.cloud import storage
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.factory import DOWNLOAD_RETRY, get_bucket_name
from app.models import AppUser, UserPrivacyAudit, UserPrivacySetting, utcnow


router = APIRouter(prefix="/api/v1/auth", tags=["app-auth"])
me_router = APIRouter(prefix="/api/v1/me", tags=["app-profile-privacy"])
bearer_scheme = HTTPBearer(auto_error=False)
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{3,32}$")
TOKEN_ALGORITHM = "HS256"
TOKEN_ISSUER = "yujian-app"
TOKEN_TTL_DAYS = 30
MAX_AVATAR_BYTES = 5 * 1024 * 1024
MAX_AVATAR_PIXELS = 12_000_000
AVATAR_CONTENT_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
AI_CONSENT_VERSION = "AI_MODEL_IMPROVEMENT_V1"
AI_CONSENT_SOURCES = {"settings", "species_correction_prompt"}


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    # bcrypt intentionally caps password input at 72 UTF-8 bytes.
    password: str = Field(min_length=6, max_length=72)
    nickname: str = Field(min_length=1, max_length=20)


class LoginRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=1, max_length=72)


class UserOut(BaseModel):
    id: str
    username: str
    nickname: str
    avatar_url: str | None = None


class RegisterResponse(BaseModel):
    user_id: str
    username: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class ProfileUpdateRequest(BaseModel):
    nickname: str = Field(min_length=1, max_length=20)

    @field_validator("nickname")
    @classmethod
    def normalize_nickname(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("昵称不能为空")
        if len(normalized) > 20:
            raise ValueError("昵称不能超过 20 个字符")
        return normalized


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=72)
    new_password: str = Field(min_length=6, max_length=72)


class ChangePasswordResponse(BaseModel):
    ok: bool = True


class AiModelImprovementOut(BaseModel):
    enabled: bool
    updated_at: datetime | None = None
    consent_version: str | None = None


class PrivacyOut(BaseModel):
    ai_model_improvement: AiModelImprovementOut


class AiModelImprovementUpdate(BaseModel):
    enabled: bool
    consent_version: str = Field(min_length=1, max_length=128)
    source: str = Field(min_length=1, max_length=64)

    @field_validator("consent_version")
    @classmethod
    def exact_version(cls, value: str) -> str:
        if value != AI_CONSENT_VERSION:
            raise ValueError("不支持的授权版本")
        return value

    @field_validator("source")
    @classmethod
    def valid_source(cls, value: str) -> str:
        if value not in AI_CONSENT_SOURCES:
            raise ValueError("无效的授权来源")
        return value


def _normalize_username(value: str) -> str:
    username = value.strip()
    if not USERNAME_PATTERN.fullmatch(username):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="账号需为 3–32 位字母、数字、下划线或连字符",
        )
    return username


def _jwt_secret() -> str:
    secret = os.getenv("USER_JWT_SECRET", "").strip()
    if secret:
        return secret
    # Local development and the isolated test suite remain usable without a
    # secret manager.  Cloud Run is never allowed to issue tokens with this
    # fallback; deployment config supplies USER_JWT_SECRET from Secret Manager.
    if os.getenv("K_SERVICE"):
        raise HTTPException(status_code=503, detail="用户登录服务尚未完成安全配置")
    return "yujian-local-development-only-jwt-secret"


def _user_out(user: AppUser) -> UserOut:
    return UserOut(id=user.id, username=user.username, nickname=user.nickname, avatar_url=user.avatar_url)


def _privacy_out(setting: UserPrivacySetting | None) -> PrivacyOut:
    return PrivacyOut(
        ai_model_improvement=AiModelImprovementOut(
            enabled=bool(setting and setting.ai_model_improvement_enabled),
            updated_at=setting.ai_model_improvement_updated_at if setting else None,
            consent_version=setting.ai_model_improvement_consent_version if setting else None,
        )
    )


def _avatar_object_name(user_id: str, avatar_id: str, suffix: str) -> str:
    return f"user_avatars/{user_id}/{avatar_id}{suffix}"


def _avatar_media_url() -> str:
    return "/api/v1/me/avatar/media"


async def _read_avatar(file: UploadFile) -> tuple[bytes, str, str]:
    data = await file.read(MAX_AVATAR_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="头像不能为空")
    if len(data) > MAX_AVATAR_BYTES:
        raise HTTPException(status_code=413, detail="头像不能超过 5MB")
    try:
        with Image.open(BytesIO(data)) as image:
            oriented = ImageOps.exif_transpose(image)
            width, height = oriented.size
            if width <= 0 or height <= 0 or width * height > MAX_AVATAR_PIXELS:
                raise HTTPException(status_code=400, detail="头像尺寸无效或过大")
            detected_format = (image.format or "").upper()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="仅支持 JPEG、PNG、WEBP 图片") from exc
    media_type = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}.get(detected_format)
    if media_type is None:
        raise HTTPException(status_code=400, detail="仅支持 JPEG、PNG、WEBP 图片")
    return data, media_type, AVATAR_CONTENT_TYPES[media_type]


def create_access_token(user: AppUser, now: datetime | None = None) -> str:
    issued_at = now or datetime.now(timezone.utc)
    payload = {
        "sub": user.id,
        "username": user.username,
        "iat": issued_at,
        "exp": issued_at + timedelta(days=TOKEN_TTL_DAYS),
        "iss": TOKEN_ISSUER,
    }
    return jwt.encode(payload, _jwt_secret(), algorithm=TOKEN_ALGORITHM)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> AppUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="请先登录", headers={"WWW-Authenticate": "Bearer"})
    try:
        claims = jwt.decode(
            credentials.credentials,
            _jwt_secret(),
            algorithms=[TOKEN_ALGORITHM],
            issuer=TOKEN_ISSUER,
        )
        user_id = str(claims.get("sub") or "")
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录", headers={"WWW-Authenticate": "Bearer"}) from exc
    user = db.get(AppUser, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录", headers={"WWW-Authenticate": "Bearer"})
    return user


@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> RegisterResponse:
    username = _normalize_username(payload.username)
    nickname = payload.nickname.strip()
    if not nickname or len(nickname) > 20:
        raise HTTPException(status_code=422, detail="昵称需为 1–20 个字符")
    existing = db.scalar(select(AppUser).where(AppUser.username == username))
    if existing is not None:
        raise HTTPException(status_code=409, detail="该账号已注册")
    user = AppUser(
        id=str(uuid.uuid4()),
        username=username,
        password_hash=bcrypt.hashpw(payload.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8"),
        nickname=nickname,
    )
    db.add(user)
    db.commit()
    return RegisterResponse(user_id=user.id, username=user.username)


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> LoginResponse:
    username = _normalize_username(payload.username)
    user = db.scalar(select(AppUser).where(AppUser.username == username))
    valid = user is not None and bcrypt.checkpw(payload.password.encode("utf-8"), user.password_hash.encode("utf-8"))
    if not valid:
        raise HTTPException(status_code=401, detail="账号或密码错误", headers={"WWW-Authenticate": "Bearer"})
    return LoginResponse(access_token=create_access_token(user), user=_user_out(user))


@router.post("/change-password", response_model=ChangePasswordResponse)
def change_password(
    payload: ChangePasswordRequest,
    user: AppUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ChangePasswordResponse:
    try:
        current_matches = bcrypt.checkpw(payload.current_password.encode("utf-8"), user.password_hash.encode("utf-8"))
    except ValueError:
        current_matches = False
    if not current_matches:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="当前密码错误")
    if payload.current_password == payload.new_password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="新密码不能与当前密码相同")
    user.password_hash = bcrypt.hashpw(payload.new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    db.commit()
    return ChangePasswordResponse()


@me_router.get("", response_model=UserOut)
def get_profile(user: AppUser = Depends(get_current_user)) -> UserOut:
    return _user_out(user)


@me_router.patch("/profile", response_model=UserOut)
def update_profile(
    payload: ProfileUpdateRequest,
    user: AppUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserOut:
    user.nickname = payload.nickname
    db.commit()
    db.refresh(user)
    return _user_out(user)


@me_router.post("/avatar", response_model=UserOut)
async def update_avatar(
    file: UploadFile = File(...),
    user: AppUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserOut:
    data, media_type, suffix = await _read_avatar(file)
    object_name = _avatar_object_name(user.id, str(uuid.uuid4()), suffix)
    previous_object = user.avatar_object_name
    try:
        bucket = storage.Client().bucket(get_bucket_name())
        bucket.blob(object_name).upload_from_string(data, content_type=media_type, if_generation_match=0)
        user.avatar_object_name = object_name
        user.avatar_url = _avatar_media_url()
        db.commit()
    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail="头像上传失败") from exc

    # Cleanup is intentionally best-effort and only touches objects owned by
    # this user's managed prefix.  Legacy/external URLs are never deleted.
    if previous_object and previous_object.startswith(f"user_avatars/{user.id}/"):
        try:
            bucket.blob(previous_object).delete()
        except Exception:
            pass
    db.refresh(user)
    return _user_out(user)


@me_router.get("/avatar/media")
def get_avatar_media(user: AppUser = Depends(get_current_user)) -> Response:
    if not user.avatar_object_name:
        raise HTTPException(status_code=404, detail="尚未设置头像")
    try:
        blob = storage.Client().bucket(get_bucket_name()).blob(user.avatar_object_name)
        if not blob.exists():
            raise HTTPException(status_code=404, detail="头像不存在")
        content = blob.download_as_bytes(timeout=120, retry=DOWNLOAD_RETRY)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail="头像暂时无法读取") from exc
    media_type = {".jpg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(
        PurePosixPath(user.avatar_object_name).suffix.lower(), "application/octet-stream"
    )
    return Response(content=content, media_type=media_type, headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})


@me_router.get("/privacy", response_model=PrivacyOut)
def get_privacy(user: AppUser = Depends(get_current_user), db: Session = Depends(get_db)) -> PrivacyOut:
    setting = db.get(UserPrivacySetting, user.id)
    return _privacy_out(setting)


@me_router.put("/privacy/ai-model-improvement", response_model=PrivacyOut)
def set_ai_model_improvement(
    payload: AiModelImprovementUpdate,
    user: AppUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PrivacyOut:
    now = utcnow()
    setting = db.get(UserPrivacySetting, user.id)
    if setting is None:
        setting = UserPrivacySetting(user_id=user.id)
        db.add(setting)
    setting.ai_model_improvement_enabled = payload.enabled
    setting.ai_model_improvement_updated_at = now
    setting.ai_model_improvement_consent_version = payload.consent_version
    db.add(UserPrivacyAudit(
        id=str(uuid.uuid4()),
        user_id=user.id,
        ai_model_improvement_enabled=payload.enabled,
        consent_version=payload.consent_version,
        source=payload.source,
        created_at=now,
    ))
    db.commit()
    db.refresh(setting)
    return _privacy_out(setting)
