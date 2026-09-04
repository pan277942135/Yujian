"""Password + JWT authentication for the YuJian consumer App.

This module is intentionally independent from the password-protected Model
Factory console.  Console cookies never grant access to a user's fish archive.
"""

from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import AppUser


router = APIRouter(prefix="/api/v1/auth", tags=["app-auth"])
bearer_scheme = HTTPBearer(auto_error=False)
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{3,32}$")
TOKEN_ALGORITHM = "HS256"
TOKEN_ISSUER = "yujian-app"
TOKEN_TTL_DAYS = 30


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    # bcrypt intentionally caps password input at 72 UTF-8 bytes.
    password: str = Field(min_length=6, max_length=72)
    nickname: str = Field(min_length=1, max_length=32)


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
    if not nickname:
        raise HTTPException(status_code=422, detail="昵称不能为空")
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
