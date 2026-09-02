from __future__ import annotations

import os
import time
from typing import Any

import jwt
from jwt import PyJWTError


JWT_ALGORITHM = "HS256"
DEFAULT_JWT_EXPIRE_SECONDS = 30 * 24 * 60 * 60


def jwt_secret() -> str:
    """Return the signing secret without introducing a second required deployment secret.

    Production can set YUJIAN_JWT_SECRET explicitly. The existing console access key is
    a stable fallback so an existing Cloud Run deployment can enable user auth without
    changing the Admin/CMS credential flow.
    """
    for name in ("YUJIAN_JWT_SECRET", "CONSOLE_ACCESS_KEY"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    raise RuntimeError("YUJIAN_JWT_SECRET or CONSOLE_ACCESS_KEY is not configured")


def create_access_token(user: Any, *, expires_seconds: int | None = None) -> str:
    now = int(time.time())
    lifetime = int(
        expires_seconds
        if expires_seconds is not None
        else os.getenv("YUJIAN_JWT_EXPIRE_SECONDS", DEFAULT_JWT_EXPIRE_SECONDS)
    )
    payload = {
        "sub": str(user.id),
        "username": str(user.username),
        "iat": now,
        "exp": now + max(60, lifetime),
        "token_type": "bearer",
    }
    return jwt.encode(payload, jwt_secret(), algorithm=JWT_ALGORITHM)


def decode_access_token(token: str | None) -> dict[str, Any] | None:
    if not token or not token.strip():
        return None
    try:
        payload = jwt.decode(token.strip(), jwt_secret(), algorithms=[JWT_ALGORITHM])
    except (PyJWTError, RuntimeError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or not str(payload.get("sub", "")).strip():
        return None
    return payload
