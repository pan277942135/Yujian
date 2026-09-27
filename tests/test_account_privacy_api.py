from __future__ import annotations

import asyncio
import io

import pytest
from PIL import Image
from fastapi import HTTPException
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.auth_api import (
    AI_CONSENT_VERSION,
    AiModelImprovementUpdate,
    ChangePasswordRequest,
    LoginRequest,
    ProfileUpdateRequest,
    RegisterRequest,
    change_password,
    get_avatar_media,
    get_privacy,
    get_profile,
    login,
    register,
    set_ai_model_improvement,
    update_avatar,
    update_profile,
)
from app.db import Base
from app.models import AppUser, UserPrivacyAudit
from app.secure import is_app_api_path
# Register the existing platform tables referenced by FishBsideJob before the
# test database is created, matching the consumer catch test's import graph.
from app.platform import models as _platform_models  # noqa: F401


class FakeBlob:
    def __init__(self, name: str):
        self.name = name
        self.data: bytes | None = None
        self.content_type: str | None = None
        self.deleted = False

    def exists(self, _client=None):
        return self.data is not None and not self.deleted

    def upload_from_string(self, data, content_type=None, **_kwargs):
        self.data = bytes(data)
        self.content_type = content_type
        self.deleted = False

    def download_as_bytes(self, **_kwargs):
        return self.data or b""

    def delete(self):
        self.deleted = True


class FakeBucket:
    def __init__(self):
        self.blobs: dict[str, FakeBlob] = {}

    def blob(self, name: str):
        return self.blobs.setdefault(name, FakeBlob(name))


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'account_privacy.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)()


class MemoryUpload:
    """Async upload double; avoids Starlette's threadpool in the isolated unit runtime."""

    def __init__(self, data: bytes):
        self.data = data

    async def read(self, _size: int = -1) -> bytes:
        return self.data


def _avatar_upload() -> MemoryUpload:
    image = Image.new("RGB", (16, 12), (30, 100, 120))
    data = io.BytesIO()
    image.save(data, format="PNG")
    return MemoryUpload(data.getvalue())


def _user(db):
    created = register(RegisterRequest(username="fisher001", password="123456", nickname="老王"), db)
    return db.get(AppUser, created.user_id)


def test_profile_avatar_and_durable_media_contract(tmp_path, monkeypatch):
    db = _session(tmp_path)
    user = _user(db)
    bucket = FakeBucket()

    class Client:
        def bucket(self, name):
            assert name == "test-bucket"
            return bucket

    monkeypatch.setattr("app.auth_api.storage.Client", Client)
    monkeypatch.setattr("app.auth_api.get_bucket_name", lambda: "test-bucket")

    assert get_profile(user).nickname == "老王"
    updated = update_profile(ProfileUpdateRequest(nickname="  木木  "), user, db)
    assert updated.nickname == "木木"
    assert db.get(AppUser, user.id).nickname == "木木"

    uploaded = asyncio.run(update_avatar(_avatar_upload(), user, db))
    assert uploaded.avatar_url == "/api/v1/me/avatar/media"
    persisted = db.get(AppUser, user.id)
    assert persisted.avatar_object_name and persisted.avatar_object_name.startswith(f"user_avatars/{user.id}/")
    media = get_avatar_media(persisted)
    assert media.media_type == "image/png"
    assert bytes(media.body).startswith(b"\x89PNG")
    db.close()


def test_change_password_replaces_existing_bcrypt_hash(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_JWT_SECRET", "test-secret")
    db = _session(tmp_path)
    user = _user(db)

    with pytest.raises(HTTPException) as wrong_current:
        change_password(ChangePasswordRequest(current_password="wrong", new_password="654321"), user, db)
    assert wrong_current.value.status_code == 403

    change_password(ChangePasswordRequest(current_password="123456", new_password="654321"), user, db)
    with pytest.raises(HTTPException) as old_login:
        login(LoginRequest(username="fisher001", password="123456"), db)
    assert old_login.value.status_code == 401
    assert login(LoginRequest(username="fisher001", password="654321"), db).user.id == user.id
    db.close()


def test_ai_consent_is_server_persistent_and_audited(tmp_path):
    db = _session(tmp_path)
    user = _user(db)
    assert get_privacy(user, db).ai_model_improvement.enabled is False

    enabled = set_ai_model_improvement(
        AiModelImprovementUpdate(enabled=True, consent_version=AI_CONSENT_VERSION, source="settings"), user, db
    )
    assert enabled.ai_model_improvement.enabled is True
    assert get_privacy(user, db).ai_model_improvement.consent_version == AI_CONSENT_VERSION

    disabled = set_ai_model_improvement(
        AiModelImprovementUpdate(enabled=False, consent_version=AI_CONSENT_VERSION, source="settings"), user, db
    )
    assert disabled.ai_model_improvement.enabled is False
    audits = db.scalars(select(UserPrivacyAudit).where(UserPrivacyAudit.user_id == user.id)).all()
    assert [audit.ai_model_improvement_enabled for audit in audits] == [True, False]
    assert all(audit.consent_version == AI_CONSENT_VERSION for audit in audits)
    db.close()


def test_profile_and_privacy_paths_bypass_console_cookie_guard():
    """App Bearer routes cannot be shadowed by the Model Factory console guard."""
    assert is_app_api_path("/api/v1/me")
    assert is_app_api_path("/api/v1/me/profile")
    assert is_app_api_path("/api/v1/me/privacy")
    assert not is_app_api_path("/api/private")
