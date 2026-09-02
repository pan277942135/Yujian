from __future__ import annotations

import bcrypt
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.auth import create_access_token, decode_access_token
from app.db import Base
from app.models import FishCatch, User
from app.secure import install_access_guard
from app.user_api import (
    CatchCreateRequest,
    LoginRequest,
    RegisterRequest,
    catch_statistics,
    create_catch,
    list_catches,
    login,
    register,
)


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def test_register_hashes_password_and_duplicate_is_rejected(db, monkeypatch):
    monkeypatch.setenv("YUJIAN_JWT_SECRET", "test-jwt-secret")
    created = register(
        RegisterRequest(username="fisher001", password="123456", nickname="老王"),
        db,
    )
    assert created["username"] == "fisher001"
    user = db.scalar(select(User).where(User.username == "fisher001"))
    assert user is not None
    assert user.password_hash != "123456"
    assert bcrypt.checkpw(b"123456", user.password_hash.encode("utf-8"))

    with pytest.raises(Exception) as duplicate:
        register(
            RegisterRequest(username="fisher001", password="abcdef", nickname="另一个人"),
            db,
        )
    assert getattr(duplicate.value, "status_code", None) == 409


def test_login_returns_jwt_and_catches_are_owner_scoped(db, monkeypatch):
    monkeypatch.setenv("YUJIAN_JWT_SECRET", "test-jwt-secret")
    register(RegisterRequest(username="fisher001", password="123456", nickname="老王"), db)
    user = db.scalar(select(User).where(User.username == "fisher001"))
    assert user is not None

    logged_in = login(LoginRequest(username="fisher001", password="123456"), db)
    assert logged_in["token_type"] == "bearer"
    claims = decode_access_token(logged_in["access_token"])
    assert claims is not None
    assert claims["sub"] == user.id

    create_catch(
        CatchCreateRequest(
            image_url="/api/v1/catches/media/user/catch.webp",
            species_id="grass_carp",
            species_name="草鱼",
            confidence=0.92,
            model_version="MODEL_M1_v0.2",
        ),
        user,
        db,
    )
    create_catch(
        CatchCreateRequest(
            species_id="crucian_carp",
            species_name="鲫鱼",
            confidence=0.81,
            model_version="MODEL_M1_v0.2",
        ),
        user,
        db,
    )
    rows = list_catches(user, db, 50, 0)
    assert len(rows) == 2
    stats = catch_statistics(user, db)
    assert stats["total_catches"] == 2
    assert stats["species_count"] == 2
    assert {item["species_id"] for item in stats["top_species"]} == {"grass_carp", "crucian_carp"}
    assert stats["recent_species"] in {"草鱼", "鲫鱼"}


def test_user_bearer_guard_does_not_open_admin_paths(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setenv("CONSOLE_ACCESS_KEY", "console-secret")
    monkeypatch.setenv("YUJIAN_JWT_SECRET", "test-jwt-secret")
    app = FastAPI()
    install_access_guard(app)

    @app.get("/api/v1/catches")
    def catches():
        return {"ok": True}

    @app.get("/api/admin/fish/species")
    def admin():
        return {"ok": True}

    token = create_access_token(type("UserObject", (), {"id": "u1", "username": "fisher001"})())
    client = TestClient(app)
    assert client.get("/api/v1/catches").status_code == 401
    assert client.get("/api/v1/catches", headers={"Authorization": f"Bearer {token}"}).status_code == 200
    assert client.get("/api/admin/fish/species", headers={"Authorization": f"Bearer {token}"}).status_code == 401
