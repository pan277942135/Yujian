from __future__ import annotations

import asyncio
import io

import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.datastructures import Headers, UploadFile

from app.auth_api import LoginRequest, RegisterRequest, get_current_user, login, register
from app.catches_api import CatchCreate, catch_statistics, create_catch, list_catches, upload_catch_image
from app.db import Base
from app.models import AppUser, FishCatch


class FakeBlob:
    def __init__(self, name: str):
        self.name = name
        self.data: bytes | None = None
        self.content_type: str | None = None

    def exists(self, _client=None):
        return self.data is not None

    def upload_from_string(self, data, content_type=None, **_kwargs):
        self.data = bytes(data)
        self.content_type = content_type

    def download_as_bytes(self, **_kwargs):
        return self.data or b""


class FakeBucket:
    def __init__(self):
        self.blobs: dict[str, FakeBlob] = {}

    def blob(self, name: str):
        return self.blobs.setdefault(name, FakeBlob(name))


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'mvp.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)()


def _image_upload() -> UploadFile:
    image = Image.new("RGB", (8, 6), (30, 100, 120))
    data = io.BytesIO()
    image.save(data, format="JPEG")
    return UploadFile(
        file=io.BytesIO(data.getvalue()),
        filename="catch.jpg",
        headers=Headers({"content-type": "image/jpeg"}),
    )


def test_register_duplicate_login_and_jwt_user(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_JWT_SECRET", "test-secret")
    db = _session(tmp_path)
    created = register(RegisterRequest(username="fisher001", password="123456", nickname="老王"), db)
    assert created.username == "fisher001"
    assert db.get(AppUser, created.user_id).password_hash != "123456"

    with pytest.raises(Exception) as duplicate:
        register(RegisterRequest(username="fisher001", password="123456", nickname="另一个老王"), db)
    assert getattr(duplicate.value, "status_code", None) == 409

    token = login(LoginRequest(username="fisher001", password="123456"), db).access_token
    credentials = __import__("fastapi.security", fromlist=["HTTPAuthorizationCredentials"]).HTTPAuthorizationCredentials(
        scheme="Bearer", credentials=token
    )
    assert get_current_user(credentials, db).id == created.user_id
    db.close()


def test_authenticated_catch_save_list_and_statistics(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_JWT_SECRET", "test-secret")
    db = _session(tmp_path)
    bucket = FakeBucket()

    class Client:
        def bucket(self, name):
            assert name == "test-bucket"
            return bucket

    monkeypatch.setattr("app.catches_api.storage.Client", Client)
    monkeypatch.setattr("app.catches_api.get_bucket_name", lambda: "test-bucket")
    user = AppUser(id="user-001", username="fisher001", password_hash="unused", nickname="老王")
    db.add(user)
    db.commit()

    upload = asyncio.run(upload_catch_image(_image_upload(), user))
    assert upload.image_url.endswith("/media")

    first = create_catch(
        CatchCreate(
            image_upload_id=upload.image_upload_id,
            species_id="grass_carp",
            species_name="草鱼",
            confidence=0.92,
            model_version="MODEL_M1_v0.5",
            detector_result={"detector_version": "DET_FISH_v0.1"},
            classifier_result={"prediction_species": "grass_carp"},
        ),
        user,
        db,
    )
    assert first.saved is True
    assert first.catch.image_url.endswith(f"/{first.catch_id}/media")
    persisted = db.get(FishCatch, first.catch_id)
    assert persisted.detector_result_json == '{"detector_version":"DET_FISH_v0.1"}'

    second_upload = asyncio.run(upload_catch_image(_image_upload(), user))
    create_catch(
        CatchCreate(
            image_url=second_upload.image_url,
            species_id="grass_carp",
            species_name="草鱼",
            confidence=0.81,
            model_version="MODEL_M1_v0.5",
        ),
        user,
        db,
    )
    rows = list_catches(user=user, db=db)
    assert len(rows) == 2
    statistics = catch_statistics(user=user, db=db)
    assert statistics.total_catches == 2
    assert statistics.species_count == 1
    assert statistics.top_species[0].species == "草鱼"
    assert statistics.top_species[0].count == 2
    assert statistics.recent_species == "草鱼"
    db.close()
