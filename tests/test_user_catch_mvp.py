from __future__ import annotations

import asyncio
import io
import json

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from PIL import Image
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from starlette.datastructures import Headers, UploadFile

from app.auth_api import LoginRequest, RegisterRequest, get_current_user, login, register
from app.catches_api import (
    CatchCreate,
    catch_capabilities,
    catch_statistics,
    create_catch,
    get_catch,
    get_catch_by_client_record_id,
    list_catches,
    upload_catch_image,
)
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
            length_cm=32,
            weight_kg=3.6,
            location="上海市青浦区",
            story="清晨湖边的一次收获",
            client_record_id="guest_record_01",
        ),
        user,
        db,
    )
    assert first.saved is True
    assert first.catch.image_url.endswith(f"/{first.catch_id}/media")
    assert (first.catch.length_cm, first.catch.weight_kg, first.catch.location) == (32, 3.6, "上海市青浦区")
    assert first.catch.story == "清晨湖边的一次收获"
    assert first.catch.client_record_id == "guest_record_01"
    assert get_catch(user=user, db=db, catch_id=first.catch_id).location == "上海市青浦区"
    assert get_catch_by_client_record_id(user=user, db=db, client_record_id="guest_record_01").id == first.catch_id
    assert catch_capabilities(user=user).idempotency_keys is True
    persisted = db.get(FishCatch, first.catch_id)
    assert persisted.detector_result_json == '{"detector_version":"DET_FISH_v0.1"}'
    assert (persisted.length_cm, persisted.weight_kg, persisted.location, persisted.story) == (
        32, 3.6, "上海市青浦区", "清晨湖边的一次收获"
    )

    # A response-lost retry with the stable client record ID returns the same
    # catch without needing to re-upload or creating a duplicate.
    retry = create_catch(
        CatchCreate(
            image_upload_id="not-needed-after-first-success",
            species_id="grass_carp",
            species_name="草鱼",
            confidence=0.92,
            model_version="MODEL_M1_v0.5",
            length_cm=32,
            weight_kg=3.6,
            location="上海市青浦区",
            story="清晨湖边的一次收获",
            client_record_id="guest_record_01",
        ), user, db
    )
    assert retry.catch_id == first.catch_id
    assert len(list_catches(user=user, db=db)) == 1
    with pytest.raises(HTTPException) as mismatch:
        create_catch(
            CatchCreate(
                image_upload_id="not-needed-after-first-success",
                species_id="grass_carp", species_name="草鱼", confidence=0.92,
                model_version="MODEL_M1_v0.5", length_cm=33, weight_kg=3.6,
                location="上海市青浦区", story="清晨湖边的一次收获",
                client_record_id="guest_record_01",
            ), user, db
        )
    assert mismatch.value.status_code == 409

    second_upload = asyncio.run(upload_catch_image(_image_upload(), user))
    create_catch(
        CatchCreate(
            image_url=second_upload.image_url,
            species_id="grass_carp",
            species_name="草鱼",
            confidence=0.81,
            model_version="MODEL_M1_v0.5",
            classifier_result={
                "length_cm": 21.5,
                "weight_kg": 0.8,
                "location": "上海市青浦区",
                "story": "旧版客户端兼容值",
            },
        ),
        user,
        db,
    )
    rows = list_catches(user=user, db=db)
    assert len(rows) == 2
    assert next(row for row in rows if row.id == first.catch_id).length_cm == 32
    legacy_client_row = next(row for row in rows if row.id != first.catch_id)
    assert (legacy_client_row.length_cm, legacy_client_row.weight_kg, legacy_client_row.location,
            legacy_client_row.story) == (21.5, 0.8, "上海市青浦区", "旧版客户端兼容值")
    statistics = catch_statistics(user=user, db=db)
    assert statistics.total_catches == 2
    assert statistics.species_count == 1
    assert statistics.top_species[0].species == "草鱼"
    assert statistics.top_species[0].count == 2
    assert statistics.recent_species == "草鱼"
    db.close()


@pytest.mark.parametrize("field,value", [
    ("length_cm", 0), ("length_cm", -1), ("length_cm", float("nan")),
    ("length_cm", float("inf")), ("length_cm", 1000.01),
    ("weight_kg", 0), ("weight_kg", -1), ("weight_kg", float("nan")),
    ("weight_kg", float("inf")), ("weight_kg", 1000.01),
])
def test_catch_metadata_rejects_invalid_measurements(field, value):
    with pytest.raises(ValidationError):
        CatchCreate(species_id="grass_carp", species_name="草鱼", confidence=0.9,
                    model_version="MODEL_M1_v0.5", **{field: value})


def test_catch_metadata_accepts_null_unicode_and_documented_upper_bounds():
    payload = CatchCreate(
        species_id="grass_carp", species_name="草鱼", confidence=0.9,
        model_version="MODEL_M1_v0.5", length_cm=1000, weight_kg=1000,
        location="上海市青浦区·淀山湖 🎣",
    )
    assert payload.length_cm == 1000
    assert payload.weight_kg == 1000
    assert payload.location == "上海市青浦区·淀山湖 🎣"
    empty = CatchCreate(species_id="grass_carp", species_name="草鱼", confidence=0.9,
                        model_version="MODEL_M1_v0.5", length_cm=None, weight_kg=None, location=None)
    assert empty.length_cm is empty.weight_kg is empty.location is None


def test_catch_request_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        CatchCreate(species_id="grass_carp", species_name="草鱼", confidence=0.9,
                    model_version="MODEL_M1_v0.5", unrecognized_metadata="must not disappear")


def test_legacy_classifier_json_read_is_record_scoped_and_never_overwrites_formal_values(tmp_path):
    db = _session(tmp_path)
    user = AppUser(id="legacy-user", username="legacy", password_hash="unused", nickname="渔友")
    db.add(user)
    db.flush()
    common = dict(user_id=user.id, image_url="/media", image_object_name="private/object.jpg",
                  species_id="grass_carp", species_name="草鱼", confidence=0.9,
                  model_version="MODEL_M1_v0.5")
    legacy = FishCatch(
        id="legacy-01", **common,
        classifier_result_json=json.dumps({"length_cm": 32, "weight_kg": 3.6, "location": "上海市青浦区", "story": "旧故事"}),
        metadata_version=0,
    )
    edited = FishCatch(
        id="legacy-02", **common,
        classifier_result_json=json.dumps({"length_cm": 40, "weight_kg": 5.0, "location": "旧地点", "story": "旧故事"}),
        length_cm=42, weight_kg=None, location=None, story=None, metadata_version=1,
    )
    db.add_all([legacy, edited])
    db.commit()
    rows = {row.id: row for row in list_catches(user=user, db=db)}
    assert (rows["legacy-01"].length_cm, rows["legacy-01"].weight_kg,
            rows["legacy-01"].location, rows["legacy-01"].story) == (32, 3.6, "上海市青浦区", "旧故事")
    assert (rows["legacy-02"].length_cm, rows["legacy-02"].weight_kg,
            rows["legacy-02"].location, rows["legacy-02"].story) == (42, None, None, None)
    db.close()


def test_additive_user_catch_migration_is_idempotent(tmp_path, monkeypatch):
    import app.db as db_module

    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE fish_catches (id VARCHAR(36) PRIMARY KEY, user_id VARCHAR(36), "
            "image_url TEXT, species_id VARCHAR(128), species_name VARCHAR(128), confidence FLOAT, "
            "model_version VARCHAR(128), classifier_result_json TEXT)"
        )
        connection.exec_driver_sql(
            "INSERT INTO fish_catches (id,user_id,image_url,species_id,species_name,confidence,model_version,classifier_result_json) "
            "VALUES ('legacy-row','user-1','legacy-url','grass_carp','草鱼',0.9,'MODEL_M1_v0.5','{\"length_cm\":32}')"
        )
    monkeypatch.setattr(db_module, "engine", engine)
    db_module._ensure_user_catch_columns()
    db_module._ensure_user_catch_columns()
    columns = {column["name"] for column in inspect(engine).get_columns("fish_catches")}
    assert {"length_cm", "weight_kg", "location", "story", "metadata_version", "client_record_id"} <= columns
    indexes = {index["name"] for index in inspect(engine).get_indexes("fish_catches")}
    assert "uq_fish_catches_user_client_record" in indexes
    with engine.connect() as connection:
        row = connection.exec_driver_sql(
            "SELECT image_url, classifier_result_json, length_cm, weight_kg, location, story, metadata_version, client_record_id "
            "FROM fish_catches WHERE id='legacy-row'"
        ).one()
    assert row == ("legacy-url", '{"length_cm":32}', None, None, None, None, 0, None)


def test_read_only_audit_marks_only_same_record_non_conflicting_values_auto_recoverable():
    from scripts.audit_catch_metadata import audit_rows

    rows = [
        {
            "id": "same-record-safe", "length_cm": None, "weight_kg": None, "location": None, "story": None,
            "classifier_result_json": '{"length_cm":32,"weight_kg":3.6,"location":"上海市青浦区"}',
            "metadata_version": 0,
        },
        {
            "id": "same-record-conflict", "length_cm": 42, "weight_kg": None, "location": None, "story": None,
            "classifier_result_json": '{"length_cm":32,"weight_kg":3.6}', "metadata_version": 0,
        },
        {
            "id": "explicit-empty", "length_cm": None, "weight_kg": None, "location": None, "story": None,
            "classifier_result_json": '{"length_cm":32}', "metadata_version": 1,
        },
    ]
    details, summary = audit_rows(rows)
    safe_length = next(item for item in details if item["recordId"] == "same-record-safe" and item["field"] == "length_cm")
    conflict_length = next(item for item in details if item["recordId"] == "same-record-conflict" and item["field"] == "length_cm")
    explicit_empty = next(item for item in details if item["recordId"] == "explicit-empty" and item["field"] == "length_cm")
    assert safe_length["suggested_value"] == "32.0" and safe_length["automatic_recovery"] == "yes"
    assert conflict_length["status"] == "CONFLICT_NO_OVERWRITE" and conflict_length["automatic_recovery"] == "no"
    assert explicit_empty["status"] == "EXPLICITLY_ABSENT" and explicit_empty["automatic_recovery"] == "no"
    assert summary["total_catch_records"] == 3
    assert summary["field_conflicts"] == 1
