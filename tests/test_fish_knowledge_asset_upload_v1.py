from __future__ import annotations

import asyncio
import io
import json

from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from starlette.datastructures import Headers
from starlette.datastructures import UploadFile

from app import models  # noqa: F401
from app.db import Base
from app.entry import app
from app.fish_knowledge import FishCard, FishSpecies, FishSpeciesCover
from app.fish_knowledge.admin import SpeciesCreate, create_admin_species, upload_cms_fish_asset
from app.fish_knowledge.gallery import KNOWLEDGE_ASSET_MAX_BYTES


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'fish-assets-v1.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)()


def _png_bytes(color=(50, 120, 80), size=(32, 20)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()


def _upload(data: bytes, filename: str = "asset.png") -> UploadFile:
    return UploadFile(
        file=io.BytesIO(data),
        filename=filename,
        headers=Headers({"content-type": "image/png"}),
    )


class FakeBlob:
    def __init__(self):
        self.data: bytes | None = None
        self.content_type: str | None = None
        self.metadata: dict[str, str] | None = None
        self.uploads = 0

    def exists(self, _client=None):
        return self.data is not None

    def upload_from_string(self, data, *, content_type, **_kwargs):
        self.data = bytes(data)
        self.content_type = content_type
        self.uploads += 1

    def download_as_bytes(self, **_kwargs):
        return self.data or b""


class FakeBucket:
    def __init__(self):
        self.blobs: dict[str, FakeBlob] = {}

    def blob(self, name):
        return self.blobs.setdefault(name, FakeBlob())


class FakeStorageClient:
    def __init__(self, bucket):
        self._bucket = bucket

    def bucket(self, name):
        assert name == "test-bucket"
        return self._bucket


def _create_species(db):
    return create_admin_species(
        SpeciesCreate(
            id="grass_carp",
            name_cn="草鱼",
            category="淡水鱼",
            summary="测试草鱼",
            status="DRAFT",
        ),
        db,
    )


def _call(db, asset_type: str, data: bytes):
    return asyncio.run(upload_cms_fish_asset("grass_carp", asset_type, _upload(data), db))


def test_legacy_upload_rejects_all_managed_roles_without_gcs_writes(monkeypatch, tmp_path):
    db = _session(tmp_path)
    bucket = FakeBucket()
    monkeypatch.setattr("app.fish_knowledge.admin.storage.Client", lambda: FakeStorageClient(bucket))
    monkeypatch.setattr("app.fish_knowledge.admin.get_bucket_name", lambda: "test-bucket")
    try:
        _create_species(db)
        for role in ("COVER", "HERO", "IDENTIFICATION", "ECO", "GEAR", "SKILL"):
            response = _call(db, role, _png_bytes())
            assert response.status_code == 409
            payload = json.loads(response.body)
            assert payload["success"] is False
            assert payload["error"] == "versioned_upload_required"
            assert "/api/v1/admin/fish/assets/single-upload" in payload["message"]
        assert bucket.blobs == {}
        assert db.query(FishCard).filter(FishCard.species_id == "grass_carp").count() == 0
        assert db.query(FishSpeciesCover).filter(FishSpeciesCover.species_id == "grass_carp").count() == 0
    finally:
        db.close()

def test_legacy_upload_cannot_create_unversioned_media_url(monkeypatch, tmp_path):
    db = _session(tmp_path)
    bucket = FakeBucket()
    monkeypatch.setattr("app.fish_knowledge.admin.storage.Client", lambda: FakeStorageClient(bucket))
    try:
        _create_species(db)
        for role in ("COVER", "HERO"):
            response = _call(db, role, _png_bytes())
            assert response.status_code == 409
            assert "url" not in json.loads(response.body)
        assert bucket.blobs == {}
    finally:
        db.close()

def test_legacy_upload_does_not_mutate_existing_card(monkeypatch, tmp_path):
    db = _session(tmp_path)
    bucket = FakeBucket()
    monkeypatch.setattr("app.fish_knowledge.admin.storage.Client", lambda: FakeStorageClient(bucket))
    try:
        _create_species(db)
        card = FishCard(
            species_id="grass_carp", card_type="HERO", title="Existing",
            image_url="/api/v1/fish/knowledge-media/grass_carp/hero/v1.webp",
            description='{"type":"HERO","tag":"preserve"}', sort_order=0, status="ACTIVE",
        )
        db.add(card)
        db.commit()
        response = _call(db, "HERO", _png_bytes((200, 210, 220)))
        assert response.status_code == 409
        db.refresh(card)
        assert card.status == "ACTIVE"
        assert card.title == "Existing"
        assert card.image_url.endswith("/hero/v1.webp")
        assert card.description == '{"type":"HERO","tag":"preserve"}'
        assert db.query(FishCard).filter(FishCard.species_id == "grass_carp").count() == 1
        assert bucket.blobs == {}
    finally:
        db.close()

def test_legacy_upload_does_not_reach_storage_or_database_commit(monkeypatch, tmp_path):
    db = _session(tmp_path)
    bucket = FakeBucket()
    monkeypatch.setattr("app.fish_knowledge.admin.storage.Client", lambda: FakeStorageClient(bucket))
    try:
        _create_species(db)
        monkeypatch.setattr(
            "app.fish_knowledge.admin._commit",
            lambda _db: (_ for _ in ()).throw(RuntimeError("legacy write attempted")),
        )
        response = _call(db, "COVER", _png_bytes())
        assert response.status_code == 409
        assert json.loads(response.body)["error"] == "versioned_upload_required"
        assert bucket.blobs == {}
    finally:
        db.close()

def test_legacy_upload_rejects_invalid_media_before_any_write(monkeypatch, tmp_path):
    db = _session(tmp_path)
    bucket = FakeBucket()
    monkeypatch.setattr("app.fish_knowledge.admin.storage.Client", lambda: FakeStorageClient(bucket))
    try:
        _create_species(db)
        for data in (b"", b"not an image", b"x" * (KNOWLEDGE_ASSET_MAX_BYTES + 1)):
            response = _call(db, "COVER", data)
            assert response.status_code == 409
            payload = json.loads(response.body)
            assert payload["success"] is False
            assert payload["error"] == "versioned_upload_required"
        assert bucket.blobs == {}
    finally:
        db.close()

def test_short_cms_upload_route_contract_is_registered():
    route = app.openapi()["paths"]["/api/admin/fish/assets/upload"]["post"]
    assert route["responses"]["200"]["description"] == "Successful Response"
    schema_ref = route["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"]
    schema = app.openapi()["components"]["schemas"][schema_ref.rsplit("/", 1)[-1]]
    assert set(schema["required"]) == {"file", "species_id", "asset_type"}
