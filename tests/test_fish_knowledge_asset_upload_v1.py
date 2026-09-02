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


def test_short_cms_upload_binds_cover_and_cards_to_fixed_gcs_objects(monkeypatch, tmp_path):
    db = _session(tmp_path)
    bucket = FakeBucket()
    client = FakeStorageClient(bucket)
    monkeypatch.setattr("app.fish_knowledge.admin.storage.Client", lambda: client)
    monkeypatch.setattr("app.fish_knowledge.admin.get_bucket_name", lambda: "test-bucket")
    try:
        _create_species(db)

        for asset_type, object_name in (
            ("COVER", "fish-assets/grass_carp/cover/cover.webp"),
            ("HERO", "fish-assets/grass_carp/cards/hero.webp"),
            ("SKILL", "fish-assets/grass_carp/cards/skill.webp"),
        ):
            result = _call(db, asset_type, _png_bytes())
            expected_url = f"https://storage.googleapis.com/test-bucket/{object_name}"
            assert result["success"] is True
            assert result["url"] == expected_url
            assert result["asset_type"] == asset_type
            assert result["species_id"] == "grass_carp"
            assert result["width"] == 32
            assert result["height"] == 20
            assert result["storage"]["object_name"] == object_name
            assert result["storage"]["content_type"] == "image/webp"
            blob = bucket.blobs[object_name]
            assert blob.data is not None
            assert blob.content_type == "image/webp"
            assert blob.metadata == {
                "width": "32",
                "height": "20",
                "original_content_type": "image/png",
            }
            with Image.open(io.BytesIO(blob.data)) as stored:
                assert stored.format == "WEBP"
                assert stored.size == (32, 20)

        cover = db.scalar(select(FishSpeciesCover).where(FishSpeciesCover.species_id == "grass_carp"))
        hero = db.scalar(
            select(FishCard).where(FishCard.species_id == "grass_carp", FishCard.card_type == "HERO")
        )
        skill = db.scalar(
            select(FishCard).where(FishCard.species_id == "grass_carp", FishCard.card_type == "SKILL")
        )
        assert cover is not None and cover.image_url.endswith("/cover/cover.webp")
        assert hero is not None and hero.image_url.endswith("/cards/hero.webp")
        assert skill is not None and skill.image_url.endswith("/cards/skill.webp")
        assert cover.status == "DRAFT"
        assert hero.status == "DRAFT"
        assert skill.status == "DRAFT"
    finally:
        db.close()


def test_short_cms_upload_replaces_existing_slot_without_creating_duplicate_card(monkeypatch, tmp_path):
    db = _session(tmp_path)
    bucket = FakeBucket()
    client = FakeStorageClient(bucket)
    monkeypatch.setattr("app.fish_knowledge.admin.storage.Client", lambda: client)
    monkeypatch.setattr("app.fish_knowledge.admin.get_bucket_name", lambda: "test-bucket")
    try:
        _create_species(db)
        first = _call(db, "HERO", _png_bytes((10, 20, 30)))
        card_id = first["id"]
        second = _call(db, "HERO", _png_bytes((200, 210, 220)))
        assert second["id"] == card_id
        assert second["url"] == first["url"]
        assert second["storage"]["status"] == "UPDATED"
        assert bucket.blobs["fish-assets/grass_carp/cards/hero.webp"].uploads == 2
        assert db.query(FishCard).filter(FishCard.species_id == "grass_carp").count() == 1
    finally:
        db.close()


def test_short_cms_upload_reports_binding_failure_after_storage(monkeypatch, tmp_path):
    db = _session(tmp_path)
    bucket = FakeBucket()
    client = FakeStorageClient(bucket)
    monkeypatch.setattr("app.fish_knowledge.admin.storage.Client", lambda: client)
    monkeypatch.setattr("app.fish_knowledge.admin.get_bucket_name", lambda: "test-bucket")
    try:
        _create_species(db)
        monkeypatch.setattr(
            "app.fish_knowledge.admin._commit",
            lambda _db: (_ for _ in ()).throw(RuntimeError("db down")),
        )
        response = _call(db, "COVER", _png_bytes())
        assert response.status_code == 503
        payload = json.loads(response.body)
        assert payload["error"] == "binding_error"
        assert payload["message"] == "图片已上传，但绑定保存失败"
        assert payload["url"].endswith("/fish-assets/grass_carp/cover/cover.webp")
        assert payload["storage"]["object_name"] == "fish-assets/grass_carp/cover/cover.webp"
    finally:
        db.close()


def test_short_cms_upload_rejects_empty_non_image_and_oversized_files(monkeypatch, tmp_path):
    db = _session(tmp_path)
    bucket = FakeBucket()
    client = FakeStorageClient(bucket)
    monkeypatch.setattr("app.fish_knowledge.admin.storage.Client", lambda: client)
    monkeypatch.setattr("app.fish_knowledge.admin.get_bucket_name", lambda: "test-bucket")
    try:
        _create_species(db)
        for data, reason in (
            (b"", "empty_file"),
            (b"not an image", "unsupported_format"),
            (b"x" * (KNOWLEDGE_ASSET_MAX_BYTES + 1), "file_too_large"),
        ):
            response = _call(db, "COVER", data)
            assert response.status_code == 400
            payload = json.loads(response.body)
            assert payload["success"] is False
            assert payload["error"] == "invalid_file"
            assert payload["reason"] == reason
        assert bucket.blobs == {}
    finally:
        db.close()


def test_short_cms_upload_route_contract_is_registered():
    route = app.openapi()["paths"]["/api/admin/fish/assets/upload"]["post"]
    assert route["responses"]["200"]["description"] == "Successful Response"
    schema_ref = route["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"]
    schema = app.openapi()["components"]["schemas"][schema_ref.rsplit("/", 1)[-1]]
    assert set(schema["required"]) == {"file", "species_id", "asset_type"}
