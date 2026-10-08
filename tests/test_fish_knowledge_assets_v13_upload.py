import asyncio
import io

import pytest
from fastapi import HTTPException
from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app import models  # noqa: F401
from app.db import Base
from app.platform import models as platform_models  # noqa: F401
from app.fish_knowledge.cards import FishCard
from app.fish_knowledge.import_batch import (
    CreateLocalBatchPayload,
    ExecuteBatchPayload,
    FishAssetImportBatch,
    FishKnowledgeAssetVersion,
    activate_asset_version_v13,
    create_local_batch,
    execute_batch,
    upload_batch_file,
    upload_single_asset_v13,
)
from app.fish_knowledge.species import FishSpecies
from app.models import SpeciesCatalog


class MemoryUpload:
    def __init__(self, data: bytes, filename: str = "upload.png"):
        self._data = data
        self.filename = filename
        self.content_type = "image/png"

    async def read(self, size=-1):
        if size < 0:
            data, self._data = self._data, b""
            return data
        data, self._data = self._data[:size], self._data[size:]
        return data

    async def close(self):
        return None


class MemoryBlob:
    def __init__(self, name: str):
        self.name = name
        self.data = None
        self.size = 0
        self.generation = None
        self.metadata = None

    def exists(self, _client=None):
        return self.data is not None

    def download_as_bytes(self, **_kwargs):
        return self.data or b""

    def upload_from_string(self, data, *, content_type, **_kwargs):
        self.data = bytes(data)
        self.size = len(self.data)
        self.content_type = content_type
        self.generation = 17

    def reload(self, _client=None):
        return None


class MemoryBucket:
    def __init__(self):
        self.blobs = {}

    def blob(self, name):
        return self.blobs.setdefault(name, MemoryBlob(name))


class MemoryStorageClient:
    def __init__(self, bucket):
        self._bucket = bucket

    def bucket(self, name):
        assert name == "test-bucket"
        return self._bucket

    def list_blobs(self, bucket, prefix=""):
        return [blob for name, blob in bucket.blobs.items() if name.startswith(prefix) and blob.data is not None]


def _png(color, *, alpha=False):
    output = io.BytesIO()
    mode = "RGBA" if alpha else "RGB"
    fill = color if alpha else color[:3]
    Image.new(mode, (1254, 1254), fill).save(output, format="PNG")
    return output.getvalue()


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'v13-upload.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    db.add(SpeciesCatalog(
        species_key="sharpbelly",
        catalog_order=1,
        common_name_zh="白条",
        status="active",
        is_other=False,
    ))
    db.add(FishSpecies(
        id="sharpbelly",
        name_cn="白条",
        category="淡水鱼",
        summary="",
        status="DRAFT",
    ))
    db.commit()
    return db


def test_single_upload_uses_batch_core_and_keeps_roles_independent(monkeypatch, tmp_path):
    db = _session(tmp_path)
    bucket = MemoryBucket()
    client = MemoryStorageClient(bucket)
    monkeypatch.setattr("app.fish_knowledge.import_batch.storage.Client", lambda: client)
    monkeypatch.setattr("app.fish_knowledge.import_batch.get_bucket_name", lambda: "test-bucket")
    try:
        hero = asyncio.run(upload_single_asset_v13(
            species_id="sharpbelly",
            asset_role="HERO",
            file=MemoryUpload(_png((20, 120, 220)), "hero.png"),
            allow_warnings=False,
            db=db,
        ))
        cover = asyncio.run(upload_single_asset_v13(
            species_id="sharpbelly",
            asset_role="COVER_LIST",
            file=MemoryUpload(_png((220, 120, 20)), "cover.png"),
            allow_warnings=False,
            db=db,
        ))

        assert hero["validation_status"] == "IMPORTED"
        assert cover["validation_status"] == "IMPORTED"
        hero_version = db.get(FishKnowledgeAssetVersion, hero["version"]["id"])
        cover_version = db.get(FishKnowledgeAssetVersion, cover["version"]["id"])
        assert hero_version.status == cover_version.status == "DRAFT"
        assert hero_version.version == cover_version.version == 1
        assert hero_version.asset_role == "HERO"
        assert cover_version.asset_role == "COVER_LIST"
        assert hero_version.object_name.endswith("/hero/v1.webp")
        assert cover_version.object_name.endswith("/cover_list/v1.webp")
        assert db.scalar(select(FishCard).where(FishCard.species_id == "sharpbelly", FishCard.card_type == "HERO")).status == "DRAFT"
        assert len([name for name in bucket.blobs if "/hero/" in name or "/cover_list/" in name]) == 2
        assert not any(version.status == "ACTIVE" for version in db.scalars(select(FishKnowledgeAssetVersion)).all())
        published = activate_asset_version_v13(hero_version.id, db)
        assert published["status"] == "ACTIVE"
        assert db.get(FishKnowledgeAssetVersion, hero_version.id).status == "ACTIVE"
        assert db.get(FishKnowledgeAssetVersion, cover_version.id).status == "DRAFT"
    finally:
        db.close()


def test_transparent_single_upload_rejects_missing_alpha_without_creating_version(monkeypatch, tmp_path):
    db = _session(tmp_path)
    bucket = MemoryBucket()
    client = MemoryStorageClient(bucket)
    monkeypatch.setattr("app.fish_knowledge.import_batch.storage.Client", lambda: client)
    monkeypatch.setattr("app.fish_knowledge.import_batch.get_bucket_name", lambda: "test-bucket")
    try:
        result = asyncio.run(upload_single_asset_v13(
            species_id="sharpbelly",
            asset_role="TRANSPARENT_MAIN",
            file=MemoryUpload(_png((20, 120, 220)), "fish.png"),
            allow_warnings=False,
            db=db,
        ))
        assert result["validation_status"] == "INVALID"
        assert any(x["code"] == "ALPHA_REQUIRED" for x in result["item"]["validation_errors"])
        assert db.scalars(select(FishKnowledgeAssetVersion)).all() == []
    finally:
        db.close()


def test_warning_execute_requires_explicit_acknowledgement(monkeypatch, tmp_path):
    db = _session(tmp_path)
    batch_id = "FK_WARN_ACK_001"
    db.add(FishAssetImportBatch(
        batch_id=batch_id,
        source_gcs_uri=f"gs://test-bucket/fish-assets/imports/{batch_id}/",
        status="READY",
        created_by="qa-operator",
        warning_files=1,
    ))
    db.commit()

    with pytest.raises(HTTPException) as blocked:
        execute_batch(batch_id, ExecuteBatchPayload(allow_warnings=False), db)
    assert blocked.value.status_code == 409
    assert db.query(FishAssetImportBatch).filter_by(batch_id=batch_id).one().warnings_acknowledged is False

    monkeypatch.setattr("app.fish_knowledge.import_batch._run_import", lambda _db, batch: setattr(batch, "status", "COMPLETED") or {})
    execute_batch(batch_id, ExecuteBatchPayload(allow_warnings=True), db)
    batch = db.query(FishAssetImportBatch).filter_by(batch_id=batch_id).one()
    assert batch.warnings_acknowledged is True
    assert batch.warnings_acknowledged_by == "qa-operator"
    assert batch.warnings_acknowledged_at is not None
    db.close()


def test_batch_upload_path_safety_and_interrupted_upload_resume(monkeypatch, tmp_path):
    db = _session(tmp_path)
    bucket = MemoryBucket()
    client = MemoryStorageClient(bucket)
    monkeypatch.setattr("app.fish_knowledge.import_batch.storage.Client", lambda: client)
    monkeypatch.setattr("app.fish_knowledge.import_batch.get_bucket_name", lambda: "test-bucket")
    batch_id = "FK_BATCH_RESUME_001"
    create_local_batch(CreateLocalBatchPayload(batch_id=batch_id), db)
    path = "sharpbelly/01_hero.png"
    data = _png((40, 100, 180))

    first = asyncio.run(upload_batch_file(batch_id, path, MemoryUpload(data), db))
    resumed = asyncio.run(upload_batch_file(batch_id, path, MemoryUpload(data), db))
    assert first["idempotent"] is False
    assert resumed["idempotent"] is True

    batch = db.query(FishAssetImportBatch).filter_by(batch_id=batch_id).one()
    batch.status = "READY"
    db.commit()
    resumed_after_scan = asyncio.run(upload_batch_file(batch_id, path, MemoryUpload(data), db))
    assert resumed_after_scan["idempotent"] is True

    with pytest.raises(HTTPException) as path_error:
        asyncio.run(upload_batch_file(batch_id, "../escape.png", MemoryUpload(data), db))
    assert path_error.value.status_code == 400
    with pytest.raises(HTTPException) as zip_error:
        asyncio.run(upload_batch_file(batch_id, "assets.zip", MemoryUpload(data, "assets.zip"), db))
    assert zip_error.value.status_code == 400
    with pytest.raises(HTTPException) as conflict:
        asyncio.run(upload_batch_file(batch_id, path, MemoryUpload(_png((180, 50, 40))), db))
    assert conflict.value.status_code == 409
    db.close()
