from __future__ import annotations

import asyncio
import io

import pytest
from fastapi import HTTPException
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.datastructures import Headers, UploadFile

from app import models  # noqa: F401
from app.db import Base
from app.fish_knowledge import FishFishing, FishGalleryImage, FishProfile, FishSimilarity, FishSpecies, FishVideo
from app.fish_knowledge.admin import (
    FishingUpsert,
    GalleryCreate,
    GalleryPatch,
    ProfileUpsert,
    SimilarityUpsert,
    SpeciesCreate,
    SpeciesPatch,
    VideoCreate,
    VideoPatch,
    create_admin_species,
    create_gallery_item,
    create_video,
    delete_gallery_item,
    delete_similarity,
    delete_video,
    update_admin_species,
    update_gallery_item,
    update_video,
    upload_gallery_image,
    upsert_fishing,
    upsert_profile,
    upsert_similarity,
)
from app.fish_knowledge.api import get_fish_species, get_gallery_media
from app.models import SpeciesCatalog


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'admin.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)()


def _species_payload(species_id="grass_carp", name_cn="草鱼", status="ACTIVE"):
    return SpeciesCreate(
        id=species_id,
        name_cn=name_cn,
        alias=[name_cn],
        scientific_name="Ctenopharyngodon idella" if species_id == "grass_carp" else None,
        category="淡水鱼",
        family="鲤科",
        genus="草鱼属" if species_id == "grass_carp" else None,
        summary=f"{name_cn}知识简介",
        status=status,
    )


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (32, 20), (50, 120, 80)).save(output, format="PNG")
    return output.getvalue()


class FakeBlob:
    def __init__(self):
        self.data: bytes | None = None
        self.content_type: str | None = None
        self.uploads = 0

    def exists(self, _client=None):
        return self.data is not None

    def download_as_bytes(self, **_kwargs):
        return self.data or b""

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


def test_admin_creates_catalog_aligned_species_and_updates_content(tmp_path):
    db = _session(tmp_path)
    try:
        created = create_admin_species(_species_payload(status="DRAFT"), db)
        assert created["id"] == "grass_carp"
        assert created["status"] == "DRAFT"
        catalog = db.get(SpeciesCatalog, "grass_carp")
        assert catalog is not None
        assert catalog.status == "candidate"

        updated = update_admin_species(
            "grass_carp",
            SpeciesPatch(alias=["鲩鱼", "草鲩", "鲩鱼"], status="ACTIVE", summary="大型淡水目标鱼"),
            db,
        )
        assert updated["alias"] == ["鲩鱼", "草鲩"]
        assert updated["status"] == "ACTIVE"
        assert catalog.status == "candidate"
    finally:
        db.close()


def test_admin_upserts_profile_and_structured_fishing_knowledge(tmp_path):
    db = _session(tmp_path)
    try:
        create_admin_species(_species_payload(), db)
        profile = upsert_profile(
            "grass_carp",
            ProfileUpsert(
                body_shape="体型修长侧扁",
                features=["背部青灰", "鳞片较大"],
                habitat=["江河", "湖泊"],
                food="草食性为主",
                season=["春", "夏", "秋"],
            ),
            db,
        )
        fishing = upsert_fishing(
            "grass_carp",
            FishingUpsert(
                water_layer="中下层",
                season=["夏", "秋"],
                bait=["玉米", "青草"],
                method=["浮钓", "底钓"],
                summary="高水温时更活跃。",
            ),
            db,
        )
        detail = get_fish_species("grass_carp", db)

        assert profile["features"] == ["背部青灰", "鳞片较大"]
        assert fishing["bait"] == ["玉米", "青草"]
        assert detail.profile.body_shape == "体型修长侧扁"
        assert detail.fishing.method == ["浮钓", "底钓"]
        assert db.get(FishProfile, "grass_carp") is not None
        assert db.get(FishFishing, "grass_carp") is not None
    finally:
        db.close()


def test_admin_manages_external_gallery_and_enforces_five_image_contract(tmp_path):
    db = _session(tmp_path)
    try:
        create_admin_species(_species_payload(), db)
        first = None
        for order in range(5):
            item = create_gallery_item(
                "grass_carp",
                GalleryCreate(
                    type="standard" if order == 0 else "catch",
                    url=f"https://cdn.example/grass-{order}.jpg",
                    title=f"图片 {order}",
                    order=order,
                ),
                db,
            )
            first = first or item
        assert db.query(FishGalleryImage).count() == 5
        with pytest.raises(HTTPException) as conflict:
            create_gallery_item(
                "grass_carp",
                GalleryCreate(type="side", url="https://cdn.example/extra.jpg", order=0),
                db,
            )
        assert conflict.value.status_code == 409

        changed = update_gallery_item(
            "grass_carp",
            first["id"],
            GalleryPatch(type="side", title="标准侧身修订"),
            db,
        )
        assert changed["type"] == "side"
        assert delete_gallery_item("grass_carp", first["id"], db)["deleted"] is True
        assert db.query(FishGalleryImage).count() == 4
    finally:
        db.close()


def test_admin_uploads_managed_gallery_image_idempotently_and_serves_it(monkeypatch, tmp_path):
    db = _session(tmp_path)
    bucket = FakeBucket()
    client = FakeStorageClient(bucket)
    monkeypatch.setattr("app.fish_knowledge.admin.storage.Client", lambda: client)
    monkeypatch.setattr("app.fish_knowledge.admin.get_bucket_name", lambda: "test-bucket")
    monkeypatch.setattr("app.fish_knowledge.api.storage.Client", lambda: client)
    monkeypatch.setattr("app.fish_knowledge.api.get_bucket_name", lambda: "test-bucket")
    try:
        create_admin_species(_species_payload(), db)
        data = _png_bytes()
        first = asyncio.run(
            upload_gallery_image(
                "grass_carp",
                type="standard",
                order=0,
                title="标准侧身",
                file=UploadFile(
                    file=io.BytesIO(data),
                    filename="grass.png",
                    headers=Headers({"content-type": "image/png"}),
                ),
                db=db,
            )
        )
        duplicate = asyncio.run(
            upload_gallery_image(
                "grass_carp",
                type="standard",
                order=0,
                title="重复重试",
                file=UploadFile(
                    file=io.BytesIO(data),
                    filename="grass.png",
                    headers=Headers({"content-type": "image/png"}),
                ),
                db=db,
            )
        )

        assert first["managed"] is True
        assert first["storage"] == "CREATED"
        assert first["url"].endswith(f"/{first['id']}/media")
        assert duplicate["duplicate"] is True
        assert db.query(FishGalleryImage).count() == 1
        assert next(iter(bucket.blobs.values())).uploads == 1
        response = get_gallery_media(first["id"], db)
        assert response.body == data
        assert response.media_type == "image/png"
    finally:
        db.close()


def test_admin_manages_videos_and_similarity(tmp_path):
    db = _session(tmp_path)
    try:
        create_admin_species(_species_payload(), db)
        create_admin_species(_species_payload("black_carp", "青鱼"), db)
        video = create_video(
            "grass_carp",
            VideoCreate(
                title="这鱼怎么钓",
                type="HOW_TO_FISH",
                cover_url="https://cdn.example/cover.jpg",
                video_url="https://cdn.example/how-to-grass.mp4",
                duration=90,
                tags=["草鱼", "玉米"],
                order=0,
            ),
            db,
        )
        changed = update_video(
            "grass_carp",
            video["id"],
            VideoPatch(duration=75, tags=["草鱼", "浮钓"]),
            db,
        )
        relation = upsert_similarity(
            "grass_carp",
            "black_carp",
            SimilarityUpsert(difference="草鱼体色偏黄绿；青鱼体色更深。"),
            db,
        )

        assert changed["duration"] == 75
        assert changed["tags"] == ["草鱼", "浮钓"]
        assert relation["similar_species_id"] == "black_carp"
        assert db.query(FishVideo).count() == 1
        assert db.query(FishSimilarity).count() == 1
        assert delete_video("grass_carp", video["id"], db)["deleted"] is True
        assert delete_similarity("grass_carp", "black_carp", db)["deleted"] is True
    finally:
        db.close()
