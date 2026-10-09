from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi import HTTPException

from app import models  # noqa: F401
from app.db import Base
from app.platform import models as platform_models  # noqa: F401
from app.fish_knowledge.api import get_fish_species_full_detail, get_knowledge_media, list_fish_species
from app.fish_knowledge.import_batch import FishKnowledgeAssetVersion
from app.fish_knowledge.species import FishSpecies
from app.models import SpeciesCatalog


class MemoryPublicBlob:
    def __init__(self, name):
        self.name = name

    def exists(self, _client=None):
        return self.name.endswith("/hero/v1.webp")

    def download_as_bytes(self, **_kwargs):
        return b"active-webp"


class MemoryPublicBucket:
    def blob(self, name):
        return MemoryPublicBlob(name)


class MemoryPublicStorage:
    def bucket(self, _name):
        return MemoryPublicBucket()


def test_public_contract_preserves_legacy_fields_and_never_exposes_draft(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'public-v13.db'}")
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
        summary="修长的小型鱼",
        status="ACTIVE",
    ))
    db.add_all([
        FishKnowledgeAssetVersion(
            species_id="sharpbelly",
            asset_type="HERO",
            asset_role="HERO",
            version=1,
            object_name="fish-assets/fish-knowledge/sharpbelly/hero/v1.webp",
            image_url="/api/v1/fish/knowledge-media/sharpbelly/hero/v1.webp",
            status="ACTIVE",
            sha256="a" * 64,
            metadata_json="{}",
        ),
        FishKnowledgeAssetVersion(
            species_id="sharpbelly",
            asset_type="HERO",
            asset_role="HERO",
            version=2,
            object_name="fish-assets/fish-knowledge/sharpbelly/hero/v2.webp",
            image_url="/api/v1/fish/knowledge-media/sharpbelly/hero/v2.webp",
            status="DRAFT",
            sha256="b" * 64,
            metadata_json="{}",
        ),
    ])
    db.commit()
    monkeypatch.setattr("app.fish_knowledge.api.storage.Client", lambda: MemoryPublicStorage())
    monkeypatch.setattr("app.fish_knowledge.api.get_bucket_name", lambda: "test-bucket")

    try:
        response = get_fish_species_full_detail("sharpbelly", db)
        assert response.species.cover_image is None
        assert response.cards == []
        assert response.cover_hero_image is None
        # An ACTIVE image without its exact ACTIVE content binding is an
        # explicit conflict, not permission to expose an unbound old card.
        assert "HERO" not in response.knowledge_assets
        assert any(item["role"] == "HERO" for item in response.publication_conflicts)
        assert all(not value["image_url"].endswith("/hero/v2.webp") for value in response.knowledge_assets.values())

        active = get_knowledge_media("sharpbelly", "hero", "v1.webp", db)
        assert active.body == b"active-webp"
        try:
            get_knowledge_media("sharpbelly", "hero", "v2.webp", db)
        except HTTPException as error:
            assert error.status_code == 404
        else:
            raise AssertionError("public media API must hide DRAFT versions")
    finally:
        db.close()


def test_cover_hero_public_list_and_detail_share_exact_active_version(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'cover-hero-public.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    db.add(SpeciesCatalog(
        species_key="grass_carp",
        catalog_order=1,
        common_name_zh="草鱼",
        status="active",
        is_other=False,
    ))
    db.add(FishSpecies(
        id="grass_carp",
        name_cn="草鱼",
        category="淡水鱼",
        summary="草鱼简介",
        status="ACTIVE",
    ))
    db.add(FishKnowledgeAssetVersion(
        species_id="grass_carp",
        asset_type="COVER",
        asset_role="COVER_HERO",
        version=7,
        object_name="fish-assets/fish-knowledge/grass_carp/cover_hero/v7.webp",
        image_url="/api/v1/fish/knowledge-media/grass_carp/cover_hero/v7.webp",
        status="ACTIVE",
        sha256="c" * 64,
        metadata_json="{}",
    ))
    db.commit()

    try:
        listing = list_fish_species(db)
        item = next(row for row in listing if row.id == "grass_carp")
        detail = get_fish_species_full_detail("grass_carp", db)
        assert item.cover_hero_status == "ACTIVE"
        assert item.cover_hero_version_id is not None
        assert item.cover_hero_image == detail.cover_hero_image
        assert item.cover_hero_version_id == detail.cover_hero_version_id
        assert detail.cover_hero_status == "ACTIVE"
        assert detail.cover_hero_image == "/api/v1/fish/knowledge-media/grass_carp/cover_hero/v7.webp"
    finally:
        db.close()
