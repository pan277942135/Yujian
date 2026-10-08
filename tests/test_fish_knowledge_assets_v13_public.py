from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi import HTTPException

from app import models  # noqa: F401
from app.db import Base
from app.platform import models as platform_models  # noqa: F401
from app.fish_knowledge.api import get_fish_species_full_detail, get_knowledge_media
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
        assert response.knowledge_assets["HERO"]["image_url"].endswith("/hero/v1.webp")
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
