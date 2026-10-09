from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app import models  # noqa: F401
from app.db import Base
import app.db as db_module
from app.fish_knowledge.cards import FishCard
from app.fish_knowledge.import_batch import FishCardContentRevision, FishKnowledgeAssetVersion
from app.fish_knowledge.species import FishSpecies
from app.models import SpeciesCatalog


def test_v14_binding_migration_is_idempotent_and_preserves_ambiguous_history(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'fish-v14-migration.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(db_module, "engine", engine)
    monkeypatch.setattr(db_module, "SessionLocal", factory)
    db = factory()
    db.add(SpeciesCatalog(
        species_key="migration_fish",
        catalog_order=1,
        common_name_zh="迁移鱼",
        status="active",
        is_other=False,
    ))
    db.add(FishSpecies(
        id="migration_fish",
        name_cn="迁移鱼",
        category="淡水鱼",
        summary="迁移验证",
        status="ACTIVE",
    ))
    version = FishKnowledgeAssetVersion(
        species_id="migration_fish",
        asset_type="HERO",
        asset_role="HERO",
        version=1,
        object_name="fish-assets/fish-knowledge/migration_fish/hero/v1.webp",
        image_url="/api/v1/fish/knowledge-media/migration_fish/hero/v1.webp",
        status="ACTIVE",
        sha256="a" * 64,
        metadata_json="{}",
    )
    db.add(version)
    db.flush()
    exact = FishCard(
        species_id="migration_fish",
        card_type="HERO",
        title="已知精确匹配",
        image_url=version.image_url,
        description='{"type":"HERO","description":"原内容不变"}',
        status="ACTIVE",
    )
    ambiguous_a = FishCard(
        species_id="migration_fish",
        card_type="GEAR",
        title="历史 A",
        image_url="https://legacy.example/gear.png",
        description="原始 A",
        status="DRAFT",
    )
    ambiguous_b = FishCard(
        species_id="migration_fish",
        card_type="GEAR",
        title="历史 B",
        image_url="https://legacy.example/gear.png",
        description="原始 B",
        status="DRAFT",
    )
    db.add_all([exact, ambiguous_a, ambiguous_b])
    db.commit()
    exact_id, ambiguous_ids, version_id = exact.id, (ambiguous_a.id, ambiguous_b.id), version.id
    db.close()

    db_module._ensure_fish_knowledge_publication_v14()
    db_module._ensure_fish_knowledge_publication_v14()

    check = factory()
    try:
        exact_after = check.get(FishCard, exact_id)
        assert exact_after.asset_version_id == version_id
        assert exact_after.description == '{"type":"HERO","description":"原内容不变"}'
        assert check.scalar(select(FishCardContentRevision.id).where(
            FishCardContentRevision.card_id == exact_id,
            FishCardContentRevision.content_revision == 1,
        )) is not None
        ambiguous_after = [check.get(FishCard, card_id) for card_id in ambiguous_ids]
        assert [card.asset_version_id for card in ambiguous_after] == [None, None]
        assert [card.title for card in ambiguous_after] == ["历史 A", "历史 B"]
    finally:
        check.close()
