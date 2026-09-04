from __future__ import annotations

import json

from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from app import models  # noqa: F401
from app.db import Base
from app.fish_knowledge import FishCard, FishFishing, FishProfile, FishSimilarity, FishSpecies, FishSpeciesCover
from app.fish_knowledge.content import EXPECTED_INITIAL_SPECIES, load_content_seed
from app.fish_knowledge.content_seed import seed_fish_knowledge_content
from app.models import SpeciesCatalog


def _session(tmp_path):
    from sqlalchemy import create_engine

    engine = create_engine(f"sqlite:///{tmp_path / 'content-init.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)()


def test_v2_seed_initializes_ten_species_and_fifty_draft_card_slots(tmp_path):
    db = _session(tmp_path)
    try:
        seed = load_content_seed()
        result = seed_fish_knowledge_content(db)

        assert {item["id"] for item in seed["species"]} == EXPECTED_INITIAL_SPECIES
        assert result["records_loaded"] == 10
        assert result["species_created"] == 10
        assert result["covers_created"] == 10
        assert result["cards_created"] == 50
        assert db.scalar(select(func.count()).select_from(FishSpecies)) == 10
        assert db.scalar(select(func.count()).select_from(FishSpeciesCover)) == 10
        assert db.scalar(select(func.count()).select_from(FishCard)) == 50
        assert db.scalar(select(func.count()).select_from(FishProfile)) == 10
        assert db.scalar(select(func.count()).select_from(FishFishing)) == 10
        assert db.scalar(select(func.count()).select_from(FishSimilarity)) == 20
        assert db.scalar(select(func.count()).select_from(SpeciesCatalog)) == 20

        for species_id in EXPECTED_INITIAL_SPECIES:
            cards = db.scalars(
                select(FishCard).where(FishCard.species_id == species_id).order_by(FishCard.sort_order)
            ).all()
            assert [card.card_type for card in cards] == ["HERO", "IDENTIFICATION", "ECO", "GEAR", "SKILL"]
            assert all(card.status == "DRAFT" and card.image_url == "" for card in cards)
            hero = json.loads(cards[0].description)
            identification = json.loads(cards[1].description)
            assert hero["tag"]
            assert len(identification["features"]) == 3
            assert len(identification["similar"]) == 2
    finally:
        db.close()


def test_v2_seed_is_idempotent_and_preserves_operator_card_content(tmp_path):
    db = _session(tmp_path)
    try:
        seed_fish_knowledge_content(db)
        hero = db.scalar(
            select(FishCard).where(FishCard.species_id == "sharpbelly", FishCard.card_type == "HERO")
        )
        assert hero is not None
        hero.title = "运营修订的白条英雄卡"
        hero.description = json.dumps({"type": "HERO", "tag": "运营自定义"}, ensure_ascii=False)
        db.commit()

        second = seed_fish_knowledge_content(db)

        assert second["species_created"] == 0
        assert second["covers_created"] == 0
        assert second["cards_created"] == 0
        assert second["cards_initialized"] == 0
        assert second["similarity_created"] == 0
        updated = db.scalar(
            select(FishCard).where(FishCard.species_id == "sharpbelly", FishCard.card_type == "HERO")
        )
        assert updated.title == "运营修订的白条英雄卡"
        assert json.loads(updated.description)["tag"] == "运营自定义"
    finally:
        db.close()

