from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from app import models  # noqa: F401
from app.db import Base
from app.fish_knowledge import FishFishing, FishGalleryImage, FishProfile, FishSimilarity, FishSpecies, FishVideo
from app.fish_knowledge.api import get_fish_species, list_fish_species
from app.fish_knowledge.seed import (
    FISH_KNOWLEDGE_SEED_VERSION,
    INITIAL_FISH_KNOWLEDGE,
    seed_initial_fish_knowledge,
)
from app.species_policy import TARGET_SPECIES_PRESETS


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'seed.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)()


def test_seed_creates_exact_model_m1_v05_twenty_species(tmp_path):
    db = _session(tmp_path)
    try:
        result = seed_initial_fish_knowledge(db)
        rows = db.scalars(select(FishSpecies).order_by(FishSpecies.id)).all()

        assert result["version"] == FISH_KNOWLEDGE_SEED_VERSION
        assert result["species_created"] == 20
        assert len(INITIAL_FISH_KNOWLEDGE) == 20
        assert len(rows) == 20
        assert {row.id for row in rows} == {item["species_key"] for item in TARGET_SPECIES_PRESETS}
        assert all(row.status == "ACTIVE" for row in rows)
        assert db.scalar(select(func.count()).select_from(FishProfile)) == 20
        assert db.scalar(select(func.count()).select_from(FishFishing)) == 20
        assert db.scalar(select(func.count()).select_from(FishSimilarity)) >= 10
        assert db.scalar(select(func.count()).select_from(FishGalleryImage)) == 0
        assert db.scalar(select(func.count()).select_from(FishVideo)) == 0
    finally:
        db.close()


def test_seed_is_idempotent_and_preserves_admin_changes(tmp_path):
    db = _session(tmp_path)
    try:
        seed_initial_fish_knowledge(db)
        grass = db.get(FishSpecies, "grass_carp")
        grass.summary = "管理员已经修订的草鱼简介"
        grass.status = "DRAFT"
        relation = db.scalar(
            select(FishSimilarity).where(
                FishSimilarity.species_id == "grass_carp",
                FishSimilarity.similar_species_id == "black_carp",
            )
        )
        db.delete(relation)
        db.commit()

        second = seed_initial_fish_knowledge(db)

        assert second["species_created"] == 0
        assert second["profile_created"] == 0
        assert second["fishing_created"] == 0
        assert second["similarity_created"] == 0
        assert db.get(FishSpecies, "grass_carp").summary == "管理员已经修订的草鱼简介"
        assert db.get(FishSpecies, "grass_carp").status == "DRAFT"
        assert db.scalar(
            select(FishSimilarity.id).where(
                FishSimilarity.species_id == "grass_carp",
                FishSimilarity.similar_species_id == "black_carp",
            )
        ) is None
        assert db.scalar(select(func.count()).select_from(FishSpecies)) == 20
    finally:
        db.close()


def test_seeded_public_api_returns_twenty_and_complete_detail_contract(tmp_path):
    db = _session(tmp_path)
    try:
        seed_initial_fish_knowledge(db)
        species = list_fish_species(db)
        detail = get_fish_species("grass_carp", db)

        assert len(species) == 20
        assert {item.id for item in species} == {item["species"]["id"] for item in INITIAL_FISH_KNOWLEDGE}
        assert detail.species.id == "grass_carp"
        assert detail.species.name_cn == "草鱼"
        assert detail.gallery.images == []
        assert detail.profile.features
        assert detail.profile.habitat
        assert detail.fishing.bait
        assert detail.fishing.method
        assert detail.videos == []
        assert detail.similarity
    finally:
        db.close()


def test_seeded_api_keeps_unknown_species_404(tmp_path):
    db = _session(tmp_path)
    try:
        seed_initial_fish_knowledge(db)
        try:
            get_fish_species("not_a_species", db)
        except HTTPException as exc:
            assert exc.status_code == 404
        else:
            raise AssertionError("unknown species must return 404")
    finally:
        db.close()
