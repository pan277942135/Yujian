from __future__ import annotations

from sqlalchemy import create_engine, inspect

from app import models  # noqa: F401
from app.db import Base
from app.fish_knowledge import (  # noqa: F401
    FishFishing,
    FishGalleryImage,
    FishProfile,
    FishRanking,
    FishSimilarity,
    FishSpecies,
    FishVideo,
)


def test_fish_knowledge_schema_creates_all_v1_tables():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    tables = set(inspect(engine).get_table_names())

    assert {
        "fish_species",
        "fish_gallery",
        "fish_profile",
        "fish_fishing",
        "fish_video",
        "fish_similarity",
        "fish_ranking",
    } <= tables


def test_fish_species_id_reuses_stable_catalog_identity():
    foreign_keys = inspect(create_engine("sqlite:///:memory:"))
    engine = foreign_keys.bind
    Base.metadata.create_all(engine)
    species_fks = inspect(engine).get_foreign_keys("fish_species")

    assert any(
        fk["referred_table"] == "species_catalog"
        and fk["constrained_columns"] == ["id"]
        and fk["referred_columns"] == ["species_key"]
        for fk in species_fks
    )
