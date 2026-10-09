from __future__ import annotations

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app import models  # noqa: F401
from app.db import Base
from app.fish_knowledge import FishCard, FishSpecies, FishSpeciesCover
from app.fish_knowledge.admin import (
    CardCreate,
    CardPatch,
    CoverPut,
    FishingUpsert,
    ProfileUpsert,
    SpeciesCreate,
    SpeciesPatch,
    compat_put_fishing,
    compat_put_profile,
    compat_put_species_card,
    compat_put_species_cover,
    compat_update_admin_species,
    create_admin_species,
    delete_admin_species,
    delete_species_card,
    get_admin_species,
    list_admin_species,
    list_species_cards,
    publish_admin_species,
    species_completion,
    update_species_card,
    update_species_cover,
)
from app.fish_knowledge.api import get_fish_species_full_detail
from app.fish_knowledge.cards import CARD_TYPE_ORDER


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'fish-crud-v1.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)()


def _create_species(db, species_id: str = "crud_test_fish") -> dict:
    return create_admin_species(
        SpeciesCreate(
            species_id=species_id,
            name="测试鱼",
            category="淡水鱼",
            summary="测试鱼种简介",
            status="DRAFT",
        ),
        db,
    )


def test_cms_crud_creates_updates_cover_and_cards(tmp_path):
    db = _session(tmp_path)
    try:
        created = _create_species(db)
        assert created["id"] == "crud_test_fish"
        assert created["status"] == "DRAFT"

        updated = compat_update_admin_species(
            "crud_test_fish",
            SpeciesPatch(
                name="测试鱼修订",
                description="修订后的简介",
            ),
            db,
        )
        assert updated["name_cn"] == "测试鱼修订"
        assert updated["summary"] == "修订后的简介"
        with pytest.raises(HTTPException) as old_cover:
            compat_put_species_cover(
                "crud_test_fish",
                CoverPut(url="https://cdn.example/crud-cover.png", status="ACTIVE"),
                db,
            )
        assert old_cover.value.status_code == 409
        with pytest.raises(HTTPException) as old_card:
            compat_put_species_card(
                "crud_test_fish",
                "hero",
                CardPatch(image_url="https://cdn.example/hero.png", status="ACTIVE"),
                db,
            )
        assert old_card.value.status_code == 409
        with pytest.raises(HTTPException) as old_hero_content:
            compat_update_admin_species("crud_test_fish", SpeciesPatch(display_tag="旧 HERO 字段"), db)
        assert old_hero_content.value.status_code == 409
        assert list_species_cards("crud_test_fish", db) == []
    finally:
        db.close()


def test_cms_soft_delete_hides_species_without_orphaning_content(tmp_path):
    db = _session(tmp_path)
    try:
        _create_species(db)
        cover = FishSpeciesCover(
            species_id="crud_test_fish",
            image_url="https://cdn.example/cover.png",
            style="ANIME_CARD",
            title="历史封面",
            status="ACTIVE",
        )
        card = FishCard(
            species_id="crud_test_fish",
            card_type="HERO",
            title="历史卡片",
            image_url="https://cdn.example/hero.png",
            description="历史内容",
            sort_order=0,
            status="ACTIVE",
        )
        db.add_all([cover, card])
        db.commit()

        deleted = delete_admin_species("crud_test_fish", db)
        assert deleted == {
            "deleted": True,
            "id": "crud_test_fish",
            "species_id": "crud_test_fish",
            "status": "DELETED",
        }
        row = db.get(FishSpecies, "crud_test_fish")
        assert row is not None and row.status == "DELETED"
        assert db.scalar(select(FishSpeciesCover).where(FishSpeciesCover.species_id == "crud_test_fish")) is not None
        assert db.get(FishCard, card.id) is not None

        assert all(item["id"] != "crud_test_fish" for item in list_admin_species(db))
        with pytest.raises(HTTPException) as hidden:
            get_admin_species("crud_test_fish", db)
        assert hidden.value.status_code == 404
        with pytest.raises(HTTPException) as public_hidden:
            get_fish_species_full_detail("crud_test_fish", db)
        assert public_hidden.value.status_code == 404
    finally:
        db.close()


def test_completion_and_publish_validate_all_required_content(tmp_path):
    db = _session(tmp_path)
    try:
        _create_species(db)
        published_species_only = publish_admin_species("crud_test_fish", db)
        assert published_species_only["success"] is True
        assert published_species_only["published_modules"] == ["SPECIES"]
        assert published_species_only["assets_changed"] is False
        assert db.get(FishSpecies, "crud_test_fish").status == "ACTIVE"
        assert db.scalar(select(FishSpeciesCover).where(FishSpeciesCover.species_id == "crud_test_fish")) is None
        assert db.scalars(select(FishCard).where(FishCard.species_id == "crud_test_fish")).all() == []

        db.add(FishSpeciesCover(
            species_id="crud_test_fish",
            image_url="https://cdn.example/publish-cover.png",
            style="ANIME_CARD",
            title="发布前封面",
            status="ACTIVE",
        ))
        for card_type in CARD_TYPE_ORDER:
            db.add(FishCard(
                species_id="crud_test_fish",
                card_type=card_type,
                title=f"{card_type} 历史卡",
                image_url=f"https://cdn.example/publish-{card_type.lower()}.png",
                description="{}",
                sort_order=CARD_TYPE_ORDER.index(card_type),
                status="ACTIVE",
            ))
        db.commit()
        compat_put_profile(
            "crud_test_fish",
            ProfileUpsert(body_shape="体型修长", features=["鳞片明显"]),
            db,
        )
        compat_put_fishing(
            "crud_test_fish",
            FishingUpsert(water_layer="中上层", bait=["玉米"]),
            db,
        )

        state = species_completion("crud_test_fish", db)
        assert state["cover"] is True
        assert state["cards"] == {
            "completed": 5,
            "total": 5,
            "HERO": True,
            "IDENTIFICATION": True,
            "ECO": True,
            "GEAR": True,
            "SKILL": True,
        }
        assert state["knowledge"] is True
        published = publish_admin_species("crud_test_fish", db)
        assert published["success"] is True
        assert db.get(FishSpecies, "crud_test_fish").status == "ACTIVE"
        # Existing legacy active rows stay visible without being changed.
        cover = db.scalar(select(FishSpeciesCover).where(FishSpeciesCover.species_id == "crud_test_fish"))
        assert cover is not None and cover.status == "ACTIVE"
    finally:
        db.close()


def test_species_publish_preserves_legacy_asset_draft_status(tmp_path):
    db = _session(tmp_path)
    try:
        _create_species(db)
        db.add(FishSpeciesCover(
            species_id="crud_test_fish",
            image_url="/api/v1/fish/knowledge-media/crud_test_fish/cover/cover.webp",
            style="ANIME_CARD",
            title="历史草稿封面",
            status="DRAFT",
        ))
        for card_type in CARD_TYPE_ORDER:
            db.add(FishCard(
                species_id="crud_test_fish",
                card_type=card_type,
                title=f"{card_type} 历史草稿",
                image_url=f"/api/v1/fish/knowledge-media/crud_test_fish/{card_type.lower()}/{card_type.lower()}.webp",
                description="{}",
                sort_order=CARD_TYPE_ORDER.index(card_type),
                status="DRAFT",
            ))
        db.commit()
        compat_put_profile(
            "crud_test_fish",
            ProfileUpsert(body_shape="体型修长", features=["鳞片明显"]),
            db,
        )
        compat_put_fishing(
            "crud_test_fish",
            FishingUpsert(water_layer="中上层", bait=["玉米"]),
            db,
        )

        published = publish_admin_species("crud_test_fish", db)
        assert published["success"] is True
        assert published["assets_changed"] is False
        assert db.get(FishSpecies, "crud_test_fish").status == "ACTIVE"
        cover = db.scalar(select(FishSpeciesCover).where(FishSpeciesCover.species_id == "crud_test_fish"))
        assert cover is not None and cover.status == "DRAFT"
        cards = db.scalars(
            select(FishCard).where(FishCard.species_id == "crud_test_fish").order_by(FishCard.sort_order)
        ).all()
        assert [card.status for card in cards] == ["DRAFT"] * 5
    finally:
        db.close()


def test_public_detail_and_routes_keep_full_shape_and_draft_is_not_public(tmp_path):
    db = _session(tmp_path)
    try:
        _create_species(db)
        with pytest.raises(HTTPException) as hidden:
            get_fish_species_full_detail("crud_test_fish", db)
        assert hidden.value.status_code == 404

        row = db.get(FishSpecies, "crud_test_fish")
        row.status = "ACTIVE"
        db.commit()
        detail = get_fish_species_full_detail("crud_test_fish", db)
        payload = detail.model_dump()
        assert {"species", "cover", "cards", "gallery", "profile", "fishing", "videos", "similarity", "knowledge", "dynamic"}.issubset(payload)
        assert payload["cards"] == []
        assert payload["dynamic"] == {}

        from app.entry import app

        paths = app.openapi()["paths"]
        assert "/api/admin/fish/species" in paths
        assert "/api/admin/fish/species/{species_id}/cover" in paths
        assert "/api/admin/fish/species/{species_id}/cards/{card_type}" in paths
        assert "/api/v1/admin/fish/species/{species_id}/publish" in paths
    finally:
        db.close()
