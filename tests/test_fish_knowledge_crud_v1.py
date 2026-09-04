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
                display_tag="河湾目标鱼",
                rarity=2,
                power=3,
                challenge=4,
                recommendation=5,
            ),
            db,
        )
        assert updated["name_cn"] == "测试鱼修订"
        assert updated["summary"] == "修订后的简介"
        assert updated["display_tag"] == "河湾目标鱼"

        cover = compat_put_species_cover(
            "crud_test_fish",
            CoverPut(url="https://cdn.example/crud-cover.png", status="DRAFT"),
            db,
        )
        assert cover["image_url"] == "https://cdn.example/crud-cover.png"
        assert cover["status"] == "DRAFT"

        for card_type in CARD_TYPE_ORDER:
            card = compat_put_species_card(
                "crud_test_fish",
                card_type.lower(),
                CardPatch(
                    title=f"测试鱼 {card_type}",
                    image_url=f"https://cdn.example/{card_type.lower()}.png",
                    content={"type": card_type, "description": f"{card_type} 内容"},
                    status="DRAFT",
                ),
                db,
            )
            assert card["card_type"] == card_type

        cards = list_species_cards("crud_test_fish", db)
        assert [card["card_type"] for card in cards] == list(CARD_TYPE_ORDER)
        assert len({card["card_type"] for card in cards}) == 5

        deleted_card = update_species_card(
            cards[-1]["id"], CardPatch(title="待删除卡"), db
        )
        assert deleted_card["title"] == "待删除卡"
        assert delete_species_card(cards[-1]["id"], db)["deleted"] is True
        assert len(list_species_cards("crud_test_fish", db)) == 4
    finally:
        db.close()


def test_cms_soft_delete_hides_species_without_orphaning_content(tmp_path):
    db = _session(tmp_path)
    try:
        _create_species(db)
        cover = compat_put_species_cover(
            "crud_test_fish",
            CoverPut(url="https://cdn.example/cover.png"),
            db,
        )
        card = compat_put_species_card(
            "crud_test_fish",
            "hero",
            CardPatch(image_url="https://cdn.example/hero.png"),
            db,
        )

        deleted = delete_admin_species("crud_test_fish", db)
        assert deleted == {
            "deleted": True,
            "id": "crud_test_fish",
            "species_id": "crud_test_fish",
            "status": "DELETED",
        }
        row = db.get(FishSpecies, "crud_test_fish")
        assert row is not None and row.status == "DELETED"
        assert db.get(FishSpeciesCover, cover["id"]) is not None
        assert db.get(FishCard, card["id"]) is not None

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
        with pytest.raises(HTTPException) as missing:
            publish_admin_species("crud_test_fish", db)
        assert missing.value.status_code == 409
        assert missing.value.detail["success"] is False
        assert "cover" in missing.value.detail["missing"]
        assert "hero" in missing.value.detail["missing"]
        assert "knowledge" in missing.value.detail["missing"]

        compat_put_species_cover(
            "crud_test_fish",
            CoverPut(url="https://cdn.example/publish-cover.png", status="ACTIVE"),
            db,
        )
        for card_type in CARD_TYPE_ORDER:
            compat_put_species_card(
                "crud_test_fish",
                card_type,
                CardPatch(
                    image_url=f"https://cdn.example/publish-{card_type.lower()}.png",
                    status="ACTIVE",
                ),
                db,
            )
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
    finally:
        db.close()


def test_publish_promotes_complete_draft_asset_package(tmp_path):
    db = _session(tmp_path)
    try:
        _create_species(db)
        compat_put_species_cover(
            "crud_test_fish",
            CoverPut(url="/api/v1/fish/knowledge-media/crud_test_fish/cover/cover.webp", status="DRAFT"),
            db,
        )
        for card_type in CARD_TYPE_ORDER:
            compat_put_species_card(
                "crud_test_fish",
                card_type,
                CardPatch(
                    image_url=f"/api/v1/fish/knowledge-media/crud_test_fish/{card_type.lower()}/{card_type.lower()}.webp",
                    status="DRAFT",
                ),
                db,
            )
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
        assert db.get(FishSpecies, "crud_test_fish").status == "ACTIVE"
        cover = db.scalar(select(FishSpeciesCover).where(FishSpeciesCover.species_id == "crud_test_fish"))
        assert cover is not None and cover.status == "ACTIVE"
        cards = db.scalars(
            select(FishCard).where(FishCard.species_id == "crud_test_fish").order_by(FishCard.sort_order)
        ).all()
        assert [card.status for card in cards] == ["ACTIVE"] * 5
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
