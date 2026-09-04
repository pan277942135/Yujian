from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import sessionmaker

from app import models  # noqa: F401
from app.db import Base, get_db
from app.fish_knowledge import FishCard, FishSpeciesCover
from app.fish_knowledge.admin import (
    CardCreate,
    CoverCreate,
    create_species_card,
    create_species_cover,
    create_admin_species,
    create_gallery_item,
    create_video,
    species_completion,
    CardPatch,
    GalleryCreate,
    VideoCreate,
    get_species_cover,
    list_species_cards,
)
from app.fish_knowledge.api import get_fish_species_full_detail, router as fish_router
from app.fish_knowledge.cards import CARD_TYPE_ORDER
from app.fish_knowledge.seed import seed_initial_fish_knowledge


def _session(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'fish-card-v11.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)()


def _species_payload(species_id="grass_carp", name_cn="草鱼", status="ACTIVE"):
    from app.fish_knowledge.admin import SpeciesCreate

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


def _public_get(app: FastAPI, path: str) -> tuple[int, object]:
    async def request():
        sent = []
        delivered = False

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": b"", "more_body": False}
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        await app(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "GET",
                "scheme": "http",
                "path": path,
                "raw_path": path.encode("utf-8"),
                "query_string": b"",
                "root_path": "",
                "headers": [],
                "client": ("test", 50000),
                "server": ("testserver", 80),
            },
            receive,
            send,
        )
        start = next(item for item in sent if item["type"] == "http.response.start")
        body = b"".join(item.get("body", b"") for item in sent if item["type"] == "http.response.body")
        return start["status"], json.loads(body.decode("utf-8"))

    return asyncio.run(request())


def _public_app(session):
    app = FastAPI()
    app.include_router(fish_router)

    def override_db():
        yield session

    app.dependency_overrides[get_db] = override_db
    return app


def _create_five_cards(db, *, status="ACTIVE", with_images=True):
    rows = []
    for order, card_type in enumerate(CARD_TYPE_ORDER):
        rows.append(
            create_species_card(
                "grass_carp",
                CardCreate(
                    card_type=card_type,
                    title=f"草鱼{card_type}",
                    image_url=f"https://cdn.example/grass-{card_type.lower()}.png" if with_images else "",
                    sort_order=order,
                    status=status,
                ),
                db,
            )
        )
    return rows


def test_cover_crud_and_five_card_slots(tmp_path):
    db = _session(tmp_path)
    try:
        create_admin_species(_species_payload(), db)
        cover = create_species_cover(
            "grass_carp",
            CoverCreate(
                image_url="https://cdn.example/grass-cover.png",
                title="草鱼图鉴卡",
                status="ACTIVE",
            ),
            db,
        )
        assert cover["style"] == "ANIME_CARD"
        assert get_species_cover("grass_carp", db)["image_url"].endswith("grass-cover.png")

        cards = _create_five_cards(db)
        assert [card["card_type"] for card in cards] == list(CARD_TYPE_ORDER)
        assert [card["sort_order"] for card in list_species_cards("grass_carp", db)] == [0, 1, 2, 3, 4]
    finally:
        db.close()


def test_duplicate_active_card_type_is_rejected(tmp_path):
    db = _session(tmp_path)
    try:
        create_admin_species(_species_payload(), db)
        create_species_card(
            "grass_carp",
            CardCreate(card_type="HERO", image_url="https://cdn.example/hero.png", status="ACTIVE"),
            db,
        )
        with pytest.raises(HTTPException) as conflict:
            create_species_card(
                "grass_carp",
                CardCreate(card_type="HERO", image_url="https://cdn.example/hero-2.png", status="ACTIVE"),
                db,
            )
        assert conflict.value.status_code == 409

        # The original v1.1 names remain accepted and are normalized to the
        # product's ECO/GEAR/SKILL card vocabulary.
        draft = create_species_card(
            "grass_carp",
            CardCreate(type="ECOLOGY", title="生态草稿", status="DRAFT"),
            db,
        )
        assert draft["card_type"] == "ECO"
    finally:
        db.close()


def test_detail_aggregate_contains_cover_cards_and_filters_draft(tmp_path):
    db = _session(tmp_path)
    try:
        create_admin_species(_species_payload(), db)
        create_species_cover(
            "grass_carp",
            CoverCreate(image_url="https://cdn.example/grass-cover.png", status="ACTIVE"),
            db,
        )
        _create_five_cards(db)
        create_species_card(
            "grass_carp",
            CardCreate(card_type="HERO", title="未发布草稿", status="DRAFT"),
            db,
        )

        detail = get_fish_species_full_detail("grass_carp", db)
        assert detail.cover["image_url"].endswith("grass-cover.png")
        assert len(detail.cards) == 5
        assert {card.card_type for card in detail.cards} == set(CARD_TYPE_ORDER)
        assert all(card.status == "ACTIVE" for card in detail.cards)
        assert detail.dynamic == {}
        assert detail.knowledge["bait"] == []

        app = _public_app(db)
        status, body = _public_get(app, "/api/v1/fish/species/grass_carp/detail")
        assert status == 200
        assert set(("species", "cover", "cards", "gallery", "profile", "fishing", "videos", "similarity", "dynamic")) <= set(body)
    finally:
        db.close()


def test_structured_card_content_round_trips_through_admin_and_public_detail(tmp_path):
    db = _session(tmp_path)
    try:
        create_admin_species(_species_payload(), db)
        create_species_cover(
            "grass_carp",
            CoverCreate(image_url="https://cdn.example/grass-cover.png", status="ACTIVE"),
            db,
        )
        create_species_card(
            "grass_carp",
            CardCreate(
                card_type="HERO",
                title="草鱼英雄卡",
                image_url="https://cdn.example/grass-hero.png",
                content={"type": "HERO", "tag": "中上层快鱼", "rarity": 2, "power": 4, "challenge": 3},
                status="ACTIVE",
            ),
            db,
        )
        create_species_card(
            "grass_carp",
            CardCreate(
                card_type="ECO",
                title="草鱼生态卡",
                image_url="https://cdn.example/grass-eco.png",
                content={
                    "type": "ECO",
                    "habitat": ["江河", "水库"],
                    "water_layer": "中下层",
                    "season": "夏季",
                    "behavior": "沿岸觅食",
                    "diet": "植物性食物",
                },
                status="ACTIVE",
            ),
            db,
        )

        admin_cards = list_species_cards("grass_carp", db)
        assert admin_cards[0]["content"]["tag"] == "中上层快鱼"
        detail = get_fish_species_full_detail("grass_carp", db)
        assert detail.cards[0].content["rarity"] == 2
        assert detail.knowledge["ecology"]["water_layer"] == "中下层"
        assert detail.knowledge["gear"]["bait"] == []
    finally:
        db.close()


def test_draft_species_is_not_public_and_baitiao_alias_resolves(tmp_path):
    db = _session(tmp_path)
    try:
        seed_initial_fish_knowledge(db)
        app = _public_app(db)

        status, body = _public_get(app, "/api/v1/fish/species/baitiao/detail")
        assert status == 200
        assert body["species"]["id"] == "sharpbelly"
        assert body["cover"]["status"] == "DRAFT" if body["cover"] else True
        assert body["cards"] == []

        status, _ = _public_get(app, "/api/v1/fish/species/not_a_species/detail")
        assert status == 404
    finally:
        db.close()


def test_completion_counts_content_without_publishing_draft_cards(tmp_path):
    db = _session(tmp_path)
    try:
        create_admin_species(_species_payload(), db)
        create_species_cover(
            "grass_carp",
            CoverCreate(image_url="https://cdn.example/grass-cover.png", status="DRAFT"),
            db,
        )
        for card_type in CARD_TYPE_ORDER[:2]:
            create_species_card(
                "grass_carp",
                CardCreate(
                    card_type=card_type,
                    image_url=f"https://cdn.example/{card_type.lower()}.png",
                    status="DRAFT",
                ),
                db,
            )
        for order in range(5):
            create_gallery_item(
                "grass_carp",
                GalleryCreate(type="standard", url=f"https://cdn.example/gallery-{order}.jpg", order=order),
                db,
            )
        result = species_completion("grass_carp", db)
        assert result["species_complete"] is True
        assert result["cover"] is True
        assert result["cards"] == {
            "completed": 2,
            "total": 5,
            "HERO": True,
            "IDENTIFICATION": True,
            "ECO": False,
            "GEAR": False,
            "SKILL": False,
        }
        assert result["gallery"] == {"completed": 5, "total": 5}
        assert result["video"] is False
        assert result["knowledge"] is False
    finally:
        db.close()


def test_card_and_cover_schema_and_seed_are_idempotent(tmp_path):
    db = _session(tmp_path)
    try:
        tables = set(inspect(db.bind).get_table_names())
        assert {"fish_cards", "fish_species_cover"} <= tables

        first = seed_initial_fish_knowledge(db)
        cover = db.scalar(select(FishSpeciesCover).where(FishSpeciesCover.species_id == "sharpbelly"))
        cards = db.scalars(
            select(FishCard).where(FishCard.species_id == "sharpbelly").order_by(FishCard.sort_order)
        ).all()
        assert first["cover_created"] == 1
        assert first["cards_created"] == 5
        assert cover.image_url == ""
        assert cover.style == "ANIME_CARD"
        assert cover.status == "DRAFT"
        assert [card.card_type for card in cards] == list(CARD_TYPE_ORDER)
        assert all(card.status == "DRAFT" and card.image_url == "" for card in cards)

        second = seed_initial_fish_knowledge(db)
        assert second["cover_created"] == 0
        assert second["cards_created"] == 0
        assert db.scalar(select(FishSpeciesCover.id).where(FishSpeciesCover.species_id == "sharpbelly")) is not None
        assert db.scalar(select(FishCard.id).where(FishCard.species_id == "sharpbelly")) is not None
    finally:
        db.close()
