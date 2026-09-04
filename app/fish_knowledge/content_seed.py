from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.fish_knowledge.cards import FishCard, normalize_card_type
from app.fish_knowledge.content import card_description, load_content_seed
from app.fish_knowledge.cover import FishSpeciesCover
from app.fish_knowledge.fishing import FishFishing
from app.fish_knowledge.profile import FishProfile
from app.fish_knowledge.similarity import FishSimilarity
from app.fish_knowledge.species import FishSpecies
from app.models import SpeciesCatalog
from app.species_policy import ensure_target_species


CONTENT_CARD_TYPES = ("HERO", "IDENTIFICATION", "ECO", "GEAR", "SKILL")


def _text(value: Any) -> str:
    return str(value or "").strip()


def _list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in (_text(entry) for entry in value) if item]


def _first_missing(target: Any, source: Any) -> Any:
    """Return seed content only when the operator has not entered a value."""

    if isinstance(target, list):
        return target or source
    if target is None:
        return source
    if isinstance(target, str):
        return target if target.strip() else source
    return target


def _species_values(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _text(item["id"]),
        "name_cn": _text(item["name"]),
        "alias": _list(item.get("alias")),
        "scientific_name": _text(item.get("scientific_name")) or None,
        "category": _text(item.get("category")) or "淡水鱼",
        "family": _text(item.get("family")) or None,
        "genus": _text(item.get("genus")) or None,
        "summary": _text(item.get("summary")),
        "status": _text(item.get("status")) or "DRAFT",
    }


def _seed_card_description(card: dict[str, Any]) -> str:
    content = dict(card)
    content.pop("image_url", None)
    content.pop("status", None)
    content["type"] = normalize_card_type(content.get("type", ""))
    return card_description(content)


def _existing_cards(db: Session, species_id: str) -> dict[str, FishCard]:
    return {
        normalize_card_type(row.card_type): row
        for row in db.scalars(select(FishCard).where(FishCard.species_id == species_id)).all()
        if normalize_card_type(row.card_type) in CONTENT_CARD_TYPES
    }


def _upsert_profile(db: Session, species_id: str, values: dict[str, Any]) -> tuple[FishProfile, bool, bool]:
    row = db.get(FishProfile, species_id)
    profile_values = {
        "body_shape": _text(values.get("body_shape")) or None,
        "features": _list(values.get("features")),
        "habitat": _list(values.get("habitat")),
        "food": _text(values.get("food")) or None,
        "season": _list(values.get("season")),
    }
    changed = False
    if row is None:
        row = FishProfile(species_id=species_id, **profile_values)
        db.add(row)
        return row, True, True
    for field, value in profile_values.items():
        current = getattr(row, field)
        replacement = _first_missing(current, value)
        if replacement != current:
            setattr(row, field, replacement)
            changed = True
    return row, changed, False


def _upsert_fishing(db: Session, species_id: str, values: dict[str, Any]) -> tuple[FishFishing, bool, bool]:
    row = db.get(FishFishing, species_id)
    fishing_values = {
        "water_layer": _text(values.get("water_layer")) or None,
        "season": _list(values.get("season")),
        "bait": _list(values.get("bait")),
        "method": _list(values.get("method")),
        "summary": _text(values.get("summary")),
    }
    changed = False
    if row is None:
        row = FishFishing(species_id=species_id, **fishing_values)
        db.add(row)
        return row, True, True
    for field, value in fishing_values.items():
        current = getattr(row, field)
        replacement = _first_missing(current, value)
        if replacement != current:
            setattr(row, field, replacement)
            changed = True
    return row, changed, False


def seed_fish_knowledge_content(
    db: Session,
    seed_path: str | Path | None = None,
) -> dict[str, int | str]:
    """Seed the v2 editorial package without overwriting operator content.

    The database schema intentionally remains the v1 schema. Rich card fields
    are persisted as JSON in the existing ``fish_cards.description`` column and
    are exposed as ``content`` by the API.
    """

    seed = load_content_seed(seed_path)
    ensure_target_species(db)
    changed = False
    species_created = 0
    covers_created = 0
    cards_created = 0
    profiles_created = 0
    fishing_created = 0
    similarity_created = 0
    cards_initialized = 0

    for item in seed["species"]:
        species_id = _text(item["id"])
        catalog = db.get(SpeciesCatalog, species_id)
        if catalog is None:
            raise ValueError(f"seed species is not present in stable SpeciesCatalog: {species_id}")

        species = db.get(FishSpecies, species_id)
        values = _species_values(item)
        if species is None:
            species = FishSpecies(**values)
            db.add(species)
            species_created += 1
            changed = True

        cover_values = item.get("cover") or {}
        cover = db.scalar(select(FishSpeciesCover).where(FishSpeciesCover.species_id == species_id))
        if cover is None:
            cover = FishSpeciesCover(
                species_id=species_id,
                image_url=_text(cover_values.get("image_url")),
                style=_text(cover_values.get("style")) or "ANIME_CARD",
                title=_text(cover_values.get("title")),
                status=_text(cover_values.get("status")) or "DRAFT",
            )
            db.add(cover)
            covers_created += 1
            changed = True
        elif cover.status == "DRAFT" and not _text(cover.image_url):
            for field, value in {
                "style": _text(cover_values.get("style")) or "ANIME_CARD",
                "title": _text(cover_values.get("title")),
            }.items():
                if not _text(getattr(cover, field)) and value:
                    setattr(cover, field, value)
                    changed = True

        cards = _existing_cards(db, species_id)
        for order, card_values in enumerate(item.get("cards") or []):
            card_type = normalize_card_type(card_values.get("type", ""))
            if card_type not in CONTENT_CARD_TYPES:
                raise ValueError(f"unsupported seed card type {card_type} for {species_id}")
            row = cards.get(card_type)
            title = _text(card_values.get("title")) or f"{values['name_cn']}{card_type}卡"
            description = _seed_card_description(card_values)
            if row is None:
                row = FishCard(
                    species_id=species_id,
                    card_type=card_type,
                    title=title,
                    image_url="",
                    description=description,
                    sort_order=order,
                    status="DRAFT",
                )
                db.add(row)
                cards[card_type] = row
                cards_created += 1
                cards_initialized += 1
                changed = True
            elif row.status == "DRAFT" and not _text(row.image_url):
                if not _text(row.title):
                    row.title = title
                    changed = True
                if not _text(row.description):
                    row.description = description
                    cards_initialized += 1
                    changed = True

        knowledge = item.get("knowledge") or {}
        _, profile_changed, profile_created = _upsert_profile(db, species_id, knowledge.get("profile") or {})
        _, fishing_changed, fishing_created_now = _upsert_fishing(db, species_id, knowledge.get("fishing") or {})
        profiles_created += int(profile_created)
        fishing_created += int(fishing_created_now)
        if profile_changed:
            changed = True
        if fishing_changed:
            changed = True

        for relation in item.get("similarity") or []:
            similar_id = _text(relation.get("species_id"))
            if not similar_id or db.get(SpeciesCatalog, similar_id) is None:
                raise ValueError(f"similarity target is not present in SpeciesCatalog: {similar_id}")
            exists = db.scalar(
                select(FishSimilarity.id).where(
                    FishSimilarity.species_id == species_id,
                    FishSimilarity.similar_species_id == similar_id,
                )
            )
            if exists is None:
                db.add(
                    FishSimilarity(
                        species_id=species_id,
                        similar_species_id=similar_id,
                        difference=_text(relation.get("difference")),
                    )
                )
                similarity_created += 1
                changed = True

    if changed:
        db.commit()
    return {
        "version": str(seed.get("schema", "FISH_KNOWLEDGE_CONTENT_V2")),
        "records_loaded": len(seed["species"]),
        "species_created": species_created,
        "covers_created": covers_created,
        "cards_created": cards_created,
        "cards_initialized": cards_initialized,
        "profiles_created": profiles_created,
        "fishing_created": fishing_created,
        "similarity_created": similarity_created,
    }
