from __future__ import annotations

import re
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from google.cloud import storage
from pydantic import AliasChoices, BaseModel, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from app.db import get_db
from app.factory import get_bucket_name
from app.fish_knowledge.api import build_species_detail, load_species_with_knowledge
from app.fish_knowledge.cards import (
    CARD_TYPE_ORDER,
    CARD_TYPES,
    FishCard,
    card_type_sort_order,
    normalize_card_type,
)
from app.fish_knowledge.cover import FishSpeciesCover
from app.fish_knowledge.content import card_description, card_display_description, parse_card_content
from app.fish_knowledge.fishing import FishFishing
from app.fish_knowledge.gallery import (
    GALLERY_TYPES,
    KNOWLEDGE_ASSET_MAX_BYTES,
    KNOWLEDGE_ASSET_TYPES,
    MAX_GALLERY_IMAGE_BYTES,
    MAX_GALLERY_IMAGES,
    FishGalleryImage,
    GalleryUploadError,
    inspect_knowledge_asset,
    inspect_gallery_image,
    managed_knowledge_asset_url,
    store_cms_knowledge_asset,
    store_knowledge_asset,
    store_gallery_image,
)
from app.fish_knowledge.profile import FishProfile
from app.fish_knowledge.similarity import FishSimilarity
from app.fish_knowledge.species import FishSpecies
from app.fish_knowledge.video import FishVideo
from app.models import SpeciesCatalog


router = APIRouter(prefix="/api/v1/admin/fish", tags=["fish-knowledge-admin"])
# Keep the original v1 admin paths for the CMS while also exposing the shorter
# CRUD contract requested by lightweight operators and integrations.
compat_router = APIRouter(prefix="/api/admin/fish", tags=["fish-knowledge-admin"])
SPECIES_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,127}$")


def _clean_list(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _validate_url(value: str | None, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    result = (value or "").strip()
    # Uploaded Fish Knowledge assets are served through the public read-only
    # media endpoint. Keep accepting normal HTTPS URLs for manually managed
    # content, while allowing the managed URL to round-trip through the CMS
    # save forms after an upload.
    if not result.startswith("https://") and not result.startswith("/api/v1/fish/knowledge-media/"):
        raise ValueError("URL 必须使用 https:// 或 YuJian 托管素材地址")
    if len(result) > 2048:
        raise ValueError("URL 过长")
    return result


class SpeciesCreate(BaseModel):
    model_config = {"populate_by_name": True}

    id: str = Field(
        validation_alias=AliasChoices("id", "species_id"),
        min_length=2,
        max_length=128,
    )
    name_cn: str = Field(
        validation_alias=AliasChoices("name_cn", "name"),
        min_length=1,
        max_length=128,
    )
    alias: list[str] = Field(default_factory=list)
    scientific_name: str | None = Field(default=None, max_length=256)
    category: str = Field(default="淡水鱼", min_length=1, max_length=64)
    family: str | None = Field(default=None, max_length=128)
    genus: str | None = Field(default=None, max_length=128)
    summary: str = Field(
        default="",
        validation_alias=AliasChoices("summary", "description"),
        max_length=500,
    )
    display_tag: str | None = Field(default=None, max_length=128)
    rarity: int | None = Field(default=None, ge=0, le=5)
    power: int | None = Field(default=None, ge=0, le=5)
    challenge: int | None = Field(
        default=None,
        validation_alias=AliasChoices("challenge", "target_difficulty"),
        ge=0,
        le=5,
    )
    recommendation: int | None = Field(default=None, ge=0, le=5)
    status: Literal["ACTIVE", "DRAFT"] = "DRAFT"

    @field_validator("id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        result = value.strip()
        if not SPECIES_ID_RE.fullmatch(result):
            raise ValueError("id 必须是小写字母开头的 snake_case")
        return result

    @field_validator("name_cn", "category")
    @classmethod
    def required_text(cls, value: str) -> str:
        result = value.strip()
        if not result:
            raise ValueError("字段不能为空")
        return result

    @field_validator("summary")
    @classmethod
    def summary_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("scientific_name", "family", "genus")
    @classmethod
    def optional_text(cls, value: str | None) -> str | None:
        result = (value or "").strip()
        return result or None

    @field_validator("alias")
    @classmethod
    def aliases(cls, value: list[str]) -> list[str]:
        return _clean_list(value) or []


class SpeciesPatch(BaseModel):
    model_config = {"populate_by_name": True}

    name_cn: str | None = Field(
        default=None,
        validation_alias=AliasChoices("name_cn", "name"),
        min_length=1,
        max_length=128,
    )
    alias: list[str] | None = None
    scientific_name: str | None = Field(default=None, max_length=256)
    category: str | None = Field(default=None, min_length=1, max_length=64)
    family: str | None = Field(default=None, max_length=128)
    genus: str | None = Field(default=None, max_length=128)
    summary: str | None = Field(
        default=None,
        validation_alias=AliasChoices("summary", "description"),
        max_length=500,
    )
    display_tag: str | None = Field(default=None, max_length=128)
    rarity: int | None = Field(default=None, ge=0, le=5)
    power: int | None = Field(default=None, ge=0, le=5)
    challenge: int | None = Field(
        default=None,
        validation_alias=AliasChoices("challenge", "target_difficulty"),
        ge=0,
        le=5,
    )
    recommendation: int | None = Field(default=None, ge=0, le=5)
    status: Literal["ACTIVE", "DRAFT"] | None = None

    @field_validator("name_cn", "category")
    @classmethod
    def required_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        result = value.strip()
        if not result:
            raise ValueError("字段不能为空")
        return result

    @field_validator("summary")
    @classmethod
    def summary_text(cls, value: str | None) -> str | None:
        return None if value is None else value.strip()

    @field_validator("scientific_name", "family", "genus")
    @classmethod
    def optional_text(cls, value: str | None) -> str | None:
        result = (value or "").strip()
        return result or None

    @field_validator("alias")
    @classmethod
    def aliases(cls, value: list[str] | None) -> list[str] | None:
        return _clean_list(value)


class ProfileUpsert(BaseModel):
    body_shape: str | None = Field(default=None, max_length=500)
    features: list[str] = Field(default_factory=list)
    habitat: list[str] = Field(default_factory=list)
    food: str | None = Field(default=None, max_length=500)
    season: list[str] = Field(default_factory=list)

    @field_validator("features", "habitat", "season")
    @classmethod
    def lists(cls, value: list[str]) -> list[str]:
        return _clean_list(value) or []


class FishingUpsert(BaseModel):
    water_layer: str | None = Field(default=None, max_length=128)
    season: list[str] = Field(default_factory=list)
    bait: list[str] = Field(default_factory=list)
    method: list[str] = Field(default_factory=list)
    summary: str = Field(default="", max_length=500)

    @field_validator("season", "bait", "method")
    @classmethod
    def lists(cls, value: list[str]) -> list[str]:
        return _clean_list(value) or []


class GalleryCreate(BaseModel):
    type: Literal["standard", "side", "top", "catch", "environment", "action"]
    url: str
    title: str | None = Field(default=None, max_length=256)
    order: int = Field(ge=0, lt=MAX_GALLERY_IMAGES)

    @field_validator("url")
    @classmethod
    def valid_url(cls, value: str) -> str:
        return str(_validate_url(value))


class GalleryPatch(BaseModel):
    type: Literal["standard", "side", "top", "catch", "environment", "action"] | None = None
    url: str | None = None
    title: str | None = Field(default=None, max_length=256)
    order: int | None = Field(default=None, ge=0, lt=MAX_GALLERY_IMAGES)

    @field_validator("url")
    @classmethod
    def valid_url(cls, value: str | None) -> str | None:
        return _validate_url(value, optional=True)


class VideoCreate(BaseModel):
    title: str = Field(min_length=1, max_length=256)
    type: Literal["INTRO", "HOW_TO_FISH", "REAL_CATCH", "EQUIPMENT"]
    cover_url: str | None = None
    video_url: str
    duration: int = Field(ge=0, le=86400)
    tags: list[str] = Field(default_factory=list)
    order: int = Field(default=0, ge=0, le=10000)

    @field_validator("title")
    @classmethod
    def valid_title(cls, value: str) -> str:
        result = value.strip()
        if not result:
            raise ValueError("title 不能为空")
        return result

    @field_validator("cover_url")
    @classmethod
    def valid_cover_url(cls, value: str | None) -> str | None:
        return _validate_url(value, optional=True)

    @field_validator("video_url")
    @classmethod
    def valid_video_url(cls, value: str) -> str:
        return str(_validate_url(value))

    @field_validator("tags")
    @classmethod
    def valid_tags(cls, value: list[str]) -> list[str]:
        return _clean_list(value) or []


class VideoPatch(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=256)
    type: Literal["INTRO", "HOW_TO_FISH", "REAL_CATCH", "EQUIPMENT"] | None = None
    cover_url: str | None = None
    video_url: str | None = None
    duration: int | None = Field(default=None, ge=0, le=86400)
    tags: list[str] | None = None
    order: int | None = Field(default=None, ge=0, le=10000)

    @field_validator("title")
    @classmethod
    def valid_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        result = value.strip()
        if not result:
            raise ValueError("title 不能为空")
        return result

    @field_validator("cover_url")
    @classmethod
    def valid_cover_url(cls, value: str | None) -> str | None:
        return _validate_url(value, optional=True)

    @field_validator("video_url")
    @classmethod
    def valid_video_url(cls, value: str | None) -> str | None:
        return _validate_url(value, optional=True)

    @field_validator("tags")
    @classmethod
    def valid_tags(cls, value: list[str] | None) -> list[str] | None:
        return _clean_list(value)


class SimilarityUpsert(BaseModel):
    difference: str = Field(min_length=1, max_length=1000)

    @field_validator("difference")
    @classmethod
    def valid_difference(cls, value: str) -> str:
        result = value.strip()
        if not result:
            raise ValueError("difference 不能为空")
        return result


class CoverCreate(BaseModel):
    image_url: str = Field(default="", max_length=2048)
    style: str = Field(default="ANIME_CARD", min_length=1, max_length=64)
    title: str = Field(default="", max_length=256)
    status: Literal["ACTIVE", "DRAFT"] = "DRAFT"

    @field_validator("image_url")
    @classmethod
    def valid_image_url(cls, value: str) -> str:
        result = value.strip()
        return "" if not result else str(_validate_url(result))

    @field_validator("style", "title")
    @classmethod
    def clean_text(cls, value: str) -> str:
        return value.strip()


class CoverPatch(BaseModel):
    image_url: str | None = Field(default=None, max_length=2048)
    style: str | None = Field(default=None, min_length=1, max_length=64)
    title: str | None = Field(default=None, max_length=256)
    status: Literal["ACTIVE", "DRAFT"] | None = None

    @field_validator("image_url")
    @classmethod
    def valid_image_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        result = value.strip()
        return "" if not result else str(_validate_url(result))

    @field_validator("style", "title")
    @classmethod
    def clean_text(cls, value: str | None) -> str | None:
        return None if value is None else value.strip()


class CardCreate(BaseModel):
    card_type: str = Field(
        validation_alias=AliasChoices("card_type", "type"),
        min_length=1,
        max_length=32,
    )
    title: str = Field(default="", max_length=256)
    image_url: str = Field(default="", max_length=2048)
    description: str = Field(default="", max_length=2000)
    content: dict[str, Any] | None = None
    sort_order: int | None = Field(default=None, ge=0, le=10000)
    status: Literal["ACTIVE", "DRAFT"] = "DRAFT"

    @field_validator("card_type")
    @classmethod
    def valid_card_type(cls, value: str) -> str:
        normalized = str(value).strip().upper()
        if normalized not in CARD_TYPES:
            raise ValueError(f"不支持的卡片类型：{value}")
        return normalize_card_type(normalized)

    @field_validator("image_url")
    @classmethod
    def valid_image_url(cls, value: str) -> str:
        result = value.strip()
        return "" if not result else str(_validate_url(result))

    @field_validator("title", "description")
    @classmethod
    def clean_text(cls, value: str) -> str:
        return value.strip()


class CardPatch(BaseModel):
    card_type: str | None = Field(
        default=None,
        validation_alias=AliasChoices("card_type", "type"),
        min_length=1,
        max_length=32,
    )
    title: str | None = Field(default=None, max_length=256)
    image_url: str | None = Field(default=None, max_length=2048)
    description: str | None = Field(default=None, max_length=2000)
    content: dict[str, Any] | None = None
    sort_order: int | None = Field(default=None, ge=0, le=10000)
    status: Literal["ACTIVE", "DRAFT"] | None = None

    @field_validator("card_type")
    @classmethod
    def valid_card_type(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip().upper()
        if normalized not in CARD_TYPES:
            raise ValueError(f"不支持的卡片类型：{value}")
        return normalize_card_type(normalized)

    @field_validator("image_url")
    @classmethod
    def valid_image_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        result = value.strip()
        return "" if not result else str(_validate_url(result))

    @field_validator("title", "description")
    @classmethod
    def clean_text(cls, value: str | None) -> str | None:
        return None if value is None else value.strip()


class CoverPut(BaseModel):
    """Short-form upsert payload used by the public CMS CRUD contract."""

    model_config = {"populate_by_name": True}

    url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("url", "image_url"),
        max_length=2048,
    )
    style: str | None = Field(default=None, min_length=1, max_length=64)
    title: str | None = Field(default=None, max_length=256)
    status: Literal["ACTIVE", "DRAFT"] | None = None

    @field_validator("url")
    @classmethod
    def valid_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        result = value.strip()
        return "" if not result else str(_validate_url(result))

    @field_validator("style", "title")
    @classmethod
    def clean_text(cls, value: str | None) -> str | None:
        return None if value is None else value.strip()


def _commit(db: Session) -> None:
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="fish knowledge data conflicts with an existing record") from exc


def _require_species(
    db: Session,
    species_id: str,
    *,
    include_deleted: bool = False,
) -> FishSpecies:
    row = db.get(FishSpecies, species_id)
    if row is None or (row.status == "DELETED" and not include_deleted):
        raise HTTPException(status_code=404, detail="fish species not found")
    return row


def _get_species_cover(db: Session, species_id: str) -> FishSpeciesCover | None:
    return db.scalar(select(FishSpeciesCover).where(FishSpeciesCover.species_id == species_id))


def _gallery_dict(row: FishGalleryImage) -> dict:
    return {
        "id": row.id,
        "species_id": row.species_id,
        "type": row.type,
        "url": row.url,
        "title": row.title,
        "order": row.sort_order,
        "managed": bool(row.object_name),
        "content_type": row.content_type,
        "size_bytes": row.size_bytes,
        "sha256": row.sha256,
    }


def _video_dict(row: FishVideo) -> dict:
    return {
        "id": row.id,
        "species_id": row.species_id,
        "title": row.title,
        "type": row.type,
        "cover_url": row.cover_url,
        "video_url": row.video_url,
        "duration": row.duration,
        "tags": row.tags or [],
        "order": row.sort_order,
    }


def _cover_dict(row: FishSpeciesCover) -> dict:
    return {
        "id": row.id,
        "species_id": row.species_id,
        "image_url": managed_knowledge_asset_url(row.species_id, "COVER", row.image_url),
        "style": row.style,
        "title": row.title,
        "status": row.status,
    }


def _hero_card(db: Session, species_id: str) -> FishCard | None:
    rows = db.scalars(
        select(FishCard)
        .where(FishCard.species_id == species_id)
        .order_by(FishCard.id)
    ).all()
    return next((card for card in rows if normalize_card_type(card.card_type) == "HERO"), None)


def _admin_species_dict(row: FishSpecies, *, hero: FishCard | None = None) -> dict:
    content = parse_card_content(hero.description if hero else "")
    return {
        "id": row.id,
        "species_id": row.id,
        "name_cn": row.name_cn,
        "name": row.name_cn,
        "alias": row.alias or [],
        "scientific_name": row.scientific_name,
        "category": row.category,
        "family": row.family,
        "genus": row.genus,
        "summary": row.summary,
        "description": row.summary,
        "status": row.status,
        "display_tag": content.get("tag"),
        "rarity": content.get("rarity"),
        "power": content.get("power"),
        "challenge": content.get("challenge"),
        "target_difficulty": content.get("challenge"),
        "recommendation": content.get("recommendation"),
    }


def _species_model_values(payload: SpeciesCreate) -> dict[str, Any]:
    return {
        "id": payload.id,
        "name_cn": payload.name_cn,
        "alias": payload.alias,
        "scientific_name": payload.scientific_name,
        "category": payload.category,
        "family": payload.family,
        "genus": payload.genus,
        "summary": payload.summary,
        "status": payload.status,
    }


def _sync_hero_fields(
    db: Session,
    species: FishSpecies,
    payload: SpeciesCreate | SpeciesPatch,
) -> FishCard | None:
    field_names = ("display_tag", "rarity", "power", "challenge", "recommendation")
    changes = {
        field: getattr(payload, field)
        for field in field_names
        if field in payload.model_fields_set
    }
    if not changes:
        return _hero_card(db, species.id)

    hero = _hero_card(db, species.id)
    if hero is None:
        hero = FishCard(
            species_id=species.id,
            card_type="HERO",
            title=f"{species.name_cn}英雄卡",
            image_url="",
            description=card_description({"type": "HERO"}),
            sort_order=card_type_sort_order("HERO"),
            status="DRAFT",
        )
        db.add(hero)
        db.flush()
    content = parse_card_content(hero.description)
    content["type"] = "HERO"
    for field, value in changes.items():
        if value is None:
            key = "tag" if field == "display_tag" else field
            content.pop(key, None)
        else:
            key = "tag" if field == "display_tag" else field
            content[key] = value
    hero.description = card_description(content)
    return hero


def _publication_missing(db: Session, species: FishSpecies) -> list[str]:
    missing: list[str] = []
    if not species.name_cn.strip() or not species.category.strip() or not species.summary.strip():
        missing.append("species")

    cover = _get_species_cover(db, species.id)
    if cover is None or not (cover.image_url or "").strip():
        missing.append("cover")

    available_types = {
        normalize_card_type(card.card_type)
        for card in db.scalars(select(FishCard).where(FishCard.species_id == species.id)).all()
        if (card.image_url or "").strip()
    }
    missing.extend(card_type.lower() for card_type in CARD_TYPE_ORDER if card_type not in available_types)

    knowledge_ready = bool(
        species.profile
        and species.fishing
        and (
            species.profile.body_shape
            or species.profile.features
            or species.profile.habitat
            or species.profile.food
            or species.profile.season
            or species.fishing.water_layer
            or species.fishing.bait
            or species.fishing.method
        )
    )
    if not knowledge_ready:
        missing.append("knowledge")
    return missing


def _card_dict(row: FishCard) -> dict:
    card_type = normalize_card_type(row.card_type)
    content = parse_card_content(row.description)
    return {
        "id": row.id,
        "species_id": row.species_id,
        "card_type": card_type,
        # ``type`` mirrors the public detail example while ``card_type`` keeps
        # the database field name explicit for Admin clients.
        "type": card_type,
        "title": row.title,
        "image_url": managed_knowledge_asset_url(row.species_id, card_type, row.image_url),
        "description": card_display_description(content, row.description),
        "content": content,
        "sort_order": row.sort_order,
        "status": row.status,
    }


def _ensure_active_card_type(
    db: Session,
    species_id: str,
    card_type: str,
    *,
    exclude_id: int | None = None,
) -> None:
    normalized = normalize_card_type(card_type)
    rows = db.scalars(
        select(FishCard).where(
            FishCard.species_id == species_id,
            FishCard.status == "ACTIVE",
        )
    ).all()
    if any(
        row.id != exclude_id and normalize_card_type(row.card_type) == normalized
        for row in rows
    ):
        raise HTTPException(status_code=409, detail=f"同一鱼种不能有两个 ACTIVE {normalized} 卡片")


def _ensure_gallery_slot(
    db: Session,
    species_id: str,
    order: int,
    *,
    exclude_id: int | None = None,
) -> None:
    count_statement = select(func.count()).select_from(FishGalleryImage).where(
        FishGalleryImage.species_id == species_id
    )
    if exclude_id is None and int(db.scalar(count_statement) or 0) >= MAX_GALLERY_IMAGES:
        raise HTTPException(status_code=409, detail="每个鱼种最多维护 5 张轮播图片")
    order_statement = select(FishGalleryImage.id).where(
        FishGalleryImage.species_id == species_id,
        FishGalleryImage.sort_order == order,
    )
    if exclude_id is not None:
        order_statement = order_statement.where(FishGalleryImage.id != exclude_id)
    if db.scalar(order_statement) is not None:
        raise HTTPException(status_code=409, detail=f"Gallery order {order} 已被占用")


@router.get("/species")
def list_admin_species(db: Session = Depends(get_db)) -> list[dict]:
    rows = db.scalars(
        select(FishSpecies)
        .join(SpeciesCatalog, SpeciesCatalog.species_key == FishSpecies.id)
        .where(FishSpecies.status != "DELETED")
        .options(
            selectinload(FishSpecies.gallery),
            selectinload(FishSpecies.cover),
            selectinload(FishSpecies.cards),
            selectinload(FishSpecies.profile),
            selectinload(FishSpecies.fishing),
            selectinload(FishSpecies.videos),
        )
        .order_by(SpeciesCatalog.catalog_order, FishSpecies.id)
    ).all()
    return [
        {
            "id": row.id,
            "name_cn": row.name_cn,
            "status": row.status,
            "category": row.category,
            "summary": row.summary,
            "gallery_count": len(row.gallery),
            "cover_ready": bool(row.cover and row.cover.status == "ACTIVE" and row.cover.image_url.strip()),
            "cards_ready": len({
                normalize_card_type(card.card_type)
                for card in row.cards
                if card.status == "ACTIVE" and card.image_url.strip()
            }),
            "cards_total": len(CARD_TYPE_ORDER),
            "video_count": len(row.videos),
            "profile_ready": row.profile is not None,
            "fishing_ready": row.fishing is not None,
            "knowledge_ready": row.profile is not None and row.fishing is not None,
            "publish_ready": bool(
                row.status == "ACTIVE"
                and row.cover is not None
                and row.cover.status == "ACTIVE"
                and row.cover.image_url.strip()
                and len({
                    normalize_card_type(card.card_type)
                    for card in row.cards
                    if card.status == "ACTIVE" and card.image_url.strip()
                }) == len(CARD_TYPE_ORDER)
            ),
        }
        for row in rows
    ]


@router.get("/species/{species_id}")
def get_admin_species(species_id: str, db: Session = Depends(get_db)):
    row = load_species_with_knowledge(db, species_id, active_only=False)
    if row is None or row.status == "DELETED":
        raise HTTPException(status_code=404, detail="fish species not found")
    return build_species_detail(row, include_inactive_similarity=True)


@router.post("/species", status_code=201)
def create_admin_species(payload: SpeciesCreate, db: Session = Depends(get_db)) -> dict:
    if db.get(FishSpecies, payload.id) is not None:
        raise HTTPException(status_code=409, detail="fish species already exists")
    catalog = db.get(SpeciesCatalog, payload.id)
    if catalog is None:
        name_owner = db.scalar(select(SpeciesCatalog).where(SpeciesCatalog.common_name_zh == payload.name_cn))
        if name_owner is not None:
            raise HTTPException(status_code=409, detail="name_cn already belongs to another species id")
        max_order = db.scalar(select(func.max(SpeciesCatalog.catalog_order)))
        next_order = int(max_order) + 1 if max_order is not None else 0
        catalog = SpeciesCatalog(
            species_key=payload.id,
            catalog_order=next_order,
            common_name_zh=payload.name_cn,
            scientific_name=payload.scientific_name,
            status="candidate",
            is_other=False,
            notes="由 Fish Knowledge Admin 创建；训练状态需在 Species Catalog 独立确认",
        )
        db.add(catalog)
    row = FishSpecies(**_species_model_values(payload))
    db.add(row)
    db.flush()
    hero = _sync_hero_fields(db, row, payload)
    _commit(db)
    db.refresh(row)
    return _admin_species_dict(row, hero=hero)


@router.patch("/species/{species_id}")
def update_admin_species(species_id: str, payload: SpeciesPatch, db: Session = Depends(get_db)) -> dict:
    row = _require_species(db, species_id)
    species_fields = {
        "name_cn",
        "alias",
        "scientific_name",
        "category",
        "family",
        "genus",
        "summary",
        "status",
    }
    for field in payload.model_fields_set:
        if field not in species_fields:
            continue
        value = getattr(payload, field)
        if field in {"name_cn", "category", "status"} and value is None:
            raise HTTPException(status_code=400, detail=f"{field} 不能为空")
        setattr(row, field, value)
    hero = _sync_hero_fields(db, row, payload)
    _commit(db)
    return _admin_species_dict(row, hero=hero)


@router.delete("/species/{species_id}")
def delete_admin_species(species_id: str, db: Session = Depends(get_db)) -> dict:
    """Soft-delete a species while retaining its content for audit/recovery."""

    row = _require_species(db, species_id, include_deleted=True)
    if row.status == "DELETED":
        raise HTTPException(status_code=404, detail="fish species not found")
    row.status = "DELETED"
    _commit(db)
    return {"deleted": True, "id": row.id, "species_id": row.id, "status": row.status}


def _publish_species(species_id: str, db: Session) -> dict:
    row = _require_species(db, species_id)
    missing = _publication_missing(db, row)
    if missing:
        raise HTTPException(status_code=409, detail={"success": False, "missing": missing})

    # Publishing is the DRAFT -> ACTIVE transition for the complete asset
    # package. Operators can upload/save content in DRAFT first and use this
    # single action to publish the Cover and one image-backed card per type.
    cover = _get_species_cover(db, row.id)
    if cover is not None:
        cover.status = "ACTIVE"
    cards = db.scalars(
        select(FishCard).where(FishCard.species_id == row.id).order_by(FishCard.sort_order, FishCard.id)
    ).all()
    for card_type in CARD_TYPE_ORDER:
        candidates = [
            card
            for card in cards
            if normalize_card_type(card.card_type) == card_type and (card.image_url or "").strip()
        ]
        active = next((card for card in candidates if card.status == "ACTIVE"), None)
        if active is None and candidates:
            candidates[0].status = "ACTIVE"
    row.status = "ACTIVE"
    _commit(db)
    return {"success": True, "id": row.id, "species_id": row.id, "status": row.status, "missing": []}


@router.post("/species/{species_id}/publish")
def publish_admin_species(species_id: str, db: Session = Depends(get_db)) -> dict:
    return _publish_species(species_id, db)


@router.get("/species/{species_id}/cover")
def get_species_cover(species_id: str, db: Session = Depends(get_db)) -> dict:
    species = _require_species(db, species_id)
    row = _get_species_cover(db, species.id)
    if row is None:
        raise HTTPException(status_code=404, detail="fish species cover not found")
    return _cover_dict(row)


@router.post("/species/{species_id}/cover", status_code=201)
def create_species_cover(species_id: str, payload: CoverCreate, db: Session = Depends(get_db)) -> dict:
    species = _require_species(db, species_id)
    if _get_species_cover(db, species.id) is not None:
        raise HTTPException(status_code=409, detail="fish species cover already exists")
    row = FishSpeciesCover(species_id=species.id, **payload.model_dump())
    db.add(row)
    _commit(db)
    db.refresh(row)
    return _cover_dict(row)


@router.patch("/species/{species_id}/cover")
def update_species_cover(species_id: str, payload: CoverPatch, db: Session = Depends(get_db)) -> dict:
    species = _require_species(db, species_id)
    row = _get_species_cover(db, species.id)
    if row is None:
        raise HTTPException(status_code=404, detail="fish species cover not found")
    for field in payload.model_fields_set:
        value = getattr(payload, field)
        if field == "image_url" and value is None:
            value = ""
        if field == "style" and value is None:
            raise HTTPException(status_code=400, detail="style 不能为空")
        setattr(row, field, value)
    _commit(db)
    return _cover_dict(row)


@router.delete("/species/{species_id}/cover")
def delete_species_cover(species_id: str, db: Session = Depends(get_db)) -> dict:
    species = _require_species(db, species_id)
    row = _get_species_cover(db, species.id)
    if row is None:
        raise HTTPException(status_code=404, detail="fish species cover not found")
    db.delete(row)
    _commit(db)
    return {"deleted": True, "species_id": species.id}


@router.get("/species/{species_id}/cards")
def list_species_cards(species_id: str, db: Session = Depends(get_db)) -> list[dict]:
    species = _require_species(db, species_id)
    rows = db.scalars(
        select(FishCard)
        .where(FishCard.species_id == species.id)
        .order_by(FishCard.sort_order, FishCard.id)
    ).all()
    return [_card_dict(row) for row in rows]


@router.post("/species/{species_id}/cards", status_code=201)
def create_species_card(species_id: str, payload: CardCreate, db: Session = Depends(get_db)) -> dict:
    species = _require_species(db, species_id)
    card_type = normalize_card_type(payload.card_type)
    if payload.status == "ACTIVE":
        _ensure_active_card_type(db, species.id, card_type)
    values = payload.model_dump(exclude={"card_type", "sort_order", "content"})
    if payload.content is not None:
        values["description"] = card_description(payload.content)
    row = FishCard(
        species_id=species.id,
        card_type=card_type,
        sort_order=payload.sort_order if payload.sort_order is not None else card_type_sort_order(card_type),
        **values,
    )
    db.add(row)
    _commit(db)
    db.refresh(row)
    return _card_dict(row)


@router.patch("/cards/{card_id}")
def update_species_card(card_id: int, payload: CardPatch, db: Session = Depends(get_db)) -> dict:
    row = db.get(FishCard, card_id)
    if row is None:
        raise HTTPException(status_code=404, detail="fish card not found")

    new_type = normalize_card_type(payload.card_type) if payload.card_type is not None else normalize_card_type(row.card_type)
    new_status = payload.status if payload.status is not None else row.status
    if new_status == "ACTIVE":
        _ensure_active_card_type(db, row.species_id, new_type, exclude_id=row.id)

    for field in payload.model_fields_set:
        value = getattr(payload, field)
        if field == "card_type":
            value = new_type
        elif field == "content":
            row.description = card_description(value or {})
            continue
        elif field in {"title", "image_url", "description"} and value is None:
            value = ""
        setattr(row, field, value)
    if "card_type" in payload.model_fields_set and "sort_order" not in payload.model_fields_set:
        row.sort_order = card_type_sort_order(new_type)
    # Structured content is the editor's source of truth. If an older client
    # sends both fields, do not let its legacy description overwrite content.
    if "content" in payload.model_fields_set:
        row.description = card_description(payload.content or {})
    _commit(db)
    return _card_dict(row)


@router.delete("/cards/{card_id}")
def delete_species_card(card_id: int, db: Session = Depends(get_db)) -> dict:
    row = db.get(FishCard, card_id)
    if row is None:
        raise HTTPException(status_code=404, detail="fish card not found")
    species_id = row.species_id
    db.delete(row)
    _commit(db)
    return {"deleted": True, "id": card_id, "species_id": species_id}


@router.get("/species/{species_id}/completion")
def species_completion(species_id: str, db: Session = Depends(get_db)) -> dict:
    species = _require_species(db, species_id)
    cover = _get_species_cover(db, species.id)
    cards = db.scalars(select(FishCard).where(FishCard.species_id == species.id)).all()
    gallery_count = int(
        db.scalar(select(func.count()).select_from(FishGalleryImage).where(FishGalleryImage.species_id == species.id))
        or 0
    )
    video_count = int(
        db.scalar(select(func.count()).select_from(FishVideo).where(FishVideo.species_id == species.id)) or 0
    )
    completed_types = {
        normalize_card_type(card.card_type)
        for card in cards
        if (card.image_url or "").strip()
        and normalize_card_type(card.card_type) in CARD_TYPE_ORDER
    }
    cards_by_type = {card_type: card_type in completed_types for card_type in CARD_TYPE_ORDER}
    knowledge_ready = bool(
        species.profile
        and species.fishing
        and (
            species.profile.body_shape
            or species.profile.features
            or species.profile.habitat
            or species.profile.food
            or species.profile.season
            or species.fishing.water_layer
            or species.fishing.bait
            or species.fishing.method
        )
    )
    return {
        "species_complete": bool(species.name_cn.strip() and species.category.strip() and species.summary.strip()),
        "cover": bool(cover is not None and (cover.image_url or "").strip()),
        "cards": {"completed": len(completed_types), "total": len(CARD_TYPE_ORDER), **cards_by_type},
        "gallery": {"completed": min(gallery_count, MAX_GALLERY_IMAGES), "total": MAX_GALLERY_IMAGES},
        "video": video_count > 0,
        "knowledge": knowledge_ready,
    }


@router.put("/species/{species_id}/profile")
def upsert_profile(species_id: str, payload: ProfileUpsert, db: Session = Depends(get_db)) -> dict:
    _require_species(db, species_id)
    row = db.get(FishProfile, species_id)
    values = payload.model_dump()
    if row is None:
        row = FishProfile(species_id=species_id, **values)
        db.add(row)
    else:
        for field, value in values.items():
            setattr(row, field, value)
    _commit(db)
    return {"species_id": species_id, **values}


@router.put("/species/{species_id}/fishing")
def upsert_fishing(species_id: str, payload: FishingUpsert, db: Session = Depends(get_db)) -> dict:
    _require_species(db, species_id)
    row = db.get(FishFishing, species_id)
    values = payload.model_dump()
    if row is None:
        row = FishFishing(species_id=species_id, **values)
        db.add(row)
    else:
        for field, value in values.items():
            setattr(row, field, value)
    _commit(db)
    return {"species_id": species_id, **values}


@router.post("/species/{species_id}/gallery", status_code=201)
def create_gallery_item(species_id: str, payload: GalleryCreate, db: Session = Depends(get_db)) -> dict:
    _require_species(db, species_id)
    _ensure_gallery_slot(db, species_id, payload.order)
    row = FishGalleryImage(
        species_id=species_id,
        type=payload.type,
        url=payload.url,
        title=payload.title,
        sort_order=payload.order,
    )
    db.add(row)
    _commit(db)
    db.refresh(row)
    return _gallery_dict(row)


@router.post("/species/{species_id}/gallery/upload", status_code=201)
async def upload_gallery_image(
    species_id: str,
    type: str = Form(...),
    order: int = Form(...),
    title: str | None = Form(default=None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    _require_species(db, species_id)
    if type not in GALLERY_TYPES:
        raise HTTPException(status_code=400, detail="不支持的 Gallery type")
    if order < 0 or order >= MAX_GALLERY_IMAGES:
        raise HTTPException(status_code=400, detail="Gallery order 必须在 0 到 4 之间")
    data = await file.read(MAX_GALLERY_IMAGE_BYTES + 1)
    try:
        metadata = inspect_gallery_image(data)
    except GalleryUploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    existing = db.scalar(
        select(FishGalleryImage).where(
            FishGalleryImage.species_id == species_id,
            FishGalleryImage.sha256 == metadata["sha256"],
        )
    )
    if existing is not None:
        return {**_gallery_dict(existing), "duplicate": True, "storage": "SKIP"}
    _ensure_gallery_slot(db, species_id, order)

    try:
        client = storage.Client()
        bucket = client.bucket(get_bucket_name())
        object_name, storage_status = store_gallery_image(
            client=client,
            bucket=bucket,
            species_id=species_id,
            data=data,
            metadata=metadata,
        )
        row = FishGalleryImage(
            species_id=species_id,
            type=type,
            url=f"managed://{metadata['sha256']}",
            title=(title or "").strip() or None,
            sort_order=order,
            object_name=object_name,
            content_type=str(metadata["content_type"]),
            size_bytes=int(metadata["size_bytes"]),
            sha256=str(metadata["sha256"]),
        )
        db.add(row)
        db.flush()
        row.url = f"/api/v1/fish/gallery/{row.id}/media"
        _commit(db)
        db.refresh(row)
    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail="知识库图片存储失败") from exc
    return {
        **_gallery_dict(row),
        "duplicate": False,
        "storage": storage_status,
        "image": {"width": metadata["width"], "height": metadata["height"]},
    }


@router.post("/species/{species_id}/assets/upload", status_code=201)
async def upload_fish_asset(
    species_id: str,
    asset_type: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict:
    """Upload a real cover/card asset and bind it to the existing row."""

    species = _require_species(db, species_id)
    normalized_type = "cover" if asset_type.strip().lower() == "cover" else normalize_card_type(asset_type)
    if normalized_type != "cover" and normalized_type not in CARD_TYPE_ORDER:
        raise HTTPException(status_code=400, detail="asset_type 必须是 cover 或五种鱼鉴卡类型")

    data = await file.read(MAX_GALLERY_IMAGE_BYTES + 1)
    try:
        metadata = inspect_gallery_image(data)
    except GalleryUploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        client = storage.Client()
        bucket = client.bucket(get_bucket_name())
        object_name, storage_status = store_knowledge_asset(
            client=client,
            bucket=bucket,
            species_id=species.id,
            asset_type=normalized_type,
            data=data,
            metadata=metadata,
        )
        asset_key = object_name.rsplit("/", 1)[-1]
        managed_url = f"/api/v1/fish/knowledge-media/{species.id}/{normalized_type.lower()}/{asset_key}"

        if normalized_type == "cover":
            row = _get_species_cover(db, species.id)
            if row is None:
                row = FishSpeciesCover(
                    species_id=species.id,
                    image_url=managed_url,
                    style="ANIME_CARD",
                    title=f"{species.name_cn}图鉴卡",
                    status="DRAFT",
                )
                db.add(row)
            else:
                row.image_url = managed_url
        else:
            row = next(
                (
                    card
                    for card in db.scalars(select(FishCard).where(FishCard.species_id == species.id)).all()
                    if normalize_card_type(card.card_type) == normalized_type
                ),
                None,
            )
            if row is None:
                row = FishCard(
                    species_id=species.id,
                    card_type=normalized_type,
                    title=f"{species.name_cn}{normalized_type}卡",
                    image_url=managed_url,
                    description="",
                    sort_order=card_type_sort_order(normalized_type),
                    status="DRAFT",
                )
                db.add(row)
            else:
                row.image_url = managed_url
        _commit(db)
        db.refresh(row)
    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail="鱼鉴素材存储失败") from exc

    result = _cover_dict(row) if normalized_type == "cover" else _card_dict(row)
    return {
        **result,
        "asset_type": normalized_type,
        "storage": storage_status,
        "image": {"width": metadata["width"], "height": metadata["height"]},
    }


def _cms_asset_error(
    error: str,
    message: str,
    *,
    status_code: int,
    reason: str | None = None,
    extra: dict[str, Any] | None = None,
) -> JSONResponse:
    payload: dict[str, Any] = {
        "success": False,
        "error": error,
        "message": message,
    }
    if reason:
        payload["reason"] = reason
    if extra:
        payload.update(extra)
    return JSONResponse(status_code=status_code, content=payload)


async def upload_cms_fish_asset(
    species_id: str,
    asset_type: str,
    file: UploadFile,
    db: Session,
) -> dict[str, Any] | JSONResponse:
    """Upload and bind one Cover/Card image for the short CMS contract."""

    normalized_type = str(asset_type or "").strip().upper()
    if normalized_type not in KNOWLEDGE_ASSET_TYPES:
        return _cms_asset_error(
            "invalid_asset_type",
            "asset_type 必须是 COVER、HERO、IDENTIFICATION、ECO、GEAR 或 SKILL",
            status_code=400,
        )

    requested_species_id = str(species_id or "").strip()
    try:
        species = _require_species(db, requested_species_id)
    except HTTPException as exc:
        return _cms_asset_error(
            "species_not_found",
            str(exc.detail),
            status_code=exc.status_code,
        )

    if file is None:
        return _cms_asset_error("invalid_file", "请先选择图片文件", status_code=400, reason="empty_file")
    try:
        data = await file.read(KNOWLEDGE_ASSET_MAX_BYTES + 1)
    except Exception:
        return _cms_asset_error("invalid_file", "图片文件读取失败", status_code=400)

    try:
        image_metadata = inspect_knowledge_asset(data)
    except GalleryUploadError as exc:
        reason = "empty_file" if not data else (
            "file_too_large" if len(data) > KNOWLEDGE_ASSET_MAX_BYTES else "unsupported_format"
        )
        return _cms_asset_error("invalid_file", str(exc), status_code=400, reason=reason)

    asset_written = False
    object_name: str | None = None
    image_url: str | None = None
    try:
        bucket_name = get_bucket_name()
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        object_name, storage_status = store_cms_knowledge_asset(
            client=client,
            bucket=bucket,
            species_id=species.id,
            asset_type=normalized_type,
            data=bytes(image_metadata["webp_data"]),
            image_metadata=image_metadata,
        )
        asset_written = True
        # The runtime service account can read the bucket, but the bucket is
        # intentionally not public. Persist a same-origin managed URL so the
        # CMS preview and the App can read the image through YuJian's public
        # Fish Knowledge media endpoint.
        asset_key = object_name.rsplit("/", 1)[-1]
        image_url = f"/api/v1/fish/knowledge-media/{species.id}/{normalized_type.lower()}/{asset_key}"

        if normalized_type == "COVER":
            row = _get_species_cover(db, species.id)
            if row is None:
                row = FishSpeciesCover(
                    species_id=species.id,
                    image_url=image_url,
                    style="ANIME_CARD",
                    title=f"{species.name_cn}图鉴卡",
                    status="DRAFT",
                )
                db.add(row)
            else:
                row.image_url = image_url
            binding = "cover.image_url"
        else:
            row = next(
                (
                    card
                    for card in db.scalars(
                        select(FishCard)
                        .where(FishCard.species_id == species.id)
                        .order_by(FishCard.id)
                    ).all()
                    if normalize_card_type(card.card_type) == normalized_type
                ),
                None,
            )
            if row is None:
                row = FishCard(
                    species_id=species.id,
                    card_type=normalized_type,
                    title=f"{species.name_cn}{normalized_type}卡",
                    image_url=image_url,
                    description=card_description({"type": normalized_type}),
                    sort_order=card_type_sort_order(normalized_type),
                    status="DRAFT",
                )
                db.add(row)
            else:
                row.image_url = image_url
            binding = "card.image_url"
        _commit(db)
        db.refresh(row)
    except HTTPException as exc:
        db.rollback()
        if asset_written:
            return _cms_asset_error(
                "binding_error",
                "图片已上传，但绑定保存失败",
                status_code=503,
                extra={
                    "url": image_url,
                    "asset_type": normalized_type,
                    "species_id": species.id,
                    "storage": {"bucket": bucket_name, "object_name": object_name},
                },
            )
        return _cms_asset_error("storage_error", "鱼鉴图片上传失败", status_code=503)
    except Exception:
        db.rollback()
        if asset_written:
            return _cms_asset_error(
                "binding_error",
                "图片已上传，但绑定保存失败",
                status_code=503,
                extra={
                    "url": image_url,
                    "asset_type": normalized_type,
                    "species_id": species.id,
                    "storage": {"bucket": bucket_name, "object_name": object_name},
                },
            )
        return _cms_asset_error("storage_error", "鱼鉴图片上传失败", status_code=503)

    result = _cover_dict(row) if normalized_type == "COVER" else _card_dict(row)
    return {
        "success": True,
        "url": image_url,
        "asset_type": normalized_type,
        "species_id": species.id,
        "width": image_metadata["width"],
        "height": image_metadata["height"],
        "image": {
            "width": image_metadata["width"],
            "height": image_metadata["height"],
            "size_bytes": image_metadata["size_bytes"],
            "stored_size_bytes": image_metadata["stored_size_bytes"],
        },
        "storage": {
            "bucket": bucket_name,
            "object_name": object_name,
            "content_type": "image/webp",
            "status": storage_status,
        },
        "binding": binding,
        **result,
    }


@router.patch("/species/{species_id}/gallery/{image_id}")
def update_gallery_item(
    species_id: str,
    image_id: int,
    payload: GalleryPatch,
    db: Session = Depends(get_db),
) -> dict:
    row = db.get(FishGalleryImage, image_id)
    if row is None or row.species_id != species_id:
        raise HTTPException(status_code=404, detail="gallery image not found")
    if payload.order is not None:
        _ensure_gallery_slot(db, species_id, payload.order, exclude_id=image_id)
        row.sort_order = payload.order
    for field in payload.model_fields_set - {"order"}:
        value = getattr(payload, field)
        if field in {"type", "url"} and value is None:
            raise HTTPException(status_code=400, detail=f"{field} 不能为空")
        setattr(row, field, value)
    _commit(db)
    return _gallery_dict(row)


@router.delete("/species/{species_id}/gallery/{image_id}")
def delete_gallery_item(species_id: str, image_id: int, db: Session = Depends(get_db)) -> dict:
    row = db.get(FishGalleryImage, image_id)
    if row is None or row.species_id != species_id:
        raise HTTPException(status_code=404, detail="gallery image not found")
    retained_object = row.object_name
    db.delete(row)
    _commit(db)
    return {
        "deleted": True,
        "id": image_id,
        "storage_object_retained": bool(retained_object),
    }


@router.post("/species/{species_id}/videos", status_code=201)
def create_video(species_id: str, payload: VideoCreate, db: Session = Depends(get_db)) -> dict:
    _require_species(db, species_id)
    values = payload.model_dump(exclude={"order"})
    row = FishVideo(species_id=species_id, sort_order=payload.order, **values)
    db.add(row)
    _commit(db)
    db.refresh(row)
    return _video_dict(row)


@router.patch("/species/{species_id}/videos/{video_id}")
def update_video(
    species_id: str,
    video_id: int,
    payload: VideoPatch,
    db: Session = Depends(get_db),
) -> dict:
    row = db.get(FishVideo, video_id)
    if row is None or row.species_id != species_id:
        raise HTTPException(status_code=404, detail="video not found")
    for field in payload.model_fields_set:
        value = getattr(payload, field)
        if field in {"title", "type", "video_url", "duration"} and value is None:
            raise HTTPException(status_code=400, detail=f"{field} 不能为空")
        setattr(row, "sort_order" if field == "order" else field, value)
    _commit(db)
    return _video_dict(row)


@router.delete("/species/{species_id}/videos/{video_id}")
def delete_video(species_id: str, video_id: int, db: Session = Depends(get_db)) -> dict:
    row = db.get(FishVideo, video_id)
    if row is None or row.species_id != species_id:
        raise HTTPException(status_code=404, detail="video not found")
    db.delete(row)
    _commit(db)
    return {"deleted": True, "id": video_id}


@router.put("/species/{species_id}/similarity/{similar_species_id}")
def upsert_similarity(
    species_id: str,
    similar_species_id: str,
    payload: SimilarityUpsert,
    db: Session = Depends(get_db),
) -> dict:
    _require_species(db, species_id)
    _require_species(db, similar_species_id)
    if species_id == similar_species_id:
        raise HTTPException(status_code=400, detail="鱼种不能与自身建立相似关系")
    row = db.scalar(
        select(FishSimilarity).where(
            FishSimilarity.species_id == species_id,
            FishSimilarity.similar_species_id == similar_species_id,
        )
    )
    if row is None:
        row = FishSimilarity(
            species_id=species_id,
            similar_species_id=similar_species_id,
            difference=payload.difference.strip(),
        )
        db.add(row)
    else:
        row.difference = payload.difference.strip()
    _commit(db)
    return {
        "species_id": species_id,
        "similar_species_id": similar_species_id,
        "difference": row.difference,
    }


@router.delete("/species/{species_id}/similarity/{similar_species_id}")
def delete_similarity(species_id: str, similar_species_id: str, db: Session = Depends(get_db)) -> dict:
    row = db.scalar(
        select(FishSimilarity).where(
            FishSimilarity.species_id == species_id,
            FishSimilarity.similar_species_id == similar_species_id,
        )
    )
    if row is None:
        raise HTTPException(status_code=404, detail="similarity not found")
    db.delete(row)
    _commit(db)
    return {"deleted": True, "species_id": species_id, "similar_species_id": similar_species_id}


# ---------------------------------------------------------------------------
# Short CRUD contract aliases
# ---------------------------------------------------------------------------
# The CMS itself uses the versioned paths above. These aliases keep the
# documented /api/admin/fish contract available to simple operators and make
# the CRUD surface independent from the older v1.1 route naming.


@compat_router.post("/assets/upload", response_model=None)
async def compat_upload_cms_fish_asset(
    file: UploadFile = File(...),
    species_id: str = Form(...),
    asset_type: str = Form(...),
    db: Session = Depends(get_db),
):
    return await upload_cms_fish_asset(species_id, asset_type, file, db)


@compat_router.get("/species")
def compat_list_admin_species(db: Session = Depends(get_db)) -> list[dict]:
    return list_admin_species(db)


@compat_router.get("/species/{species_id}")
def compat_get_admin_species(species_id: str, db: Session = Depends(get_db)):
    return get_admin_species(species_id, db)


@compat_router.post("/species", status_code=201)
def compat_create_admin_species(payload: SpeciesCreate, db: Session = Depends(get_db)) -> dict:
    return create_admin_species(payload, db)


@compat_router.put("/species/{species_id}")
def compat_update_admin_species(species_id: str, payload: SpeciesPatch, db: Session = Depends(get_db)) -> dict:
    if payload.status != "ACTIVE":
        return update_admin_species(species_id, payload, db)
    values = payload.model_dump(exclude_unset=True, exclude={"status"})
    updated = update_admin_species(species_id, SpeciesPatch(**values), db)
    # The short CRUD contract treats an ACTIVE update as a publish action, so
    # it receives the same completeness validation as the explicit publish
    # endpoint. The legacy PATCH endpoint remains backward compatible.
    return _publish_species(species_id, db) | {"species": updated}


@compat_router.delete("/species/{species_id}")
def compat_delete_admin_species(species_id: str, db: Session = Depends(get_db)) -> dict:
    return delete_admin_species(species_id, db)


@compat_router.post("/species/{species_id}/publish")
def compat_publish_admin_species(species_id: str, db: Session = Depends(get_db)) -> dict:
    return _publish_species(species_id, db)


@compat_router.get("/species/{species_id}/completion")
def compat_species_completion(species_id: str, db: Session = Depends(get_db)) -> dict:
    return species_completion(species_id, db)


@compat_router.get("/species/{species_id}/cover")
def compat_get_species_cover(species_id: str, db: Session = Depends(get_db)) -> dict:
    return get_species_cover(species_id, db)


@compat_router.put("/species/{species_id}/cover")
def compat_put_species_cover(species_id: str, payload: CoverPut, db: Session = Depends(get_db)) -> dict:
    existing = _get_species_cover(db, _require_species(db, species_id).id)
    if existing is None:
        return create_species_cover(
            species_id,
            CoverCreate(
                image_url=payload.url or "",
                style=payload.style or "ANIME_CARD",
                title=payload.title or "",
                status=payload.status or "DRAFT",
            ),
            db,
        )

    values = {}
    if "url" in payload.model_fields_set:
        values["image_url"] = payload.url
    if "style" in payload.model_fields_set:
        values["style"] = payload.style
    if "title" in payload.model_fields_set:
        values["title"] = payload.title
    if "status" in payload.model_fields_set:
        values["status"] = payload.status
    return update_species_cover(species_id, CoverPatch(**values), db)


@compat_router.delete("/species/{species_id}/cover")
def compat_delete_species_cover(species_id: str, db: Session = Depends(get_db)) -> dict:
    return delete_species_cover(species_id, db)


def _compat_card_type(value: str) -> str:
    normalized = normalize_card_type(value)
    if normalized not in CARD_TYPE_ORDER:
        raise HTTPException(status_code=400, detail="card_type 必须是 HERO、IDENTIFICATION、ECO、GEAR 或 SKILL")
    return normalized


@compat_router.put("/species/{species_id}/cards/{card_type}")
def compat_put_species_card(
    species_id: str,
    card_type: str,
    payload: CardPatch,
    db: Session = Depends(get_db),
) -> dict:
    normalized = _compat_card_type(card_type)
    species = _require_species(db, species_id)
    existing = next(
        (
            card
            for card in db.scalars(
                select(FishCard)
                .where(FishCard.species_id == species.id)
                .order_by(FishCard.id)
            ).all()
            if normalize_card_type(card.card_type) == normalized
        ),
        None,
    )
    values = payload.model_dump(exclude_unset=True)
    values.pop("card_type", None)
    if existing is not None:
        values["card_type"] = normalized
        return update_species_card(existing.id, CardPatch(**values), db)
    return create_species_card(
        species.id,
        CardCreate(card_type=normalized, **values),
        db,
    )


@compat_router.get("/species/{species_id}/cards")
def compat_list_species_cards(species_id: str, db: Session = Depends(get_db)) -> list[dict]:
    return list_species_cards(species_id, db)


@compat_router.delete("/cards/{card_id}")
def compat_delete_species_card(card_id: int, db: Session = Depends(get_db)) -> dict:
    return delete_species_card(card_id, db)


@compat_router.put("/species/{species_id}/profile")
def compat_put_profile(species_id: str, payload: ProfileUpsert, db: Session = Depends(get_db)) -> dict:
    return upsert_profile(species_id, payload, db)


@compat_router.put("/species/{species_id}/fishing")
def compat_put_fishing(species_id: str, payload: FishingUpsert, db: Session = Depends(get_db)) -> dict:
    return upsert_fishing(species_id, payload, db)
