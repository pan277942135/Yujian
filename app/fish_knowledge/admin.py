from __future__ import annotations

import re
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
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
    MAX_GALLERY_IMAGE_BYTES,
    MAX_GALLERY_IMAGES,
    FishGalleryImage,
    GalleryUploadError,
    inspect_gallery_image,
    store_knowledge_asset,
    store_gallery_image,
)
from app.fish_knowledge.profile import FishProfile
from app.fish_knowledge.similarity import FishSimilarity
from app.fish_knowledge.species import FishSpecies
from app.fish_knowledge.video import FishVideo
from app.models import SpeciesCatalog


router = APIRouter(prefix="/api/v1/admin/fish", tags=["fish-knowledge-admin"])
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
    if not result.startswith("https://"):
        raise ValueError("URL 必须使用 https://")
    if len(result) > 2048:
        raise ValueError("URL 过长")
    return result


class SpeciesCreate(BaseModel):
    id: str = Field(min_length=2, max_length=128)
    name_cn: str = Field(min_length=1, max_length=128)
    alias: list[str] = Field(default_factory=list)
    scientific_name: str | None = Field(default=None, max_length=256)
    category: str = Field(min_length=1, max_length=64)
    family: str | None = Field(default=None, max_length=128)
    genus: str | None = Field(default=None, max_length=128)
    summary: str = Field(min_length=1, max_length=500)
    status: Literal["ACTIVE", "DRAFT"] = "DRAFT"

    @field_validator("id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        result = value.strip()
        if not SPECIES_ID_RE.fullmatch(result):
            raise ValueError("id 必须是小写字母开头的 snake_case")
        return result

    @field_validator("name_cn", "category", "summary")
    @classmethod
    def required_text(cls, value: str) -> str:
        result = value.strip()
        if not result:
            raise ValueError("字段不能为空")
        return result

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
    name_cn: str | None = Field(default=None, min_length=1, max_length=128)
    alias: list[str] | None = None
    scientific_name: str | None = Field(default=None, max_length=256)
    category: str | None = Field(default=None, min_length=1, max_length=64)
    family: str | None = Field(default=None, max_length=128)
    genus: str | None = Field(default=None, max_length=128)
    summary: str | None = Field(default=None, min_length=1, max_length=500)
    status: Literal["ACTIVE", "DRAFT"] | None = None

    @field_validator("name_cn", "category", "summary")
    @classmethod
    def required_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        result = value.strip()
        if not result:
            raise ValueError("字段不能为空")
        return result

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


def _commit(db: Session) -> None:
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="fish knowledge data conflicts with an existing record") from exc


def _require_species(db: Session, species_id: str) -> FishSpecies:
    row = db.get(FishSpecies, species_id)
    if row is None:
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
        "image_url": row.image_url,
        "style": row.style,
        "title": row.title,
        "status": row.status,
    }


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
        "image_url": row.image_url,
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
    if row is None:
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
    row = FishSpecies(**payload.model_dump())
    db.add(row)
    _commit(db)
    db.refresh(row)
    return {
        "id": row.id,
        "name_cn": row.name_cn,
        "alias": row.alias,
        "scientific_name": row.scientific_name,
        "category": row.category,
        "family": row.family,
        "genus": row.genus,
        "summary": row.summary,
        "status": row.status,
    }


@router.patch("/species/{species_id}")
def update_admin_species(species_id: str, payload: SpeciesPatch, db: Session = Depends(get_db)) -> dict:
    row = _require_species(db, species_id)
    for field in payload.model_fields_set:
        value = getattr(payload, field)
        if field in {"name_cn", "category", "summary", "status"} and value is None:
            raise HTTPException(status_code=400, detail=f"{field} 不能为空")
        setattr(row, field, value)
    _commit(db)
    return {
        "id": row.id,
        "name_cn": row.name_cn,
        "alias": row.alias,
        "scientific_name": row.scientific_name,
        "category": row.category,
        "family": row.family,
        "genus": row.genus,
        "summary": row.summary,
        "status": row.status,
    }


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
