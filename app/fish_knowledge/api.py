from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse, Response
from google.cloud import storage
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db import get_db
from app.factory import DOWNLOAD_RETRY, get_bucket_name
from app.fish_knowledge.asset_types import ALL_ASSET_TYPES, COVER_ASSET_TYPES, asset_slot_key, normalize_asset_type
from app.fish_knowledge.cards import FishCard, normalize_card_type
from app.fish_knowledge.cover import FishSpeciesCover
from app.fish_knowledge.content import card_display_description, parse_card_content
from app.fish_knowledge.fishing import FishFishing
from app.fish_knowledge.gallery import FishGalleryImage, managed_knowledge_asset_url
from app.fish_knowledge.profile import FishProfile
from app.fish_knowledge.similarity import FishSimilarity
from app.fish_knowledge.species import SPECIES_ID_ALIASES, FishSpecies
from app.fish_knowledge.video import FishVideo
from app.models import SpeciesCatalog
from app.platform.models import FishAsset


router = APIRouter(prefix="/api/v1/fish", tags=["fish-knowledge"])


class SpeciesListItem(BaseModel):
    id: str
    name_cn: str
    category: str
    cover_image: str | None
    summary: str
    # Omitted for legacy-only rows; present when the new fish_asset index has
    # a canonical Cover Card slot.
    cover_assets: dict[str, str | None] | None = None


class SpeciesOut(BaseModel):
    id: str
    name_cn: str
    alias: list[str]
    scientific_name: str | None
    category: str
    family: str | None
    genus: str | None
    summary: str
    status: str
    cover_image: str | None


class GalleryImageOut(BaseModel):
    id: int
    type: str
    url: str
    title: str | None
    order: int


class GalleryOut(BaseModel):
    species_id: str
    images: list[GalleryImageOut]


class ProfileOut(BaseModel):
    species_id: str
    body_shape: str | None
    features: list[str]
    habitat: list[str]
    food: str | None
    season: list[str]


class FishingOut(BaseModel):
    species_id: str
    water_layer: str | None
    season: list[str]
    bait: list[str]
    method: list[str]
    summary: str


class VideoOut(BaseModel):
    id: int
    species_id: str
    title: str
    type: str
    cover_url: str | None
    video_url: str
    duration: int
    tags: list[str]


class SimilarityOut(BaseModel):
    species_id: str
    similar_species_id: str
    similar_species_name_cn: str
    difference: str


class CoverOut(BaseModel):
    id: int
    species_id: str
    image_url: str
    style: str
    title: str
    status: str


class CardOut(BaseModel):
    id: int
    species_id: str
    card_type: str
    type: str
    title: str
    image_url: str
    description: str
    content: dict[str, Any]
    sort_order: int
    status: str


class SpeciesDetailOut(BaseModel):
    species: SpeciesOut
    gallery: GalleryOut
    profile: ProfileOut
    fishing: FishingOut
    videos: list[VideoOut]
    similarity: list[SimilarityOut]


class SpeciesFullDetailOut(SpeciesDetailOut):
    cover: dict[str, Any]
    cards: list[CardOut]
    knowledge: dict[str, Any]
    dynamic: dict[str, Any]
    cover_assets: dict[str, str | None]
    knowledge_assets: dict[str, str | None]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _gallery_item(row: FishGalleryImage) -> GalleryImageOut:
    return GalleryImageOut(
        id=row.id,
        type=row.type,
        url=row.url,
        title=row.title,
        order=row.sort_order,
    )


def _profile(species_id: str, row: FishProfile | None) -> ProfileOut:
    return ProfileOut(
        species_id=species_id,
        body_shape=row.body_shape if row else None,
        features=_string_list(row.features if row else None),
        habitat=_string_list(row.habitat if row else None),
        food=row.food if row else None,
        season=_string_list(row.season if row else None),
    )


def _fishing(species_id: str, row: FishFishing | None) -> FishingOut:
    return FishingOut(
        species_id=species_id,
        water_layer=row.water_layer if row else None,
        season=_string_list(row.season if row else None),
        bait=_string_list(row.bait if row else None),
        method=_string_list(row.method if row else None),
        summary=row.summary if row else "",
    )


def _video(row: FishVideo) -> VideoOut:
    return VideoOut(
        id=row.id,
        species_id=row.species_id,
        title=row.title,
        type=row.type,
        cover_url=row.cover_url,
        video_url=row.video_url,
        duration=row.duration,
        tags=_string_list(row.tags),
    )


def _cover_dict(row: FishSpeciesCover | None, *, active_only: bool = True) -> dict[str, Any]:
    if row is None or (active_only and row.status != "ACTIVE"):
        return {}
    return {
        "id": row.id,
        "species_id": row.species_id,
        "image_url": managed_knowledge_asset_url(row.species_id, "COVER", row.image_url),
        "style": row.style,
        "title": row.title,
        "status": row.status,
    }


def _knowledge_asset_maps(db: Session | None, species: FishSpecies) -> tuple[dict[str, str | None], dict[str, str | None]]:
    """Build the additive 3+5 response without removing v1 fields."""

    cover_assets = {
        "cover_card": None,
        "transparent_left": None,
        "transparent_right": None,
    }
    knowledge_assets = {
        "hero": None,
        "identification": None,
        "eco": None,
        "gear": None,
        "skill": None,
    }
    allowed_statuses = {"ACTIVE", "PUBLISHED", "READY"}
    if db is not None:
        rows = db.scalars(
            select(FishAsset)
            .where(
                FishAsset.species.in_({species.id, species.name_cn}),
                FishAsset.asset_type.in_(ALL_ASSET_TYPES),
                FishAsset.status.in_(allowed_statuses),
            )
            .order_by(FishAsset.created_at.desc(), FishAsset.asset_id.desc())
        ).all()
        for row in rows:
            key = asset_slot_key(row.asset_type, row.direction)
            url = (row.asset_uri or "").strip()
            if key in cover_assets and url and cover_assets[key] is None:
                cover_assets[key] = url
            if key in knowledge_assets and url and knowledge_assets[key] is None:
                knowledge_assets[key] = url

    if cover_assets["cover_card"] is None and species.cover is not None and species.cover.status in allowed_statuses:
        cover_assets["cover_card"] = managed_knowledge_asset_url(species.id, "COVER", species.cover.image_url)
    for card in species.cards:
        card_type = normalize_card_type(card.card_type)
        key = asset_slot_key(card_type)
        if key in knowledge_assets and knowledge_assets[key] is None and card.status in allowed_statuses:
            knowledge_assets[key] = managed_knowledge_asset_url(species.id, card_type, card.image_url)
    return cover_assets, knowledge_assets


def _cover_image(species: FishSpecies) -> str | None:
    if species.cover is not None and species.cover.status == "ACTIVE" and species.cover.image_url.strip():
        return managed_knowledge_asset_url(species.id, "COVER", species.cover.image_url)
    return species.gallery[0].url if species.gallery else None


def _card(row: FishCard) -> CardOut:
    card_type = normalize_card_type(row.card_type)
    content = parse_card_content(row.description)
    return CardOut(
        id=row.id,
        species_id=row.species_id,
        card_type=card_type,
        type=card_type,
        title=row.title,
        image_url=managed_knowledge_asset_url(row.species_id, card_type, row.image_url),
        description=card_display_description(content, row.description),
        content=content,
        sort_order=row.sort_order,
        status=row.status,
    )


def _species(species: FishSpecies) -> SpeciesOut:
    return SpeciesOut(
        id=species.id,
        name_cn=species.name_cn,
        alias=_string_list(species.alias),
        scientific_name=species.scientific_name,
        category=species.category,
        family=species.family,
        genus=species.genus,
        summary=species.summary,
        status=species.status,
        cover_image=_cover_image(species),
    )


def _active_species_query():
    return (
        select(FishSpecies)
        .join(SpeciesCatalog, SpeciesCatalog.species_key == FishSpecies.id)
        .where(FishSpecies.status == "ACTIVE")
        .options(selectinload(FishSpecies.gallery), selectinload(FishSpecies.cover))
        .order_by(SpeciesCatalog.catalog_order, FishSpecies.id)
    )


def _knowledge_options():
    return (
        selectinload(FishSpecies.gallery),
        selectinload(FishSpecies.cover),
        selectinload(FishSpecies.cards),
        selectinload(FishSpecies.profile),
        selectinload(FishSpecies.fishing),
        selectinload(FishSpecies.videos),
        selectinload(FishSpecies.similarities).selectinload(FishSimilarity.similar_species),
    )


def load_species_with_knowledge(
    db: Session,
    species_id: str,
    *,
    active_only: bool,
) -> FishSpecies | None:
    requested_id = species_id.strip()
    statement = select(FishSpecies).where(FishSpecies.id == requested_id).options(*_knowledge_options())
    if not db.get(FishSpecies, requested_id):
        alias = SPECIES_ID_ALIASES.get(requested_id)
        if alias:
            statement = select(FishSpecies).where(FishSpecies.id == alias).options(*_knowledge_options())
    if active_only:
        statement = statement.where(FishSpecies.status == "ACTIVE")
    return db.scalar(statement)


def build_species_detail(
    row: FishSpecies,
    *,
    include_inactive_similarity: bool = False,
) -> SpeciesDetailOut:
    gallery = [_gallery_item(item) for item in row.gallery[:5]]
    similarity = [
        SimilarityOut(
            species_id=item.species_id,
            similar_species_id=item.similar_species_id,
            similar_species_name_cn=item.similar_species.name_cn,
            difference=item.difference,
        )
        for item in row.similarities
        if item.similar_species is not None
        and (include_inactive_similarity or item.similar_species.status == "ACTIVE")
    ]
    return SpeciesDetailOut(
        species=_species(row),
        gallery=GalleryOut(species_id=row.id, images=gallery),
        profile=_profile(row.id, row.profile),
        fishing=_fishing(row.id, row.fishing),
        videos=[_video(item) for item in row.videos],
        similarity=similarity,
    )


def build_species_full_detail(row: FishSpecies, db: Session | None = None) -> SpeciesFullDetailOut:
    base = build_species_detail(row)
    cover_assets, knowledge_assets = _knowledge_asset_maps(db, row)
    active_card_rows = [item for item in row.cards if item.status == "ACTIVE"]
    cards = [_card(item) for item in active_card_rows]
    profile = base.profile
    fishing = base.fishing
    card_content = {
        normalize_card_type(item.card_type): parse_card_content(item.description)
        for item in active_card_rows
    }
    ecology = card_content.get("ECO", {})
    gear = card_content.get("GEAR", {})
    skill = card_content.get("SKILL", {})
    return SpeciesFullDetailOut(
        species=base.species,
        cover=_cover_dict(row.cover),
        cards=cards,
        gallery=base.gallery,
        profile=profile,
        fishing=fishing,
        videos=base.videos,
        similarity=base.similarity,
        knowledge={
            "body_shape": profile.body_shape,
            "features": profile.features,
            "habitat": profile.habitat,
            "food": profile.food,
            "season": profile.season,
            "water_layer": fishing.water_layer,
            "bait": fishing.bait,
            "method": fishing.method,
            "display_tag": card_content.get("HERO", {}).get("tag"),
            "ecology": {
                "habitat": ecology.get("habitat", profile.habitat),
                "water_layer": ecology.get("water_layer", fishing.water_layer),
                "season": ecology.get("season", "、".join(profile.season)),
                "behavior": ecology.get("behavior", ""),
                "diet": ecology.get("diet", profile.food),
            },
            "gear": {
                "method": gear.get("method", fishing.method),
                "rod": gear.get("rod", ""),
                "line": gear.get("line", ""),
                "hook": gear.get("hook", ""),
                "bait": gear.get("bait", fishing.bait),
            },
            "skill": {
                "find": skill.get("find", ""),
                "attract": skill.get("attract", ""),
                "action": skill.get("action", ""),
                "tip": skill.get("tip", ""),
            },
        },
        # Dynamic user catches/rankings are intentionally a stable placeholder
        # until their separate content domain is implemented.
        dynamic={},
        cover_assets=cover_assets,
        knowledge_assets=knowledge_assets,
    )


@router.get("/species", response_model=list[SpeciesListItem], response_model_exclude_none=True)
def list_fish_species(db: Session = Depends(get_db)) -> list[SpeciesListItem]:
    rows = db.scalars(_active_species_query()).all()
    result = []
    for row in rows:
        cover_assets, _ = _knowledge_asset_maps(db, row)
        has_indexed_cover = db.scalar(
            select(FishAsset.asset_id).where(
                FishAsset.species.in_({row.id, row.name_cn}),
                FishAsset.asset_type.in_(COVER_ASSET_TYPES),
            ).limit(1)
        ) is not None
        result.append(
            SpeciesListItem(
                id=row.id,
                name_cn=row.name_cn,
                category=row.category,
                cover_image=_cover_image(row),
                summary=row.summary,
                cover_assets=cover_assets if has_indexed_cover else None,
            )
        )
    return result


@router.get("/species/{species_id}/detail", response_model=SpeciesFullDetailOut)
def get_fish_species_full_detail(species_id: str, db: Session = Depends(get_db)) -> SpeciesFullDetailOut:
    row = load_species_with_knowledge(db, species_id, active_only=True)
    if row is None:
        raise HTTPException(status_code=404, detail="fish species not found")
    return build_species_full_detail(row, db=db)


@router.get("/species/{species_id}", response_model=SpeciesDetailOut)
def get_fish_species(species_id: str, db: Session = Depends(get_db)) -> SpeciesDetailOut:
    row = load_species_with_knowledge(db, species_id, active_only=True)
    if row is None:
        raise HTTPException(status_code=404, detail="fish species not found")
    return build_species_detail(row)


@router.get("/gallery/{image_id}/media")
def get_gallery_media(image_id: int, db: Session = Depends(get_db)):
    row = db.scalar(
        select(FishGalleryImage)
        .join(FishSpecies, FishSpecies.id == FishGalleryImage.species_id)
        .where(FishGalleryImage.id == image_id, FishSpecies.status == "ACTIVE")
    )
    if row is None:
        raise HTTPException(status_code=404, detail="gallery image not found")
    if not row.object_name:
        if row.url.startswith("https://"):
            return RedirectResponse(row.url, status_code=307)
        raise HTTPException(status_code=404, detail="gallery media is not managed by YuJian")

    try:
        bucket_name = get_bucket_name()
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(row.object_name)
        if not blob.exists(client):
            raise HTTPException(status_code=404, detail="gallery object not found")
        content = blob.download_as_bytes(timeout=120, retry=DOWNLOAD_RETRY)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail="gallery storage is unavailable") from exc
    return Response(
        content=content,
        media_type=row.content_type or "application/octet-stream",
        headers={
            "Cache-Control": "public, max-age=86400",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/knowledge-media/{species_id}/{asset_type}/{asset_key}")
def get_knowledge_media(species_id: str, asset_type: str, asset_key: str, db: Session = Depends(get_db)):
    """Serve a managed cover/card image, including canonical 3+5 slots."""

    canonical_type = "COVER" if asset_type.strip().upper() == "COVER" else normalize_asset_type(asset_type)
    if canonical_type is None or canonical_type not in {"COVER", *ALL_ASSET_TYPES}:
        raise HTTPException(status_code=404, detail="knowledge asset not found")
    storage_type = "cover" if canonical_type == "COVER" else canonical_type.lower()
    fixed_asset_key = {
        "cover": "cover.webp",
        "cover_card": "cover_card.webp",
        "cover_card_transparent_left": "transparent_left.webp",
        "cover_card_transparent_right": "transparent_right.webp",
    }.get(storage_type, f"{storage_type}.webp")
    is_hashed_asset = bool(re.fullmatch(r"[a-f0-9]{64}\.(?:jpg|png|webp)", asset_key))
    is_version_asset = bool(re.fullmatch(r"v\d+\.webp", asset_key))
    if not is_hashed_asset and not is_version_asset and asset_key != fixed_asset_key:
        raise HTTPException(status_code=404, detail="knowledge asset not found")

    row = load_species_with_knowledge(db, species_id, active_only=False)
    if row is None:
        raise HTTPException(status_code=404, detail="fish species not found")
    expected_url = f"/api/v1/fish/knowledge-media/{row.id}/{storage_type}/{asset_key}"
    indexed = db.scalar(
        select(FishAsset).where(
            FishAsset.species.in_({row.id, row.name_cn}),
            FishAsset.asset_type == canonical_type if canonical_type in ALL_ASSET_TYPES else FishAsset.asset_type == "COVER_CARD",
            FishAsset.asset_uri == expected_url,
        )
    )
    version = None
    if is_version_asset:
        from app.fish_knowledge.import_batch import FishKnowledgeAssetVersion

        version = db.scalar(
            select(FishKnowledgeAssetVersion).where(
                FishKnowledgeAssetVersion.species_id == row.id,
                FishKnowledgeAssetVersion.asset_type == canonical_type,
                FishKnowledgeAssetVersion.image_url == expected_url,
                FishKnowledgeAssetVersion.status.in_({"ACTIVE", "DRAFT"}),
            )
        )
        is_referenced = version is not None and (
            version.status == "ACTIVE" or indexed is not None
        )
    elif indexed is not None:
        is_referenced = True
    elif canonical_type == "COVER":
        is_referenced = (
            row.cover is not None
            and managed_knowledge_asset_url(row.id, "COVER", row.cover.image_url) == expected_url
        )
    elif canonical_type == "COVER_CARD":
        is_referenced = (
            row.cover is not None
            and managed_knowledge_asset_url(row.id, "COVER_CARD", row.cover.image_url) == expected_url
        )
    else:
        is_referenced = any(
            normalize_card_type(card.card_type) == canonical_type
            and managed_knowledge_asset_url(row.id, canonical_type, card.image_url) == expected_url
            for card in row.cards
        )
    if not is_referenced:
        raise HTTPException(status_code=404, detail="knowledge asset not found")

    try:
        client = storage.Client()
        if version is not None:
            blob = client.bucket(get_bucket_name()).blob(version.object_name)
        else:
            object_prefix = "fish_knowledge" if is_hashed_asset else "fish-assets"
            if is_hashed_asset:
                object_directory = storage_type
            elif canonical_type == "COVER":
                object_directory = "cover"
            elif canonical_type in COVER_ASSET_TYPES:
                object_directory = "cover-card"
            else:
                object_directory = "cards"
            blob = client.bucket(get_bucket_name()).blob(
                f"{object_prefix}/{row.id}/{object_directory}/{asset_key}"
            )
        if not blob.exists(client):
            raise HTTPException(status_code=404, detail="knowledge asset not found")
        content = blob.download_as_bytes(timeout=120, retry=DOWNLOAD_RETRY)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail="knowledge storage is unavailable") from exc
    suffix = asset_key.rsplit(".", 1)[-1]
    media_type = {"jpg": "image/jpeg", "png": "image/png", "webp": "image/webp"}[suffix]
    return Response(
        content=content,
        media_type=media_type,
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )
