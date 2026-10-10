from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse, Response
from google.cloud import storage
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db import get_db
from app.factory import DOWNLOAD_RETRY, get_bucket_name
from app.fish_knowledge.cards import CARD_TYPE_ORDER, FishCard, normalize_card_type
from app.fish_knowledge.cover import FishSpeciesCover
from app.fish_knowledge.content import card_display_description, parse_card_content
from app.fish_knowledge.fishing import FishFishing
from app.fish_knowledge.gallery import FishGalleryImage, managed_knowledge_asset_url
from app.fish_knowledge.profile import FishProfile
from app.fish_knowledge.similarity import FishSimilarity
from app.fish_knowledge.species import SPECIES_ID_ALIASES, FishSpecies
from app.fish_knowledge.video import FishVideo
from app.models import SpeciesCatalog


router = APIRouter(prefix="/api/v1/fish", tags=["fish-knowledge"])


class SpeciesListItem(BaseModel):
    id: str
    name_cn: str
    category: str
    cover_image: str | None
    cover_hero_image: str | None = None
    cover_hero_version_id: int | None = None
    cover_hero_status: str = "MISSING"
    summary: str


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
    asset_version_id: int | None = None
    publication_source: str = "LEGACY_CARD"


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
    cover_hero_image: str | None = None
    cover_hero_version_id: int | None = None
    cover_hero_status: str = "MISSING"
    cover_assets: dict[str, Any] = Field(default_factory=dict)
    knowledge_assets: dict[str, Any] = Field(default_factory=dict)
    publication_conflicts: list[dict[str, Any]] = Field(default_factory=list)


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


def _cover_image(species: FishSpecies) -> str | None:
    if species.cover is not None and species.cover.status == "ACTIVE" and species.cover.image_url.strip():
        return managed_knowledge_asset_url(species.id, "COVER", species.cover.image_url)
    return species.gallery[0].url if species.gallery else None


def _card(
    row: FishCard,
    *,
    image_url: str | None = None,
    publication_source: str = "LEGACY_CARD",
    asset_version_id: int | None = None,
) -> CardOut:
    card_type = normalize_card_type(row.card_type)
    content = parse_card_content(row.description)
    return CardOut(
        id=row.id,
        species_id=row.species_id,
        card_type=card_type,
        type=card_type,
        title=row.title,
        image_url=image_url or managed_knowledge_asset_url(row.species_id, card_type, row.image_url),
        description=card_display_description(content, row.description),
        content=content,
        sort_order=row.sort_order,
        status=row.status,
        asset_version_id=asset_version_id if asset_version_id is not None else row.asset_version_id,
        publication_source=publication_source,
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


def _published_role_projection(db: Session, species_id: str, role: str) -> dict[str, Any]:
    from app.fish_knowledge.import_batch import FishKnowledgeAssetVersion, _asset_role_for_version

    active = db.scalars(select(FishKnowledgeAssetVersion).where(
        FishKnowledgeAssetVersion.species_id == species_id,
        FishKnowledgeAssetVersion.status == "ACTIVE",
    )).all()
    active = [item for item in active if _asset_role_for_version(item) == role]
    if len(active) != 1:
        return {"asset_version_id": None, "asset_image_url": None, "card_id": None, "card_image_url": None}
    version = active[0]
    cards = db.scalars(select(FishCard).where(FishCard.asset_version_id == version.id)).all()
    if len(cards) != 1:
        return {"asset_version_id": version.id, "asset_image_url": version.image_url, "card_id": None, "card_image_url": None}
    card = cards[0]
    if card.status != "ACTIVE" or normalize_card_type(card.card_type) != role or card.image_url != version.image_url:
        return {"asset_version_id": version.id, "asset_image_url": version.image_url, "card_id": card.id, "card_image_url": card.image_url}
    return {
        "asset_version_id": version.id,
        "asset_image_url": version.image_url,
        "card_id": card.id,
        "card_image_url": card.image_url,
    }


def build_species_full_detail(row: FishSpecies, db: Session) -> SpeciesFullDetailOut:
    base = build_species_detail(row)
    from app.fish_knowledge.import_batch import FishKnowledgeAssetVersion, _asset_role_for_version

    active_versions = db.scalars(select(FishKnowledgeAssetVersion).where(
        FishKnowledgeAssetVersion.species_id == row.id,
        FishKnowledgeAssetVersion.status == "ACTIVE",
    )).all()
    versions_by_role: dict[str, list[FishKnowledgeAssetVersion]] = {}
    for version in active_versions:
        role = _asset_role_for_version(version)
        versions_by_role.setdefault(role, []).append(version)

    cards: list[CardOut] = []
    active_card_rows: list[FishCard] = []
    knowledge_assets: dict[str, Any] = {}
    conflicts: list[dict[str, Any]] = []
    for role in CARD_TYPE_ORDER:
        versions = versions_by_role.get(role, [])
        if len(versions) > 1:
            conflicts.append({"role": role, "code": "ACTIVE_VERSION_CONFLICT", "active_version_ids": [item.id for item in versions]})
            continue
        if versions:
            version = versions[0]
            bound = db.scalars(select(FishCard).where(FishCard.asset_version_id == version.id)).all()
            if len(bound) != 1:
                conflicts.append({"role": role, "code": "VERSION_BINDING_MISSING" if not bound else "DUPLICATE_VERSION_BINDING", "version_id": version.id})
                continue
            card = bound[0]
            if (
                card.status != "ACTIVE"
                or card.species_id != row.id
                or normalize_card_type(card.card_type) != role
                or card.image_url != version.image_url
            ):
                conflicts.append({"role": role, "code": "VERSION_BINDING_MISMATCH", "version_id": version.id, "card_id": card.id})
                continue
            cards.append(_card(card, image_url=version.image_url, publication_source="VERSIONED_ASSET", asset_version_id=version.id))
            active_card_rows.append(card)
            knowledge_assets[role] = {
                "asset_role": role,
                "image_url": version.image_url,
                "version": version.version,
                "version_id": version.id,
                "asset_status": "ACTIVE",
                "source_sha256": version.sha256,
                "publication_source": "VERSIONED_ASSET",
                "card_id": card.id,
            }
            continue

        # Compatibility is explicit and limited to unbound legacy cards when
        # no versioned publication exists for that role.
        legacy = [
            item for item in row.cards
            if item.status == "ACTIVE"
            and item.asset_version_id is None
            and normalize_card_type(item.card_type) == role
        ]
        if len(legacy) == 1:
            card = legacy[0]
            cards.append(_card(card, publication_source="LEGACY_CARD"))
            active_card_rows.append(card)
            knowledge_assets[role] = {
                "asset_role": role,
                "image_url": managed_knowledge_asset_url(row.id, role, card.image_url),
                "version": None,
                "version_id": None,
                "asset_status": "ACTIVE",
                "source_sha256": None,
                "publication_source": "LEGACY_CARD",
                "card_id": card.id,
            }
        elif len(legacy) > 1:
            conflicts.append({"role": role, "code": "LEGACY_ACTIVE_CARD_CONFLICT", "card_ids": [item.id for item in legacy]})
        else:
            conflicts.append({"role": role, "code": "ACTIVE_CARD_MISSING"})

    profile = base.profile
    fishing = base.fishing
    card_content = {
        normalize_card_type(item.card_type): parse_card_content(item.description)
        for item in active_card_rows
    }
    ecology = card_content.get("ECO", {})
    gear = card_content.get("GEAR", {})
    skill = card_content.get("SKILL", {})
    cover_assets: dict[str, Any] = {}
    for role in ("COVER_LIST", "COVER_HERO", "TRANSPARENT_MAIN", "TRANSPARENT_ALT"):
        versions = versions_by_role.get(role, [])
        if len(versions) > 1:
            conflicts.append({"role": role, "code": "ACTIVE_VERSION_CONFLICT", "active_version_ids": [item.id for item in versions]})
            continue
        if not versions:
            continue
        version = versions[0]
        if role == "COVER_LIST" and (row.cover is None or row.cover.status != "ACTIVE" or row.cover.image_url != version.image_url):
            conflicts.append({"role": role, "code": "COVER_BINDING_MISMATCH", "version_id": version.id})
            continue
        cover_assets[role] = {
            "asset_role": role,
            "image_url": version.image_url,
            "version": version.version,
            "version_id": version.id,
            "asset_status": "ACTIVE",
            "source_sha256": version.sha256,
            "publication_source": "VERSIONED_ASSET",
        }
    if "COVER_LIST" not in cover_assets and row.cover is not None and row.cover.status == "ACTIVE":
        cover_assets["COVER_LIST"] = {
            "asset_role": "COVER_LIST",
            "image_url": managed_knowledge_asset_url(row.id, "COVER", row.cover.image_url),
            "version": None,
            "version_id": None,
            "asset_status": "ACTIVE",
            "source_sha256": None,
            "publication_source": "LEGACY_COVER",
        }

    active_cover_hero = versions_by_role.get("COVER_HERO", [])
    cover_hero_status = "ACTIVE" if len(active_cover_hero) == 1 and "COVER_HERO" in cover_assets else (
        "CONFLICT" if len(active_cover_hero) > 1 else "MISSING"
    )
    cover_hero_version = active_cover_hero[0] if cover_hero_status == "ACTIVE" else None

    legacy_cover = _cover_dict(row.cover)
    if "COVER_LIST" in cover_assets:
        legacy_cover = {**legacy_cover, "image_url": cover_assets["COVER_LIST"]["image_url"]}

    return SpeciesFullDetailOut(
        species=base.species,
        cover=legacy_cover,
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
        cover_hero_image=cover_assets.get("COVER_HERO", {}).get("image_url"),
        cover_hero_version_id=cover_hero_version.id if cover_hero_version else None,
        cover_hero_status=cover_hero_status,
        publication_conflicts=conflicts,
    )


@router.get("/species", response_model=list[SpeciesListItem])
def list_fish_species(db: Session = Depends(get_db)) -> list[SpeciesListItem]:
    rows = db.scalars(_active_species_query()).all()
    from app.fish_knowledge.import_batch import FishKnowledgeAssetVersion, _asset_role_for_version

    species_ids = [row.id for row in rows]
    active_cover_hero: dict[str, list[FishKnowledgeAssetVersion]] = {species_id: [] for species_id in species_ids}
    versions = db.scalars(select(FishKnowledgeAssetVersion).where(
        FishKnowledgeAssetVersion.species_id.in_(species_ids or ["__none__"]),
        FishKnowledgeAssetVersion.status == "ACTIVE",
    )).all()
    for version in versions:
        if version.species_id in active_cover_hero and _asset_role_for_version(version) == "COVER_HERO":
            active_cover_hero[version.species_id].append(version)
    return [
        SpeciesListItem(
            id=row.id,
            name_cn=row.name_cn,
            category=row.category,
            cover_image=_cover_image(row),
            cover_hero_image=(active_cover_hero[row.id][0].image_url if len(active_cover_hero[row.id]) == 1 else None),
            cover_hero_version_id=(active_cover_hero[row.id][0].id if len(active_cover_hero[row.id]) == 1 else None),
            cover_hero_status=("ACTIVE" if len(active_cover_hero[row.id]) == 1 else "CONFLICT" if len(active_cover_hero[row.id]) > 1 else "MISSING"),
            summary=row.summary,
        )
        for row in rows
    ]


@router.get("/species/{species_id}/detail", response_model=SpeciesFullDetailOut)
def get_fish_species_full_detail(species_id: str, db: Session = Depends(get_db)) -> SpeciesFullDetailOut:
    row = load_species_with_knowledge(db, species_id, active_only=True)
    if row is None:
        raise HTTPException(status_code=404, detail="fish species not found")
    return build_species_full_detail(row, db)


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
    """Serve a managed cover/card image without adding a media table."""

    role_by_path = {
        "cover_list": "COVER_LIST", "cover_hero": "COVER_HERO",
        "transparent_main": "TRANSPARENT_MAIN", "transparent_alt": "TRANSPARENT_ALT",
    }
    path_key = asset_type.strip().lower()
    requested_role = role_by_path.get(path_key)
    normalized_type = "cover" if path_key == "cover" else (requested_role or normalize_card_type(asset_type))
    if normalized_type != "cover" and normalized_type not in {"COVER_LIST", "COVER_HERO", "TRANSPARENT_MAIN", "TRANSPARENT_ALT", "HERO", "IDENTIFICATION", "ECO", "GEAR", "SKILL"}:
        raise HTTPException(status_code=404, detail="knowledge asset not found")
    is_hashed_asset = bool(re.fullmatch(r"[a-f0-9]{64}\.(?:jpg|png|webp)", asset_key))
    is_version_asset = bool(re.fullmatch(r"v\d+\.webp", asset_key))
    fixed_asset_key = "cover.webp" if normalized_type == "cover" else f"{normalized_type.lower()}.webp"
    if not is_hashed_asset and not is_version_asset and asset_key != fixed_asset_key:
        raise HTTPException(status_code=404, detail="knowledge asset not found")
    row = load_species_with_knowledge(db, species_id, active_only=False)
    if row is None:
        raise HTTPException(status_code=404, detail="fish species not found")
    storage_type = "cover" if normalized_type == "cover" else normalized_type.lower()
    expected_url = f"/api/v1/fish/knowledge-media/{row.id}/{storage_type}/{asset_key}"
    version = None
    legacy_version_object_name = None
    if is_version_asset:
        from app.fish_knowledge.import_batch import FishKnowledgeAssetVersion, _asset_role_for_version

        candidates = db.scalars(select(FishKnowledgeAssetVersion).where(
            FishKnowledgeAssetVersion.species_id == row.id,
        )).all()
        expected_role = "COVER_LIST" if normalized_type == "cover" else normalized_type
        role_candidates = [candidate for candidate in candidates if _asset_role_for_version(candidate) == expected_role]
        exact_url_versions = [candidate for candidate in role_candidates if candidate.image_url == expected_url]
        active_role_versions = [candidate for candidate in role_candidates if candidate.status == "ACTIVE"]
        if len(active_role_versions) == 1 and active_role_versions[0].image_url == expected_url:
            version = active_role_versions[0]
        is_referenced = version is not None
        if (
            not active_role_versions
            and not exact_url_versions
            and normalized_type in {"HERO", "IDENTIFICATION", "ECO", "GEAR", "SKILL"}
        ):
            # Older public cards can already point at a version-shaped URL
            # before the exact asset-version binding was backfilled. Keep
            # those ACTIVE legacy rows readable, but only when no version row
            # (including a DRAFT) claims the same URL. This never promotes or
            # serves a DRAFT version.
            legacy_cards = [
                card for card in row.cards
                if card.status == "ACTIVE"
                and card.species_id == row.id
                and card.asset_version_id is None
                and normalize_card_type(card.card_type) == expected_role
                and card.image_url == expected_url
            ]
            if len(legacy_cards) == 1:
                is_referenced = True
                legacy_version_object_name = (
                    f"fish-assets/fish-knowledge/{row.id}/{expected_role.lower()}/{asset_key}"
                )
    elif normalized_type == "cover":
        is_referenced = (
            row.cover is not None
            and managed_knowledge_asset_url(row.id, "COVER", row.cover.image_url) == expected_url
        )
    elif normalized_type in {"HERO", "IDENTIFICATION", "ECO", "GEAR", "SKILL"}:
        is_referenced = any(
            normalize_card_type(card.card_type) == normalized_type
            and managed_knowledge_asset_url(row.id, normalized_type, card.image_url) == expected_url
            for card in row.cards
        )
    else:
        is_referenced = False
    if not is_referenced:
        raise HTTPException(status_code=404, detail="knowledge asset not found")

    try:
        client = storage.Client()
        if version is not None:
            blob = client.bucket(get_bucket_name()).blob(version.object_name)
        elif legacy_version_object_name is not None:
            blob = client.bucket(get_bucket_name()).blob(legacy_version_object_name)
        else:
            object_prefix = "fish_knowledge" if is_hashed_asset else "fish-assets"
            object_directory = storage_type if is_hashed_asset else ("cover" if storage_type == "cover" else "cards")
            blob = client.bucket(get_bucket_name()).blob(f"{object_prefix}/{row.id}/{object_directory}/{asset_key}")
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
        # Cover/Card slots are replaceable. Do not let a browser keep an old
        # image after an operator uploads a replacement to the same slot URL.
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )
