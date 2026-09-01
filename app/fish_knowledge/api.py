from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db import get_db
from app.fish_knowledge.fishing import FishFishing
from app.fish_knowledge.gallery import FishGalleryImage
from app.fish_knowledge.profile import FishProfile
from app.fish_knowledge.similarity import FishSimilarity
from app.fish_knowledge.species import FishSpecies
from app.fish_knowledge.video import FishVideo
from app.models import SpeciesCatalog


router = APIRouter(prefix="/api/v1/fish", tags=["fish-knowledge"])


class SpeciesListItem(BaseModel):
    id: str
    name_cn: str
    cover_image: str | None
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


class SpeciesDetailOut(BaseModel):
    species: SpeciesOut
    gallery: GalleryOut
    profile: ProfileOut
    fishing: FishingOut
    videos: list[VideoOut]
    similarity: list[SimilarityOut]


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


def _cover_image(species: FishSpecies) -> str | None:
    return species.gallery[0].url if species.gallery else None


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
        .options(selectinload(FishSpecies.gallery))
        .order_by(SpeciesCatalog.catalog_order, FishSpecies.id)
    )


@router.get("/species", response_model=list[SpeciesListItem])
def list_fish_species(db: Session = Depends(get_db)) -> list[SpeciesListItem]:
    rows = db.scalars(_active_species_query()).all()
    return [
        SpeciesListItem(
            id=row.id,
            name_cn=row.name_cn,
            cover_image=_cover_image(row),
            summary=row.summary,
        )
        for row in rows
    ]


@router.get("/species/{species_id}", response_model=SpeciesDetailOut)
def get_fish_species(species_id: str, db: Session = Depends(get_db)) -> SpeciesDetailOut:
    row = db.scalar(
        select(FishSpecies)
        .where(FishSpecies.id == species_id, FishSpecies.status == "ACTIVE")
        .options(
            selectinload(FishSpecies.gallery),
            selectinload(FishSpecies.profile),
            selectinload(FishSpecies.fishing),
            selectinload(FishSpecies.videos),
            selectinload(FishSpecies.similarities).selectinload(FishSimilarity.similar_species),
        )
    )
    if row is None:
        raise HTTPException(status_code=404, detail="fish species not found")

    gallery = [_gallery_item(item) for item in row.gallery[:5]]
    similarity = [
        SimilarityOut(
            species_id=item.species_id,
            similar_species_id=item.similar_species_id,
            similar_species_name_cn=item.similar_species.name_cn,
            difference=item.difference,
        )
        for item in row.similarities
        if item.similar_species is not None and item.similar_species.status == "ACTIVE"
    ]
    return SpeciesDetailOut(
        species=_species(row),
        gallery=GalleryOut(species_id=row.id, images=gallery),
        profile=_profile(row.id, row.profile),
        fishing=_fishing(row.id, row.fishing),
        videos=[_video(item) for item in row.videos],
        similarity=similarity,
    )
