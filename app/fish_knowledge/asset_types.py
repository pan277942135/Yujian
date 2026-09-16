"""Canonical Fish Knowledge asset slots.

The registry is additive: COVER remains a legacy alias used by the v1
CMS/API, while new imports and clients use the explicit 3+5 slots.
"""
from __future__ import annotations

from app.platform.models import FishAsset

COVER_ASSET_TYPES = (
    "COVER_CARD",
    "COVER_CARD_TRANSPARENT_LEFT",
    "COVER_CARD_TRANSPARENT_RIGHT",
)
KNOWLEDGE_ASSET_TYPES = ("HERO", "IDENTIFICATION", "ECO", "GEAR", "SKILL")
ALL_ASSET_TYPES = COVER_ASSET_TYPES + KNOWLEDGE_ASSET_TYPES
LEGACY_ASSET_TYPES = ("COVER",) + KNOWLEDGE_ASSET_TYPES

ASSET_SLOT_KEYS = {
    "COVER": "cover_card",
    "COVER_CARD": "cover_card",
    "COVER_CARD_TRANSPARENT_LEFT": "transparent_left",
    "COVER_CARD_TRANSPARENT_RIGHT": "transparent_right",
    "HERO": "hero",
    "IDENTIFICATION": "identification",
    "ECO": "eco",
    "GEAR": "gear",
    "SKILL": "skill",
}
ASSET_DIRECTIONS = {
    "COVER": "NONE",
    "COVER_CARD": "NONE",
    "COVER_CARD_TRANSPARENT_LEFT": "LEFT",
    "COVER_CARD_TRANSPARENT_RIGHT": "RIGHT",
    "HERO": "NONE",
    "IDENTIFICATION": "NONE",
    "ECO": "NONE",
    "GEAR": "NONE",
    "SKILL": "NONE",
}


def normalize_asset_type(value: str | None, direction: str | None = None) -> str | None:
    """Normalize v1 names and Manifest V2 aliases to canonical slots."""

    raw = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    side = str(direction or "").strip().upper()
    if raw in {"COVER", "COVER_CARD"}:
        return raw
    if raw in {"COVER_CARD_TRANSPARENT_LEFT", "COVER_CARD_LEFT", "TRANSPARENT_LEFT", "TRANSPARENT_MAIN"}:
        return "COVER_CARD_TRANSPARENT_LEFT"
    if raw in {"COVER_CARD_TRANSPARENT_RIGHT", "COVER_CARD_RIGHT", "TRANSPARENT_RIGHT", "TRANSPARENT_ALT"}:
        return "COVER_CARD_TRANSPARENT_RIGHT"
    if raw in {"COVER_CARD_TRANSPARENT", "TRANSPARENT"}:
        if side in {"LEFT", "L"}:
            return "COVER_CARD_TRANSPARENT_LEFT"
        if side in {"RIGHT", "R"}:
            return "COVER_CARD_TRANSPARENT_RIGHT"
        return None
    if raw in KNOWLEDGE_ASSET_TYPES:
        return raw
    return None


def asset_direction(asset_type: str | None, direction: str | None = None) -> str:
    normalized = normalize_asset_type(asset_type, direction) or str(asset_type or "").strip().upper()
    if normalized.endswith("_LEFT"):
        return "LEFT"
    if normalized.endswith("_RIGHT"):
        return "RIGHT"
    return ASSET_DIRECTIONS.get(normalized, "NONE")


def asset_slot_key(asset_type: str | None, direction: str | None = None) -> str:
    normalized = normalize_asset_type(asset_type, direction)
    return ASSET_SLOT_KEYS.get(normalized or "", str(asset_type or "").strip().lower())


def fish_asset_slot_id(species_id: str, asset_type: str) -> str:
    normalized = normalize_asset_type(asset_type) or str(asset_type).strip().upper()
    return f"fish-knowledge:{species_id}:{normalized.lower()}"


def upsert_fish_asset_index(
    db,
    *,
    species_id: str,
    asset_type: str,
    url: str,
    object_name: str | None = None,
    direction: str | None = None,
    status: str = "DRAFT",
    version: str = "v1",
    source_batch_id: str | None = None,
) -> FishAsset:
    """Index one knowledge slot in the existing fish_asset table."""

    normalized = normalize_asset_type(asset_type, direction)
    if normalized is None:
        raise ValueError(f"unsupported fish knowledge asset type: {asset_type}")
    row = db.get(FishAsset, fish_asset_slot_id(species_id, normalized))
    if row is None:
        row = FishAsset(
            asset_id=fish_asset_slot_id(species_id, normalized),
            species=species_id,
        )
        db.add(row)
    row.species = species_id
    row.asset_type = normalized
    row.direction = asset_direction(normalized, direction)
    row.asset_uri = url
    row.asset_object_name = object_name
    row.status = status
    row.version = version
    if source_batch_id:
        row.source_batch_id = source_batch_id
    return row


__all__ = [
    "ALL_ASSET_TYPES",
    "ASSET_DIRECTIONS",
    "ASSET_SLOT_KEYS",
    "COVER_ASSET_TYPES",
    "KNOWLEDGE_ASSET_TYPES",
    "LEGACY_ASSET_TYPES",
    "asset_direction",
    "asset_slot_key",
    "fish_asset_slot_id",
    "normalize_asset_type",
    "upsert_fish_asset_index",
]
