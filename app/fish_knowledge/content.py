from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CONTENT_SEED_PATH = Path(__file__).resolve().parents[2] / "data" / "fish_knowledge" / "fish_seed.json"
EXPECTED_INITIAL_SPECIES = frozenset(
    {
        "sharpbelly",
        "crucian_carp",
        "grass_carp",
        "common_carp",
        "snakehead",
        "topmouth_culter",
        "mandarin_fish",
        "black_carp",
        "tilapia",
        "largemouth_bass",
    }
)
EXPECTED_CARD_TYPES = frozenset({"HERO", "IDENTIFICATION", "ECOLOGY", "GEAR", "SKILL"})


def load_content_seed(path: str | Path | None = None) -> dict[str, Any]:
    seed_path = Path(path) if path is not None else DEFAULT_CONTENT_SEED_PATH
    with seed_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != "FISH_KNOWLEDGE_CONTENT_V2":
        raise ValueError("unsupported Fish Knowledge content seed schema")
    species = payload.get("species")
    if not isinstance(species, list) or not species:
        raise ValueError("Fish Knowledge content seed must contain species records")
    ids = {str(item.get("id", "")).strip() for item in species if isinstance(item, dict)}
    if ids != EXPECTED_INITIAL_SPECIES:
        raise ValueError(f"Fish Knowledge v2 seed must contain the initial 10 species: {sorted(ids)}")
    for item in species:
        cards = item.get("cards")
        card_types = {str(card.get("type", "")).strip().upper() for card in cards or []}
        if card_types != EXPECTED_CARD_TYPES or len(cards or []) != 5:
            raise ValueError(f"{item.get('id')} must contain exactly five Fish Knowledge cards")
        if any(card.get("image_url") is not None or card.get("status") != "DRAFT" for card in cards):
            raise ValueError(f"{item.get('id')} seed cards must use null image URLs and DRAFT status")
        identification = next(card for card in cards if str(card.get("type")).upper() == "IDENTIFICATION")
        if len(identification.get("features") or []) != 3 or len(identification.get("similar") or []) != 2:
            raise ValueError(f"{item.get('id')} identification card must contain 3 features and 2 similar fish")
    return payload


def parse_card_content(description: str | None) -> dict[str, Any]:
    value = (description or "").strip()
    if not value:
        return {}
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {"description": value}
    return payload if isinstance(payload, dict) else {"description": value}


def card_description(content: dict[str, Any]) -> str:
    return json.dumps(content, ensure_ascii=False, separators=(",", ":"))


def card_display_description(content: dict[str, Any], fallback: str = "") -> str:
    value = content.get("description")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback
