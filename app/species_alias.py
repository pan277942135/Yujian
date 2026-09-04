from __future__ import annotations

from app.species_policy import TARGET_SPECIES_PRESETS

# Broad / ambiguous collection search words are useful for discovery but must not
# be silently collapsed into one canonical training identity.
SEARCH_ONLY_ALIASES = {"鲢鱼", "鳊鱼"}


def _build_alias_map() -> dict[str, str]:
    result: dict[str, str] = {}
    for preset in TARGET_SPECIES_PRESETS:
        canonical = str(preset["common_name_zh"]).strip()
        result[canonical] = canonical
        for alias in preset.get("aliases") or []:
            name = str(alias).strip()
            if name and name not in SEARCH_ONLY_ALIASES:
                result[name] = canonical
    return result


ALIAS_TO_CANONICAL = _build_alias_map()
CANONICAL_NAMES = frozenset(str(x["common_name_zh"]).strip() for x in TARGET_SPECIES_PRESETS)


def normalize_species_name(value: str | None) -> str | None:
    """Return a safe canonical product label while preserving unknown text.

    This only normalizes naming. It never promotes a label into Ground Truth.
    Ambiguous search-only aliases intentionally remain unchanged.
    """
    raw = (value or "").strip()
    if not raw:
        return None
    return ALIAS_TO_CANONICAL.get(raw, raw)


def alias_resolution(value: str | None) -> dict:
    raw = (value or "").strip()
    canonical = normalize_species_name(raw)
    return {
        "raw": raw or None,
        "canonical": canonical,
        "normalized": bool(raw and canonical and raw != canonical),
        "known": bool(canonical in CANONICAL_NAMES),
        "search_only": raw in SEARCH_ONLY_ALIASES,
    }
