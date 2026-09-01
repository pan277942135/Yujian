from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .confusion_analyzer import _first, _text

DEFAULT_TARGETS: dict[str, int] = {
    "grass_carp": 300,
    "common_carp": 300,
    "black_carp": 200,
    "mandarin_fish": 250,
}
DEFAULT_REQUIRED_SCENES = ("hand_hold", "river", "fish_basket")
_SPECIES_KEYS = ("species_key", "true_species", "claimed_species", "species", "label", "class_name")
_SCENE_KEYS = ("scene", "scene_type", "background", "environment")
_ANGLE_KEYS = ("angle", "view_angle", "camera_angle")
_QUALITY_KEYS = ("image_quality", "quality", "quality_bucket")


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(source: Any) -> Any:
    if isinstance(source, (str, Path)):
        path = Path(source)
        with path.open("r", encoding="utf-8-sig") as handle:
            if path.suffix.lower() == ".json":
                return json.load(handle)
            # CSV support stays dependency-free and keeps this helper useful for
            # Dataset/manifest exports.
            import csv

            return list(csv.DictReader(handle))
    return source


def load_target_config(source: Any = None) -> dict[str, Any]:
    """Load target counts and optional scene requirements.

    Accepted documents are either a direct ``{species_key: count}`` mapping or
    an object with ``targets``/``species_targets`` plus optional
    ``required_scenes`` and ``scene_requirements`` keys.
    """

    if source is None:
        for candidate in (Path("species_target_config.json"), Path("config/species_target_config.json")):
            if candidate.exists():
                source = candidate
                break
    document = _load(source) if source is not None else {}
    if not isinstance(document, Mapping):
        document = {}
    raw_targets = document.get("targets", document.get("species_targets"))
    if raw_targets is None:
        raw_targets = {
            key: value
            for key, value in document.items()
            if key not in {"required_scenes", "scene_requirements", "scenes", "metadata"}
            and isinstance(value, (int, float, str))
        }
    targets: dict[str, int] = {}
    for key, value in (raw_targets or {}).items():
        try:
            targets[str(key).strip()] = max(0, int(value))
        except (TypeError, ValueError):
            continue
    if not targets:
        targets = dict(DEFAULT_TARGETS)
    required = document.get("required_scenes", document.get("scenes")) or []
    if isinstance(required, str):
        required = [required]
    required_scenes = [str(value).strip() for value in required if str(value).strip()]
    scene_requirements = document.get("scene_requirements") or {}
    if not isinstance(scene_requirements, Mapping):
        scene_requirements = {}
    normalized_scene_requirements: dict[str, list[str]] = {}
    for species, scenes in scene_requirements.items():
        if isinstance(scenes, str):
            scenes = [scenes]
        normalized_scene_requirements[str(species)] = [str(value).strip() for value in scenes or [] if str(value).strip()]
    return {
        "targets": targets,
        "required_scenes": required_scenes,
        "scene_requirements": normalized_scene_requirements,
    }


def _rows_and_counts(dataset: Any) -> tuple[list[Mapping[str, Any]], Counter[str]]:
    dataset = _load(dataset)
    if isinstance(dataset, list):
        return [row for row in dataset if isinstance(row, Mapping)], Counter()
    if not isinstance(dataset, Mapping):
        return [], Counter()
    rows: list[Mapping[str, Any]] = []
    for key in ("rows", "items", "images", "manifest", "samples", "dataset_items"):
        candidate = dataset.get(key)
        if isinstance(candidate, list):
            rows = [row for row in candidate if isinstance(row, Mapping)]
            if rows:
                break
    raw_counts = dataset.get("species_counts", dataset.get("counts", {}))
    if not raw_counts and not rows:
        # A compact metadata export may be a direct ``{species_key: count}``
        # object rather than wrapping counts under ``species_counts``.
        raw_counts = {
            key: value
            for key, value in dataset.items()
            if isinstance(value, (int, float, str))
        }
    counts: Counter[str] = Counter()
    if isinstance(raw_counts, Mapping):
        for key, value in raw_counts.items():
            try:
                counts[str(key).strip()] = max(0, int(value))
            except (TypeError, ValueError):
                continue
    return rows, counts


def _species(row: Mapping[str, Any]) -> str:
    value = _first(row, _SPECIES_KEYS)
    if isinstance(value, Mapping):
        value = _first(value, ("species_key", "name", "label", "common_name_en", "common_name_zh"))
    return _text(value)


def _values(row: Mapping[str, Any], keys: Iterable[str]) -> set[str]:
    value = _first(row, keys)
    if value in (None, ""):
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    return {str(value).strip()}


def analyze_data_gaps(
    dataset_metadata: Any,
    target_config: Any = None,
    *,
    required_scenes: Iterable[str] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Compare current Dataset/manifest coverage with collection targets."""

    config = load_target_config(target_config)
    targets = config["targets"]
    rows, provided_counts = _rows_and_counts(dataset_metadata)
    counts: Counter[str] = Counter(provided_counts)
    species_scene_values: dict[str, set[str]] = defaultdict(set)
    species_angle_values: dict[str, set[str]] = defaultdict(set)
    species_quality_values: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        species = _species(row)
        if not species:
            continue
        counts[species] += 1
        species_scene_values[species].update(_values(row, _SCENE_KEYS))
        species_angle_values[species].update(_values(row, _ANGLE_KEYS))
        species_quality_values[species].update(_values(row, _QUALITY_KEYS))

    species_gaps: list[dict[str, Any]] = []
    for species, target in targets.items():
        current = int(counts.get(species, 0))
        species_gaps.append({
            "species": species,
            "current": current,
            "target": int(target),
            "gap": max(0, int(target) - current),
        })

    global_required = list(required_scenes) if required_scenes is not None else config.get("required_scenes") or list(DEFAULT_REQUIRED_SCENES)
    global_required = [str(value).strip() for value in global_required if str(value).strip()]
    scene_gaps: list[dict[str, Any]] = []
    scene_requirements = config.get("scene_requirements") or {}
    for species in targets:
        required = scene_requirements.get(species) or global_required
        required = [str(value).strip() for value in required if str(value).strip()]
        observed = sorted(species_scene_values.get(species, set()))
        missing = [scene for scene in required if scene not in observed]
        scene_gaps.append({
            "species": species,
            "required_scenes": required,
            "observed_scenes": observed,
            "missing_scenes": missing,
            # ``missing`` is a short compatibility alias for UI/CLI consumers.
            "missing": missing,
            "angles": sorted(species_angle_values.get(species, set())),
            "image_quality": sorted(species_quality_values.get(species, set())),
        })

    dimension_gaps = {
        "scene": {row["species"]: row["missing_scenes"] for row in scene_gaps},
        # Angle and quality are reported as observed dimensions.  A caller can
        # provide per-species requirements through the same scene_requirements
        # structure without changing the manifest contract.
        "angle": {species: sorted(values) for species, values in species_angle_values.items()},
        "image_quality": {species: sorted(values) for species, values in species_quality_values.items()},
    }

    return {
        "generated_at": generated_at or utcnow_iso(),
        "current_counts": dict(counts),
        "targets": dict(targets),
        "species_gaps": species_gaps,
        "scene_gaps": scene_gaps,
        "dimension_gaps": dimension_gaps,
        "required_scenes": global_required,
        "recommended_scenes": sorted({scene for row in scene_gaps for scene in row["missing_scenes"]}),
        "row_count": len(rows),
    }


def build_data_gap_report(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return analyze_data_gaps(*args, **kwargs)


def analyze_gaps(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return analyze_data_gaps(*args, **kwargs)


class DataGapAnalyzer:
    def __init__(self, target_config: Any = None):
        self.target_config = target_config

    def analyze(self, dataset_metadata: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("target_config", self.target_config)
        return analyze_data_gaps(dataset_metadata, **kwargs)

    __call__ = analyze


__all__ = [
    "DEFAULT_REQUIRED_SCENES",
    "DEFAULT_TARGETS",
    "DataGapAnalyzer",
    "analyze_data_gaps",
    "analyze_gaps",
    "build_data_gap_report",
    "load_target_config",
]
