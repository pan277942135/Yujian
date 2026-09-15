from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .data_gap_analyzer import DEFAULT_REQUIRED_SCENES, load_target_config


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _load(source: Any) -> Any:
    if isinstance(source, (str, Path)):
        with Path(source).open("r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    return source


def _pairs(report: Any) -> list[Mapping[str, Any]]:
    report = _load(report)
    if not isinstance(report, Mapping):
        return []
    rows = report.get("top_confusions") or report.get("confusions") or []
    return [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("true_species") and row.get("pred_species")
    ]


def _gap_rows(gaps: Any, key: str = "species_gaps") -> dict[str, Mapping[str, Any]]:
    gaps = _load(gaps)
    if not isinstance(gaps, Mapping):
        return {}
    rows = gaps.get(key) or gaps.get("species_gaps") or gaps.get("gaps") or []
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if isinstance(row, Mapping) and row.get("species"):
            result[str(row["species"])] = row
    return result


def _date_part(stamp: str) -> str:
    return stamp[:10].replace("-", "") if len(stamp) >= 10 else utcnow().strftime("%Y%m%d")


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return default


def _priority(row: Mapping[str, Any]) -> str:
    return str(row.get("priority") or "P2").strip().upper()


def _hard_case_count(priority: str, error_count: int) -> int:
    """Return an intentionally bounded operator target for a hard case."""

    if priority == "P0":
        return min(150, max(100, error_count * 20))
    if priority == "P1":
        return min(80, max(50, error_count * 15))
    return 0


def _species_scenes(
    species: str,
    gap_rows: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    explicit: Iterable[str] | None = None,
) -> list[str]:
    if explicit is not None:
        return [str(value).strip() for value in explicit if str(value).strip()]
    row = gap_rows.get(species, {})
    missing = row.get("missing_scenes") or row.get("missing") or []
    values = [str(value).strip() for value in missing if str(value).strip()]
    configured = (config.get("scene_requirements") or {}).get(species, [])
    return values or list(configured or config.get("required_scenes") or DEFAULT_REQUIRED_SCENES)


def _reason_row(pair: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "true": str(pair.get("true_species")),
        "pred": str(pair.get("pred_species")),
        "errors": _as_int(pair.get("error_count", pair.get("errors"))),
        "error_rate": _as_float(pair.get("error_rate")),
        "priority": _priority(pair),
        "test_support": _as_int(pair.get("test_support")),
        "priority_score": _as_float(pair.get("priority_score")),
    }


def _hard_case_task(
    pair: Mapping[str, Any],
    *,
    model_version: str,
    gap_rows: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    task_id: str,
    stamp: str,
    scenes: Iterable[str] | None = None,
) -> dict[str, Any]:
    priority = _priority(pair)
    true_species = str(pair.get("true_species") or "").strip()
    pred_species = str(pair.get("pred_species") or "").strip()
    count = _hard_case_count(priority, _as_int(pair.get("error_count", pair.get("errors"))))
    scene_values = _species_scenes(true_species, gap_rows, config, scenes)
    reason = [_reason_row(pair)]
    metadata = {
        "task_type": "HARD_CASE_COLLECTION",
        "model_version": model_version,
        "reason": f"{true_species}_to_{pred_species}",
        "target_species": [true_species],
        "scene_requirement": scene_values,
    }
    return {
        "task_id": task_id,
        "type": "HARD_CASE_COLLECTION",
        "task_type": "HARD_CASE_COLLECTION",
        "model_version": model_version,
        "true_species": true_species,
        "confused_species": pred_species,
        "priority": priority,
        "error_count": _as_int(pair.get("error_count", pair.get("errors"))),
        "error_rate": _as_float(pair.get("error_rate")),
        "test_support": _as_int(pair.get("test_support")),
        "priority_score": _as_float(pair.get("priority_score")),
        "generated_at": stamp,
        "status": "OPEN",
        "source": "CONFUSION_MATRIX",
        "reason": reason,
        "requirements": {
            "species": [{"name": true_species, "count": count, "priority": priority}],
            "scenes": scene_values,
            "difficulty": "hard",
        },
        "batch_suggestion": {
            "batch_id": f"BATCH_HARDCASE_{_date_part(stamp)}_{task_id.rsplit('_', 1)[-1]}",
            "source": "MODEL_ERROR_DRIVEN",
            "batch_type": "HARD_CASE_COLLECTION",
            "metadata": metadata,
            "upload_url": "/platform/data/import",
        },
        "safety": {
            "creates_batch": False,
            "modifies_labels": False,
            "auto_freezes_dataset": False,
            "auto_starts_training": False,
        },
    }


def _quantity_task(
    row: Mapping[str, Any],
    *,
    model_version: str,
    task_id: str,
    stamp: str,
) -> dict[str, Any]:
    species = str(row.get("species") or "").strip()
    gap = max(0, _as_int(row.get("gap")))
    metadata = {
        "task_type": "DATA_BALANCE_COLLECTION",
        "model_version": model_version,
        "reason": "quantity_gap",
        "target_species": [species],
        "scene_requirement": [],
    }
    return {
        "task_id": task_id,
        "type": "DATA_BALANCE_COLLECTION",
        "task_type": "DATA_BALANCE_COLLECTION",
        "model_version": model_version,
        "generated_at": stamp,
        "status": "OPEN",
        "source": "SPECIES_TARGET_CONFIG",
        "reason": [{"type": "QUANTITY_GAP", "species": species, "current": _as_int(row.get("current")), "target": _as_int(row.get("target")), "gap": gap}],
        "requirements": {"species": [{"name": species, "count": gap}], "scenes": [], "difficulty": "coverage"},
        "batch_suggestion": {
            "batch_id": f"BATCH_BALANCE_{_date_part(stamp)}_{task_id.rsplit('_', 1)[-1]}",
            "source": "DATA_GAP",
            "batch_type": "DATA_BALANCE_COLLECTION",
            "metadata": metadata,
            "upload_url": "/platform/data/import",
        },
        "safety": {"creates_batch": False, "modifies_labels": False, "auto_freezes_dataset": False, "auto_starts_training": False},
    }


def _scene_task(
    row: Mapping[str, Any],
    *,
    model_version: str,
    task_id: str,
    stamp: str,
) -> dict[str, Any]:
    species = str(row.get("species") or "").strip()
    scenes = [str(value).strip() for value in (row.get("missing_scenes") or row.get("missing") or []) if str(value).strip()]
    metadata = {
        "task_type": "SCENE_GAP_COLLECTION",
        "model_version": model_version,
        "reason": "scene_gap",
        "target_species": [species],
        "scene_requirement": scenes,
    }
    return {
        "task_id": task_id,
        "type": "SCENE_GAP_COLLECTION",
        "task_type": "SCENE_GAP_COLLECTION",
        "model_version": model_version,
        "generated_at": stamp,
        "status": "OPEN",
        "source": "SCENE_REQUIREMENT_CONFIG",
        "reason": [{"type": "SCENE_GAP", "species": species, "missing_scenes": scenes}],
        "requirements": {"species": [{"name": species, "count": 0}], "scenes": scenes, "difficulty": "coverage"},
        "batch_suggestion": {
            "batch_id": f"BATCH_SCENE_{_date_part(stamp)}_{task_id.rsplit('_', 1)[-1]}",
            "source": "DATA_GAP",
            "batch_type": "SCENE_GAP_COLLECTION",
            "metadata": metadata,
            "upload_url": "/platform/data/import",
        },
        "safety": {"creates_batch": False, "modifies_labels": False, "auto_freezes_dataset": False, "auto_starts_training": False},
    }


def generate_collection_task(
    confusion_report: Any,
    data_gap_report: Any,
    *,
    task_id: str | None = None,
    model_version: str | None = None,
    target_config: Any = None,
    scenes: Iterable[str] | None = None,
    generated_at: str | None = None,
    sequence: int = 1,
) -> dict[str, Any]:
    """Generate the highest-value single operator task.

    Hard Case tasks are driven by one directed confusion pair and only collect
    the true species.  P2 pairs are intentionally not auto-promoted to a
    collection task; quantity/scene tasks are generated by
    :func:`generate_collection_tasks` as separate task types.
    """

    report = _load(confusion_report)
    gaps = _load(data_gap_report)
    config = load_target_config(target_config)
    pairs = _pairs(report)
    gap_rows = _gap_rows(gaps, "scene_gaps")
    fallback_scenes = (
        [str(value).strip() for value in gaps.get("recommended_scenes", []) if str(value).strip()]
        if isinstance(gaps, Mapping)
        else []
    )
    resolved_model = model_version or (str(report.get("model_version")) if isinstance(report, Mapping) else "unknown") or "unknown"
    stamp = generated_at or utcnow().isoformat()
    date = _date_part(stamp)
    default_id = task_id or f"TASK_{date}_{int(sequence):03d}"
    prioritized = [pair for pair in pairs if _priority(pair) in {"P0", "P1"}]
    if prioritized:
        return _hard_case_task(
            prioritized[0],
            model_version=resolved_model,
            gap_rows=gap_rows,
            config=config,
            task_id=default_id,
            stamp=stamp,
            scenes=scenes if scenes is not None else (fallback_scenes or None),
        )

    quantity_rows = _gap_rows(gaps, "quantity_gaps")
    quantity = next((row for row in quantity_rows.values() if _as_int(row.get("gap")) > 0), None)
    if quantity:
        return _quantity_task(quantity, model_version=resolved_model, task_id=default_id, stamp=stamp)

    return {
        "task_id": default_id,
        "task_type": "NO_ACTION",
        "model_version": resolved_model,
        "generated_at": stamp,
        "status": "NO_ACTION",
        "source": "EVALUATION_ARTIFACT",
        "reason": [],
        "requirements": {"species": [], "scenes": [], "difficulty": "none"},
        "batch_suggestion": {},
        "safety": {"creates_batch": False, "modifies_labels": False, "auto_freezes_dataset": False, "auto_starts_training": False},
    }


def generate_collection_tasks(
    confusion_report: Any,
    data_gap_report: Any,
    *,
    model_version: str | None = None,
    target_config: Any = None,
    generated_at: str | None = None,
) -> list[dict[str, Any]]:
    """Generate separate Hard Case, quantity-gap and scene-gap proposals."""

    report = _load(confusion_report)
    gaps = _load(data_gap_report)
    config = load_target_config(target_config)
    pairs = _pairs(report)
    gap_rows = _gap_rows(gaps, "scene_gaps")
    resolved_model = model_version or (str(report.get("model_version")) if isinstance(report, Mapping) else "unknown") or "unknown"
    stamp = generated_at or utcnow().isoformat()
    date = _date_part(stamp)
    tasks: list[dict[str, Any]] = []

    # Error-driven tasks always appear first.  P2 remains visible in the
    # confusion table but does not create an automatic collection task.
    hard_pairs = [pair for pair in pairs if _priority(pair) in {"P0", "P1"}]
    for index, pair in enumerate(hard_pairs, start=1):
        tasks.append(
            _hard_case_task(
                pair,
                model_version=resolved_model,
                gap_rows=gap_rows,
                config=config,
                task_id=f"TASK_{date}_HARDCASE_{index:03d}",
                stamp=stamp,
            )
        )

    quantity_rows = _gap_rows(gaps, "quantity_gaps")
    for index, row in enumerate(quantity_rows.values(), start=1):
        if _as_int(row.get("gap")) <= 0:
            continue
        tasks.append(
            _quantity_task(
                row,
                model_version=resolved_model,
                task_id=f"TASK_{date}_BALANCE_{index:03d}",
                stamp=stamp,
            )
        )

    scene_source = gaps.get("scene_gaps", []) if isinstance(gaps, Mapping) else []
    for index, row in enumerate(scene_source, start=1):
        if not isinstance(row, Mapping) or not (row.get("missing_scenes") or row.get("missing")):
            continue
        tasks.append(
            _scene_task(
                row,
                model_version=resolved_model,
                task_id=f"TASK_{date}_SCENE_{index:03d}",
                stamp=stamp,
            )
        )
    return tasks


def training_recommendations(tasks: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    for task in tasks:
        if str(task.get("task_type")) != "HARD_CASE_COLLECTION":
            continue
        reasons = task.get("reason") or []
        reason = reasons[0] if isinstance(reasons, list) and reasons and isinstance(reasons[0], Mapping) else {}
        species = [row.get("name") for row in (task.get("requirements", {}).get("species") or []) if row.get("name")]
        recommendations.append(
            {
                "type": "HARD_CASE",
                "title": f"增加 {', '.join(species)} 真实钓获照片",
                "count": sum(_as_int(row.get("count")) for row in (task.get("requirements", {}).get("species") or [])),
                "focus_scenes": list(task.get("requirements", {}).get("scenes") or []),
                "reason": f"{reason.get('true', '')} 易被误判为 {reason.get('pred', '')}".strip(),
            }
        )
    return recommendations


def write_collection_task(task: Mapping[str, Any], output_path: str | Path) -> dict[str, Any]:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(task), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return dict(task)


def build_collection_task(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return generate_collection_task(*args, **kwargs)


def generate_task(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return generate_collection_task(*args, **kwargs)


class CollectionTaskGenerator:
    def __init__(self, target_config: Any = None):
        self.target_config = target_config

    def generate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("target_config", self.target_config)
        return generate_collection_task(*args, **kwargs)

    __call__ = generate


__all__ = [
    "CollectionTaskGenerator",
    "build_collection_task",
    "generate_collection_task",
    "generate_collection_tasks",
    "generate_task",
    "training_recommendations",
    "write_collection_task",
]
