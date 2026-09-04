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
    return [row for row in rows if isinstance(row, Mapping) and row.get("true_species") and row.get("pred_species")]


def _gap_rows(gaps: Any) -> dict[str, Mapping[str, Any]]:
    gaps = _load(gaps)
    if not isinstance(gaps, Mapping):
        return {}
    rows = gaps.get("species_gaps") or gaps.get("gaps") or []
    result = {}
    for row in rows:
        if isinstance(row, Mapping) and row.get("species"):
            result[str(row["species"])] = row
    return result


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
    """Turn prioritized model errors and data gaps into an operator task.

    The result is a proposal only.  It contains a suggested Batch ID/source,
    but no Batch row is created and no data is auto-added to a Dataset.
    """

    report = _load(confusion_report)
    gaps = _load(data_gap_report)
    pairs = _pairs(report)
    gap_rows = _gap_rows(gaps)
    config = load_target_config(target_config)
    targets = config["targets"]
    if model_version is None and isinstance(report, Mapping):
        model_version = str(report.get("model_version") or "unknown").strip() or "unknown"
    model_version = model_version or "unknown"
    stamp = generated_at or utcnow().isoformat()
    date_part = stamp[:10].replace("-", "") if len(stamp) >= 10 else utcnow().strftime("%Y%m%d")
    task_id = task_id or f"TASK_{date_part}_{int(sequence):03d}"

    # P0/P1 pairs drive the first task; if an evaluation only contains P2
    # errors, retain the highest-ranked pair rather than silently dropping it.
    prioritized = [row for row in pairs if str(row.get("priority") or "P2").upper() in {"P0", "P1"}]
    reasons_source = prioritized or pairs
    reasons = [
        {
            "true": str(row.get("true_species")),
            "pred": str(row.get("pred_species")),
            "errors": int(row.get("error_count", row.get("errors", 0)) or 0),
            "error_rate": float(row.get("error_rate", 0) or 0),
            "priority": str(row.get("priority") or "P2"),
        }
        for row in reasons_source[:10]
    ]

    species: list[str] = []
    for row in reasons_source:
        for key in ("true_species", "pred_species"):
            name = str(row.get(key) or "").strip()
            if name and name not in species:
                species.append(name)
    if not species:
        for row in (gaps.get("species_gaps", []) if isinstance(gaps, Mapping) else []):
            name = str(row.get("species") or "").strip() if isinstance(row, Mapping) else ""
            if name and name not in species:
                species.append(name)

    requirements: list[dict[str, Any]] = []
    for name in species:
        row = gap_rows.get(name, {})
        target = int(row.get("target", targets.get(name, 300)) or 0)
        gap = int(row.get("gap", max(0, target - int(row.get("current", 0) or 0))) or 0)
        requirements.append({
            "name": name,
            "count": target if target > 0 else max(gap, 1),
            "target": target,
            "gap": gap,
        })

    if scenes is not None:
        scene_values = [str(value).strip() for value in scenes if str(value).strip()]
    elif isinstance(gaps, Mapping) and gaps.get("recommended_scenes"):
        scene_values = [str(value).strip() for value in gaps["recommended_scenes"] if str(value).strip()]
    else:
        scene_values = list(config.get("required_scenes") or DEFAULT_REQUIRED_SCENES)

    task_type = "HARD_CASE_COLLECTION" if reasons else "DATA_GAP_COLLECTION"
    difficulty = "hard" if reasons else "coverage"
    batch_id = f"BATCH_HARDCASE_{date_part}_{int(sequence):03d}"
    return {
        "task_id": task_id,
        "task_type": task_type,
        "model_version": model_version,
        "generated_at": stamp,
        "status": "OPEN",
        "source": "MODEL_ERROR_DRIVEN",
        "reason": reasons,
        "requirements": {
            "species": requirements,
            "scenes": scene_values,
            "difficulty": difficulty,
        },
        "batch_suggestion": {
            "batch_id": batch_id,
            "source": "MODEL_ERROR_DRIVEN",
        },
        "safety": {
            "creates_batch": False,
            "modifies_labels": False,
            "auto_freezes_dataset": False,
            "auto_starts_training": False,
        },
    }


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
    "generate_task",
    "write_collection_task",
]
