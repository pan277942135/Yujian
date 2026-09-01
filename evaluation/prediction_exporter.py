from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .artifact_schema import ERROR_SAMPLE_FIELDS, PREDICTION_FIELDS


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        for key in ("species_key", "common_name_en", "common_name_zh", "name", "label", "id"):
            if value.get(key) not in (None, ""):
                return str(value[key]).strip()
        return ""
    return str(value).strip()


def _first(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _class_labels(class_map: Any) -> dict[int, str]:
    if class_map is None:
        return {}
    if isinstance(class_map, Mapping):
        class_map = class_map.get("classes", class_map.get("labels", class_map))
    result: dict[int, str] = {}
    if isinstance(class_map, Mapping):
        for raw_index, value in class_map.items():
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if (label := _text(value)):
                result[index] = label
    elif isinstance(class_map, (list, tuple)):
        for fallback, value in enumerate(class_map):
            if isinstance(value, Mapping):
                raw_index = value.get("class_index", value.get("index", fallback))
                try:
                    index = int(raw_index)
                except (TypeError, ValueError):
                    index = fallback
                label = _text(value)
            else:
                index, label = fallback, _text(value)
            if label:
                result[index] = label
    return result


def _confidence(value: Any) -> float | str:
    if value in (None, ""):
        return ""
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return str(value).strip()


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "correct", "ok"}:
        return True
    if value in {"false", "0", "no", "n", "wrong", "error"}:
        return False
    return None


def normalize_prediction_rows(
    predictions: Iterable[Mapping[str, Any]] | None,
    *,
    model_version: str,
    dataset_version: str,
    class_map: Any = None,
) -> list[dict[str, Any]]:
    """Normalize test predictions and preserve optional provenance fields."""

    labels = _class_labels(class_map)
    result: list[dict[str, Any]] = []
    for index, source in enumerate(predictions or []):
        if not isinstance(source, Mapping):
            continue
        truth = _text(_first(source, "true_species", "true", "ground_truth", "claimed_species", "label"))
        pred = _text(_first(source, "pred_species", "prediction", "predicted_species", "pred"))
        true_index = _first(source, "true_class_index", "true_index", "label_index", "target", "class_index")
        pred_index = _first(source, "pred_class_index", "pred_index", "prediction_index")
        if not truth and true_index not in (None, ""):
            try:
                truth = labels.get(int(true_index), f"class_{int(true_index)}")
            except (TypeError, ValueError):
                truth = _text(true_index)
        if not pred and pred_index not in (None, ""):
            try:
                pred = labels.get(int(pred_index), f"class_{int(pred_index)}")
            except (TypeError, ValueError):
                pred = _text(pred_index)
        correct = _bool(_first(source, "correct", "is_correct"))
        if correct is None:
            correct = bool(truth and pred and truth == pred)
        row: dict[str, Any] = {
            "image_id": _text(_first(source, "image_id", "id", "asset_id")) or f"sample_{index + 1:06d}",
            "true_species": truth,
            "pred_species": pred,
            "confidence": _confidence(_first(source, "confidence", "predicted_confidence", "prediction_confidence", "score")),
            "correct": bool(correct),
            "model_version": model_version,
            "dataset_version": dataset_version,
        }
        # Keep paths and capture context for hard-case extraction without
        # changing the fixed CSV columns.
        for key in ("file_name", "filename", "local_path", "source_uri", "gcs_uri", "image_uri", "scene", "angle", "image_quality", "error_group"):
            if source.get(key) not in (None, ""):
                row[key] = source[key]
        result.append(row)
    return result


def build_error_samples(
    prediction_rows: Iterable[Mapping[str, Any]],
    *,
    model_version: str,
    dataset_version: str = "",
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for row in prediction_rows:
        if not isinstance(row, Mapping):
            continue
        if bool(_bool(row.get("correct")) if row.get("correct") is not None else row.get("true_species") != row.get("pred_species")):
            continue
        truth = _text(row.get("true_species"))
        pred = _text(row.get("pred_species"))
        if not truth or not pred or truth == pred:
            continue
        error_group = _text(row.get("error_group")) or f"{truth}_vs_{pred}"
        error = {
            "image_id": _text(row.get("image_id")),
            "true_species": truth,
            "pred_species": pred,
            "confidence": row.get("confidence", ""),
            "error_group": error_group,
            "model_version": model_version,
        }
        for key in ("file_name", "local_path", "source_uri", "gcs_uri", "scene", "angle", "image_quality"):
            if row.get(key) not in (None, ""):
                error[key] = row[key]
        if dataset_version:
            error["dataset_version"] = dataset_version
        errors.append(error)
    return errors


def write_predictions_csv(rows: Iterable[Mapping[str, Any]], output_path: str | Path) -> int:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PREDICTION_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in PREDICTION_FIELDS})
            count += 1
    return count


def write_error_samples(rows: Iterable[Mapping[str, Any]], output_path: str | Path) -> int:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    values = list(rows)
    destination.write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(values)


__all__ = [
    "ERROR_SAMPLE_FIELDS",
    "PREDICTION_FIELDS",
    "build_error_samples",
    "normalize_prediction_rows",
    "write_error_samples",
    "write_predictions_csv",
]
