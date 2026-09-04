from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .artifact_schema import ConfusionMatrixArtifact


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        for key in ("species_key", "common_name_en", "common_name_zh", "name", "label", "id"):
            if value.get(key) not in (None, ""):
                return str(value[key]).strip()
        return ""
    return str(value).strip()


def _labels_from_class_map(class_map: Any) -> list[str]:
    if class_map is None:
        return []
    if isinstance(class_map, Mapping):
        if isinstance(class_map.get("labels"), (list, tuple)):
            return [_text(value) for value in class_map["labels"]]
        class_map = class_map.get("classes", class_map)
    if isinstance(class_map, Mapping):
        indexed: list[tuple[int, str]] = []
        for raw_index, value in class_map.items():
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            label = _text(value)
            if label:
                indexed.append((index, label))
        return [label for _index, label in sorted(indexed)]
    if isinstance(class_map, (list, tuple)):
        indexed: list[tuple[int, str]] = []
        for fallback, row in enumerate(class_map):
            if isinstance(row, Mapping):
                raw_index = row.get("class_index", row.get("index", fallback))
                try:
                    index = int(raw_index)
                except (TypeError, ValueError):
                    index = fallback
                label = _text(row)
            else:
                index = fallback
                label = _text(row)
            if label:
                indexed.append((index, label))
        return [label for _index, label in sorted(indexed)]
    return []


def _prediction_value(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _normalise_matrix(matrix: Any, size: int) -> list[list[int]]:
    rows = matrix if isinstance(matrix, (list, tuple)) else []
    result: list[list[int]] = []
    for row in rows:
        values: list[int] = []
        if isinstance(row, (list, tuple)):
            for value in row[:size]:
                try:
                    values.append(max(0, int(value or 0)))
                except (TypeError, ValueError):
                    values.append(0)
        values.extend([0] * (size - len(values)))
        result.append(values)
    result.extend([[0] * size for _ in range(size - len(result))])
    return result


def _matrix_from_predictions(rows: Iterable[Mapping[str, Any]], labels: list[str]) -> list[list[int]]:
    indexes = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for row in rows:
        truth = _text(_prediction_value(row, "true_species", "true", "ground_truth", "claimed_species"))
        pred = _text(_prediction_value(row, "pred_species", "prediction", "predicted_species", "pred"))
        if truth in indexes and pred in indexes:
            matrix[indexes[truth]][indexes[pred]] += 1
    return matrix


def build_confusion_matrix_artifact(
    evaluation_report: Mapping[str, Any] | None = None,
    *,
    labels: Iterable[Any] | None = None,
    matrix: Any = None,
    predictions: Iterable[Mapping[str, Any]] | None = None,
    class_map: Any = None,
    model_version: str = "",
    dataset_version: str = "",
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Normalize trainer output into the v1 ``labels`` + ``matrix`` contract."""

    report = evaluation_report if isinstance(evaluation_report, Mapping) else {}
    if labels is None:
        labels = report.get("labels")
    resolved_labels = [_text(value) for value in (labels or []) if _text(value)]
    if not resolved_labels:
        resolved_labels = _labels_from_class_map(class_map)
    if not resolved_labels and isinstance(report.get("classes"), (list, tuple)):
        resolved_labels = _labels_from_class_map(report.get("classes"))

    if matrix is None:
        matrix = report.get("confusion_matrix")
        test = report.get("test")
        if matrix is None and isinstance(test, Mapping):
            matrix = test.get("confusion_matrix")
        nested = report.get("metrics")
        if matrix is None and isinstance(nested, Mapping):
            matrix = nested.get("confusion_matrix")

    prediction_rows = list(predictions or [])
    if not resolved_labels:
        # Preserve first-seen order, which is deterministic for a test split.
        discovered = OrderedDict()
        for row in prediction_rows:
            for key in ("true_species", "true", "ground_truth", "pred_species", "prediction", "predicted_species", "pred"):
                value = _text(row.get(key))
                if value:
                    discovered.setdefault(value, None)
        resolved_labels = list(discovered)

    size = max(len(resolved_labels), len(matrix or []) if isinstance(matrix, (list, tuple)) else 0)
    if not resolved_labels:
        resolved_labels = [f"class_{index}" for index in range(size)]
    if size > len(resolved_labels):
        resolved_labels.extend(f"class_{index}" for index in range(len(resolved_labels), size))
    normalized = _normalise_matrix(matrix, len(resolved_labels))
    if not any(any(row) for row in normalized) and prediction_rows:
        normalized = _matrix_from_predictions(prediction_rows, resolved_labels)
    artifact = ConfusionMatrixArtifact(
        labels=resolved_labels,
        matrix=normalized,
        model_version=model_version or _text(report.get("model_version")),
        dataset_version=dataset_version or _text(report.get("dataset_version")),
        generated_at=generated_at or "",
    )
    return artifact.to_dict()


def write_confusion_matrix(artifact: Mapping[str, Any], output_path: str | Path) -> dict[str, Any]:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(artifact), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return dict(artifact)


def export_confusion_matrix(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return build_confusion_matrix_artifact(*args, **kwargs)


__all__ = [
    "build_confusion_matrix_artifact",
    "export_confusion_matrix",
    "write_confusion_matrix",
]
