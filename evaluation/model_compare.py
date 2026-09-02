"""Compare two classifier evaluation artifact bundles.

The comparison is artifact-only and read-only: it never changes labels,
Dataset Freeze state, model registry state, or training inputs.  It is usable
both by the trainer after a successful run and by an operator CLI later.
"""

from __future__ import annotations

import json
import csv
import io
from pathlib import Path
from typing import Any, Iterable, Mapping


COMPARE_REPORT_TYPE = "MODEL_COMPARE_REPORT"
FOCUS_PAIRS = (
    ("crucian_carp", "common_carp"),
    ("silver_carp", "bighead_carp"),
    ("topmouth_culter", "chinese_hooksnout_carp"),
    ("yellowcheek", "topmouth_culter"),
)
METRIC_KEYS = ("top1_accuracy", "top3_accuracy", "macro_precision", "macro_recall", "macro_f1")


def _text(value: Any) -> str:
    return str(value or "").strip()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, Mapping):
            rows.append(dict(value))
    return rows


def _read_confusion_csv(path: Path) -> dict[str, Any]:
    rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8-sig"))))
    if not rows or len(rows[0]) < 2:
        return {}
    labels = [str(value).strip() for value in rows[0][1:]]
    matrix: list[list[int]] = []
    for row in rows[1:]:
        values: list[int] = []
        for value in row[1 : len(labels) + 1]:
            values.append(_safe_int(value, 0))
        values.extend([0] * max(0, len(labels) - len(values)))
        matrix.append(values)
    return {"labels": labels, "matrix": matrix}


def _load_bundle(source: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    if path.is_dir():
        result: dict[str, Any] = {}
        for name, key in (
            ("metrics.json", "metrics_document"),
            ("confusion_matrix.json", "confusion_matrix_document"),
            ("report.json", "report_document"),
            ("class_map.json", "class_map_document"),
        ):
            candidate = path / name
            if candidate.is_file():
                result[key] = _read_json(candidate)
        confusion_csv = path / "confusion_matrix.csv"
        if confusion_csv.is_file() and "confusion_matrix_document" not in result:
            result["confusion_matrix_document"] = _read_confusion_csv(confusion_csv)
        prediction_rows = path / "prediction_rows.jsonl"
        if prediction_rows.is_file():
            result["prediction_rows_document"] = _read_jsonl(prediction_rows)
        result["artifact_root"] = str(path)
        return result
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl":
        return {"prediction_rows_document": _read_jsonl(path), "artifact_root": str(path.parent)}
    document = _read_json(path)
    if not isinstance(document, Mapping):
        raise ValueError(f"evaluation artifact must be a JSON object: {path}")
    result = dict(document)
    # Accept a canonical artifact file path directly while loading siblings
    # when they are available.  The trainer and CLI commonly receive a
    # metrics.json URI rather than the bundle directory.
    if path.name in {"metrics.json", "confusion_matrix.json", "report.json"}:
        result = {f"{path.stem}_document": result, **result}
        for name, key in (
            ("metrics.json", "metrics_document"),
            ("confusion_matrix.json", "confusion_matrix_document"),
            ("report.json", "report_document"),
            ("class_map.json", "class_map_document"),
        ):
            sibling = path.parent / name
            if sibling.is_file() and key not in result:
                result[key] = _read_json(sibling)
        confusion_csv = path.parent / "confusion_matrix.csv"
        if confusion_csv.is_file() and "confusion_matrix_document" not in result:
            result["confusion_matrix_document"] = _read_confusion_csv(confusion_csv)
        prediction_rows = path.parent / "prediction_rows.jsonl"
        if prediction_rows.is_file():
            result["prediction_rows_document"] = _read_jsonl(prediction_rows)
    return result


def _metrics(bundle: Mapping[str, Any]) -> dict[str, float | None]:
    document = bundle.get("metrics_document") if isinstance(bundle.get("metrics_document"), Mapping) else bundle
    nested = document.get("metrics") if isinstance(document, Mapping) and isinstance(document.get("metrics"), Mapping) else document
    result: dict[str, float | None] = {}
    aliases = {
        "top1_accuracy": ("top1_accuracy", "accuracy", "top1", "top1_acc"),
        "top3_accuracy": ("top3_accuracy", "top3", "top3_acc"),
        "macro_precision": ("macro_precision", "precision"),
        "macro_recall": ("macro_recall", "recall"),
        "macro_f1": ("macro_f1", "f1"),
    }
    for canonical, keys in aliases.items():
        value = next((nested.get(key) for key in keys if nested.get(key) not in (None, "")), None)
        try:
            result[canonical] = round(float(value), 6) if value is not None else None
        except (TypeError, ValueError):
            result[canonical] = None
    return result


def _model_version(bundle: Mapping[str, Any]) -> str:
    for key in ("model_version",):
        if _text(bundle.get(key)):
            return _text(bundle.get(key))
    for key in ("metrics_document", "report_document", "confusion_matrix_document"):
        document = bundle.get(key)
        if isinstance(document, Mapping) and _text(document.get("model_version")):
            return _text(document.get("model_version"))
    return "unknown"


def _labels_and_matrix(bundle: Mapping[str, Any]) -> tuple[list[str], list[list[int]]]:
    document = bundle.get("confusion_matrix_document")
    if not isinstance(document, Mapping):
        document = bundle.get("report_document") if isinstance(bundle.get("report_document"), Mapping) else bundle
    if not isinstance(document, Mapping) or not (document.get("matrix") or document.get("confusion_matrix")):
        metrics_document = bundle.get("metrics_document")
        if isinstance(metrics_document, Mapping):
            document = metrics_document
    labels = [_text(value) for value in (document.get("labels") or []) if _text(value)] if isinstance(document, Mapping) else []
    if not labels:
        class_map = bundle.get("class_map_document")
        classes = class_map.get("classes") if isinstance(class_map, Mapping) else class_map
        if isinstance(classes, Mapping):
            classes = classes.get("classes", classes.get("labels", classes))
        if isinstance(classes, Mapping):
            ordered_mapping = []
            for raw_index, value in classes.items():
                index = _safe_int(raw_index, len(ordered_mapping))
                label = _text(value)
                if label:
                    ordered_mapping.append((index, label))
            labels = [label for _index, label in sorted(ordered_mapping)]
        elif isinstance(classes, list):
            ordered: list[tuple[int, str]] = []
            for fallback, value in enumerate(classes):
                if isinstance(value, Mapping):
                    index = _safe_int(value.get("class_index", value.get("index", fallback)), fallback)
                    label = _text(value.get("species_key") or value.get("common_name_en") or value.get("common_name_zh") or value.get("name"))
                else:
                    index, label = fallback, _text(value)
                if label:
                    ordered.append((index, label))
            labels = [label for _index, label in sorted(ordered)]
    matrix = document.get("matrix") if isinstance(document, Mapping) else None
    if matrix is None and isinstance(document, Mapping):
        matrix = document.get("confusion_matrix")
    if matrix is None and isinstance(document, Mapping) and isinstance(document.get("test"), Mapping):
        matrix = document["test"].get("confusion_matrix")
    normalized = []
    for row in matrix or []:
        if isinstance(row, (list, tuple)):
            normalized.append([max(0, _safe_int(value, 0)) for value in row])
    size = max(len(labels), len(normalized))
    labels.extend(f"class_{index}" for index in range(len(labels), size))
    normalized = [row[:size] + [0] * max(0, size - len(row)) for row in normalized[:size]]
    normalized.extend([[0] * size for _ in range(size - len(normalized))])
    return labels, normalized


def _per_class(bundle: Mapping[str, Any], labels: list[str], matrix: list[list[int]]) -> dict[str, dict[str, float | int]]:
    document = bundle.get("report_document") if isinstance(bundle.get("report_document"), Mapping) else bundle
    metrics_document = bundle.get("metrics_document") if isinstance(bundle.get("metrics_document"), Mapping) else {}
    candidates = []
    for parent in (document, metrics_document, document.get("test") if isinstance(document, Mapping) else None, metrics_document.get("test") if isinstance(metrics_document, Mapping) else None):
        if isinstance(parent, Mapping) and isinstance(parent.get("per_class"), list):
            candidates = parent["per_class"]
            break
        if isinstance(parent, Mapping) and isinstance(parent.get("per_class_metrics"), list):
            candidates = parent["per_class_metrics"]
            break
    result: dict[str, dict[str, float | int]] = {}
    for index, label in enumerate(labels):
        row = next(
            (
                item
                for item in candidates
                if isinstance(item, Mapping)
                and _safe_int(item.get("class_index", item.get("index", -1)), -1) == index
            ),
            None,
        )
        if row is not None:
            values: dict[str, float | int] = {}
            for key in ("precision", "recall", "f1"):
                try:
                    values[key] = float(row.get(key, 0.0) or 0.0)
                except (TypeError, ValueError):
                    values[key] = 0.0
            values["support"] = _safe_int(row.get("support", 0), 0)
            result[label] = values
            continue
        support = sum(matrix[index]) if index < len(matrix) else 0
        tp = matrix[index][index] if index < len(matrix) and index < len(matrix[index]) else 0
        predicted = sum(row[index] for row in matrix if index < len(row))
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        result[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return result


def _pair_errors(labels: list[str], matrix: list[list[int]], left: str, right: str) -> int:
    indexes = {label: index for index, label in enumerate(labels)}
    i, j = indexes.get(left), indexes.get(right)
    if i is None or j is None:
        return 0
    return (matrix[i][j] if i < len(matrix) and j < len(matrix[i]) else 0) + (
        matrix[j][i] if j < len(matrix) and i < len(matrix[j]) else 0
    )


def compare_model_artifacts(
    baseline: Mapping[str, Any] | str | Path,
    candidate: Mapping[str, Any] | str | Path,
    *,
    focus_pairs: Iterable[tuple[str, str]] = FOCUS_PAIRS,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a MODEL_COMPARE_REPORT with overall, per-class and pair deltas."""

    base = _load_bundle(baseline)
    new = _load_bundle(candidate)
    base_metrics = _metrics(base)
    new_metrics = _metrics(new)
    metric_report = {}
    for key in METRIC_KEYS:
        before, after = base_metrics.get(key), new_metrics.get(key)
        metric_report[key] = {
            "baseline": before,
            "candidate": after,
            "delta": round(after - before, 6) if before is not None and after is not None else None,
        }

    base_labels, base_matrix = _labels_and_matrix(base)
    new_labels, new_matrix = _labels_and_matrix(new)
    base_classes = _per_class(base, base_labels, base_matrix)
    new_classes = _per_class(new, new_labels, new_matrix)
    per_class = []
    for species in sorted(set(base_classes) | set(new_classes)):
        before = base_classes.get(species, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0})
        after = new_classes.get(species, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0})
        per_class.append(
            {
                "species": species,
                "baseline": before,
                "candidate": after,
                "precision_delta": round(float(after["precision"]) - float(before["precision"]), 6),
                "recall_delta": round(float(after["recall"]) - float(before["recall"]), 6),
                "f1_delta": round(float(after["f1"]) - float(before["f1"]), 6),
            }
        )

    pairs = []
    for left, right in focus_pairs:
        before = _pair_errors(base_labels, base_matrix, left, right)
        after = _pair_errors(new_labels, new_matrix, left, right)
        pairs.append({"species": [left, right], "baseline_errors": before, "candidate_errors": after, "delta": after - before})

    improved = [row for row in per_class if row["f1_delta"] > 0]
    declined = [row for row in per_class if row["f1_delta"] < 0]
    return {
        "report_type": COMPARE_REPORT_TYPE,
        "contract_version": "model-compare-v1",
        "generated_at": generated_at or "",
        "baseline_model_version": _model_version(base),
        "candidate_model_version": _model_version(new),
        "metrics": metric_report,
        # ``overall`` is a readable alias for consumers that render the
        # comparison as a report rather than an artifact API response.
        "overall": metric_report,
        "per_class": per_class,
        "confusion_matrix": {
            "baseline": {"labels": base_labels, "matrix": base_matrix},
            "candidate": {"labels": new_labels, "matrix": new_matrix},
        },
        "focus_pairs": pairs,
        "crop_gain": {"improved": improved, "declined": declined, "improved_count": len(improved), "declined_count": len(declined)},
        "safety": {"labels_modified": False, "training_triggered": False, "dataset_modified": False},
    }


def write_model_compare_report(report: Mapping[str, Any], output_path: str | Path) -> dict[str, Any]:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return dict(report)


__all__ = ["COMPARE_REPORT_TYPE", "FOCUS_PAIRS", "compare_model_artifacts", "write_model_compare_report"]
