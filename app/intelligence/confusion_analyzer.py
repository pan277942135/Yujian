from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schemas import ConfusionPair, ConfusionReport

_TRUE_KEYS = (
    "true_species",
    "true",
    "ground_truth",
    "ground_truth_species",
    "actual_species",
    "actual",
    "label_species",
    "claimed_species",
    "label",
)
_PRED_KEYS = (
    "pred_species",
    "predicted_species",
    "prediction",
    "predicted",
    "pred",
    "model_prediction",
)
_SAMPLE_KEYS = ("samples", "evaluation_samples", "predictions", "results", "items", "errors")
_MATRIX_KEYS = ("confusion_matrix", "matrix")


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_document(source: Mapping[str, Any] | list[Any] | str | Path) -> tuple[dict[str, Any], str | None]:
    if isinstance(source, list):
        return {"samples": source}, None
    if isinstance(source, Mapping):
        return dict(source), None
    path = Path(source)
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle), str(path)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        for key in ("species_key", "common_name_en", "common_name_zh", "name", "label", "id"):
            if value.get(key) not in (None, ""):
                return str(value[key]).strip()
        return ""
    return str(value).strip()


def _first(mapping: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return None


def _class_labels(document: Mapping[str, Any]) -> dict[int, str]:
    """Return class-index -> stable species label for common eval formats."""

    candidates = document.get("classes")
    if candidates is None:
        candidates = document.get("class_map")
    if candidates is None:
        candidates = document.get("labels")

    result: dict[int, str] = {}
    if isinstance(candidates, Mapping):
        # A mapping may be ``{"0": "grass_carp"}`` or a class-map document
        # containing a ``classes`` list.
        nested = candidates.get("classes")
        if nested is not None:
            return _class_labels({"classes": nested})
        for raw_index, value in candidates.items():
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            label = _text(value)
            if label:
                result[index] = label
        return result

    if isinstance(candidates, (list, tuple)):
        for fallback_index, row in enumerate(candidates):
            if isinstance(row, Mapping):
                raw_index = row.get("class_index", row.get("index", fallback_index))
                try:
                    index = int(raw_index)
                except (TypeError, ValueError):
                    index = fallback_index
                label = _text(row)
            else:
                index = fallback_index
                label = _text(row)
            if label:
                result[index] = label
    return result


def _find_matrix(document: Mapping[str, Any]) -> tuple[list[list[int]] | None, Mapping[str, Any]]:
    """Find the test/evaluation confusion matrix and its containing document."""

    def scan(container: Any) -> tuple[list[list[int]] | None, Mapping[str, Any]]:
        if not isinstance(container, Mapping):
            return None, {}
        for key in _MATRIX_KEYS:
            matrix = container.get(key)
            if isinstance(matrix, (list, tuple)) and matrix and all(isinstance(row, (list, tuple)) for row in matrix):
                return [list(row) for row in matrix], container
        for key in ("test", "validation", "evaluation", "metrics", "report"):
            matrix, owner = scan(container.get(key))
            if matrix is not None:
                return matrix, owner
        return None, {}

    return scan(document)


def _sample_rows(document: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in _SAMPLE_KEYS:
        rows = document.get(key)
        if isinstance(rows, list) and rows and all(isinstance(row, Mapping) for row in rows):
            return rows
    for key in ("test", "validation", "evaluation", "metrics", "report"):
        nested = document.get(key)
        if isinstance(nested, Mapping):
            rows = _sample_rows(nested)
            if rows:
                return rows
    return []


def _model_version(document: Mapping[str, Any], explicit: str | None) -> str:
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    for key in ("model_version", "model", "version"):
        value = document.get(key)
        if isinstance(value, Mapping):
            value = _first(value, ("model_version", "version", "name"))
        if value not in (None, ""):
            return str(value).strip()
    return "unknown"


def _normalise_matrix(matrix: list[list[int]]) -> list[list[int]]:
    size = max(len(matrix), *(len(row) for row in matrix))
    result: list[list[int]] = []
    for row in matrix:
        values = []
        for value in row:
            try:
                values.append(max(0, int(value)))
            except (TypeError, ValueError):
                values.append(0)
        values.extend([0] * (size - len(values)))
        result.append(values)
    result.extend([[0] * size for _ in range(size - len(result))])
    return result


def _priority(error_count: int, error_rate: float) -> str:
    if error_count >= 3 or error_rate >= 0.25:
        return "P0"
    if error_count >= 2:
        return "P1"
    return "P2"


def _sample_confusions(rows: Iterable[Mapping[str, Any]]) -> tuple[Counter[tuple[str, str]], Counter[str], int]:
    pair_counts: Counter[tuple[str, str]] = Counter()
    support: Counter[str] = Counter()
    total = 0
    for row in rows:
        truth = _text(_first(row, _TRUE_KEYS))
        pred = _text(_first(row, _PRED_KEYS))
        if not truth or not pred:
            continue
        total += 1
        support[truth] += 1
        if truth != pred:
            pair_counts[(truth, pred)] += 1
    return pair_counts, support, total


def build_confusion_report(
    evaluation: Mapping[str, Any] | str | Path,
    *,
    model_version: str | None = None,
    species_importance: Mapping[str, float] | None = None,
    generated_at: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Build the versioned confusion report from an existing evaluation result.

    The classifier trainer's ``test.confusion_matrix`` format is the primary
    input.  Sample-level ``true/pred`` rows are also accepted, which lets the
    hard-case miner use the same evaluation artifact without introducing a new
    evaluation contract.
    """

    document, source_path = _load_document(evaluation)
    labels = _class_labels(document)
    matrix, matrix_owner = _find_matrix(document)
    sample_rows = _sample_rows(document)
    pair_counts, sample_support, sample_total = _sample_confusions(sample_rows)

    support: Counter[str] = Counter(sample_support)
    counts: Counter[tuple[str, str]] = Counter(pair_counts)
    total_samples = sample_total or None

    if matrix is not None:
        matrix = _normalise_matrix(matrix)
        owner_labels = _class_labels(matrix_owner)
        if owner_labels:
            labels = {**labels, **owner_labels}
        for index, row in enumerate(matrix):
            truth = labels.get(index, f"class_{index}")
            support[truth] = sum(row)
            for pred_index, value in enumerate(row):
                count = max(0, int(value or 0))
                pred = labels.get(pred_index, f"class_{pred_index}")
                if truth != pred:
                    counts[(truth, pred)] = count
        total_samples = sum(support.values())

    importance = {str(key): float(value) for key, value in (species_importance or {}).items()}
    pairs: list[ConfusionPair] = []
    for (truth, pred), count in counts.items():
        count = int(count)
        if count <= 0 or truth == pred:
            continue
        denominator = int(support.get(truth, 0))
        rate = count / denominator if denominator else 0.0
        weight = importance.get(truth, importance.get(pred, 1.0))
        if weight < 0:
            weight = 0.0
        score = count * rate * weight
        pairs.append(
            ConfusionPair(
                true_species=truth,
                pred_species=pred,
                error_count=count,
                error_rate=round(rate, 6),
                priority=_priority(count, rate),
                priority_score=round(score, 6),
                species_importance=round(weight, 6),
            )
        )

    pairs.sort(key=lambda pair: (-pair.priority_score, -pair.error_count, -pair.error_rate, pair.true_species, pair.pred_species))
    if limit is not None:
        pairs = pairs[: max(0, int(limit))]
    report = ConfusionReport(
        model_version=_model_version(document, model_version),
        generated_at=generated_at or utcnow_iso(),
        top_confusions=pairs,
        source=source_path or document.get("source_uri") or document.get("metrics_uri"),
        total_samples=total_samples,
    ).to_dict()
    return report


def analyze_confusion(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Public alias used by the Console and external evaluation scripts."""

    return build_confusion_report(*args, **kwargs)


def analyze_confusion_matrix(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return build_confusion_report(*args, **kwargs)


def generate_confusion_report(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return build_confusion_report(*args, **kwargs)


def write_confusion_report(
    evaluation: Mapping[str, Any] | str | Path,
    output_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    report = build_confusion_report(evaluation, **kwargs)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


class ConfusionAnalyzer:
    """Small object-oriented facade for jobs that prefer dependency injection."""

    def __init__(self, *, species_importance: Mapping[str, float] | None = None):
        self.species_importance = species_importance or {}

    def analyze(self, evaluation: Mapping[str, Any] | str | Path, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("species_importance", self.species_importance)
        return build_confusion_report(evaluation, **kwargs)

    __call__ = analyze


__all__ = [
    "ConfusionAnalyzer",
    "analyze_confusion",
    "analyze_confusion_matrix",
    "build_confusion_report",
    "generate_confusion_report",
    "write_confusion_report",
]
