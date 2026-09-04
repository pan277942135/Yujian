from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .artifact_schema import EvaluationArtifact, utcnow_iso
from .confusion_exporter import build_confusion_matrix_artifact, write_confusion_matrix
from .prediction_exporter import (
    build_error_samples,
    normalize_prediction_rows,
    write_error_samples,
    write_prediction_rows_jsonl,
    write_predictions_csv,
)
from .report_generator import build_evaluation_report, write_evaluation_report


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _source_metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    source = report.get("metrics")
    if not isinstance(source, Mapping):
        source = report.get("test")
    if not isinstance(source, Mapping):
        source = report
    source = dict(source)
    aliases = {
        "top1_accuracy": ("top1_accuracy", "accuracy", "top1", "top1_acc"),
        "top3_accuracy": ("top3_accuracy", "top3", "top3_acc"),
        "macro_precision": ("macro_precision", "precision"),
        "macro_recall": ("macro_recall", "recall"),
        "macro_f1": ("macro_f1", "f1"),
    }
    result: dict[str, Any] = {}
    for canonical, keys in aliases.items():
        value = None
        for key in keys:
            if source.get(key) not in (None, ""):
                value = source[key]
                break
        if value is None:
            value = 0.0
        try:
            result[canonical] = round(float(value), 6)
        except (TypeError, ValueError):
            result[canonical] = value
    # Preserve additive metric fields (loss, per-class diagnostics, calibration
    # values, etc.) so future evaluation versions do not lose information.
    for key, value in source.items():
        if key not in {alias for aliases in aliases.values() for alias in aliases} and key not in {"confusion_matrix", "per_class"}:
            result.setdefault(key, value)
    return result


def _test_samples(report: Mapping[str, Any], predictions: list[Mapping[str, Any]]) -> int:
    for source in (report, report.get("test") if isinstance(report.get("test"), Mapping) else {}):
        if not isinstance(source, Mapping):
            continue
        for key in ("test_samples", "count", "support", "n_samples"):
            value = source.get(key)
            if value not in (None, ""):
                try:
                    return max(0, int(value))
                except (TypeError, ValueError):
                    pass
    return len(predictions)


def _prediction_source(report: Mapping[str, Any], predictions: Iterable[Mapping[str, Any]] | None) -> list[Mapping[str, Any]]:
    if predictions is not None:
        return [row for row in predictions if isinstance(row, Mapping)]
    for key in ("predictions", "evaluation_samples", "samples"):
        value = report.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, Mapping)]
    test = report.get("test")
    if isinstance(test, Mapping):
        for key in ("predictions", "samples"):
            value = test.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, Mapping)]
    return []


def build_evaluation_artifacts(
    evaluation_report: Mapping[str, Any],
    predictions: Iterable[Mapping[str, Any]] | None = None,
    output_root: str | Path = "evaluation_artifacts",
    *,
    model_version: str | None = None,
    dataset_version: str | None = None,
    class_map: Any = None,
    labels: Iterable[Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Write the complete Evaluation Artifact Contract v1 directory."""

    report = dict(evaluation_report or {})
    resolved_model = _text(model_version or report.get("model_version")) or "unknown"
    resolved_dataset = _text(dataset_version or report.get("dataset_version")) or "unknown"
    generated = generated_at or utcnow_iso()
    source_predictions = _prediction_source(report, predictions)
    normalized_predictions = normalize_prediction_rows(
        source_predictions,
        model_version=resolved_model,
        dataset_version=resolved_dataset,
        class_map=class_map,
    )
    errors = build_error_samples(
        normalized_predictions,
        model_version=resolved_model,
        dataset_version=resolved_dataset,
    )
    metrics = _source_metrics(report)
    test_samples = _test_samples(report, normalized_predictions)
    confusion = build_confusion_matrix_artifact(
        report,
        labels=labels,
        predictions=normalized_predictions,
        class_map=class_map,
        model_version=resolved_model,
        dataset_version=resolved_dataset,
        generated_at=generated,
    )
    destination = Path(output_root) / resolved_model
    destination.mkdir(parents=True, exist_ok=True)

    metrics_document = {
        "model_version": resolved_model,
        "dataset_version": resolved_dataset,
        "test_samples": test_samples,
        "metrics": metrics,
        "generated_at": generated,
        "contract_version": "evaluation-artifact-v1",
    }
    metrics_path = destination / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    confusion_path = destination / "confusion_matrix.json"
    write_confusion_matrix(confusion, confusion_path)
    predictions_path = destination / "predictions.csv"
    write_predictions_csv(normalized_predictions, predictions_path)
    prediction_rows_path = destination / "prediction_rows.jsonl"
    write_prediction_rows_jsonl(normalized_predictions, prediction_rows_path)
    errors_path = destination / "error_samples.json"
    write_error_samples(errors, errors_path)
    summary = build_evaluation_report(
        model_version=resolved_model,
        dataset_version=resolved_dataset,
        metrics=metrics,
        test_samples=test_samples,
        prediction_rows=len(normalized_predictions),
        error_rows=len(errors),
        generated_at=generated,
    )
    report_path = destination / "report.json"
    write_evaluation_report(summary, report_path)
    artifact = EvaluationArtifact(
        model_version=resolved_model,
        dataset_version=resolved_dataset,
        artifact_root=str(destination),
        metrics_path=str(metrics_path),
        confusion_matrix_path=str(confusion_path),
        predictions_path=str(predictions_path),
        prediction_rows_path=str(prediction_rows_path),
        error_samples_path=str(errors_path),
        report_path=str(report_path),
        test_samples=test_samples,
        prediction_rows=len(normalized_predictions),
        error_rows=len(errors),
        generated_at=generated,
    )
    result = artifact.to_dict()
    result.update({"metrics": metrics, "confusion_matrix": confusion, "predictions": normalized_predictions, "prediction_rows": normalized_predictions, "error_samples": errors, "report": summary})
    return result


def write_evaluation_artifacts(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return build_evaluation_artifacts(*args, **kwargs)


def build_evaluation_artifact(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return build_evaluation_artifacts(*args, **kwargs)


class EvaluationArtifactBuilder:
    def __init__(self, output_root: str | Path = "evaluation_artifacts", **defaults: Any):
        self.output_root = output_root
        self.defaults = defaults

    def build(self, evaluation_report: Mapping[str, Any], predictions: Iterable[Mapping[str, Any]] | None = None, **kwargs: Any) -> dict[str, Any]:
        options = dict(self.defaults)
        options.update(kwargs)
        options.setdefault("output_root", self.output_root)
        return build_evaluation_artifacts(evaluation_report, predictions, **options)

    __call__ = build


__all__ = [
    "EvaluationArtifactBuilder",
    "build_evaluation_artifact",
    "build_evaluation_artifacts",
    "write_evaluation_artifacts",
]
