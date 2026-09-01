from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .artifact_schema import ARTIFACT_CONTRACT_VERSION


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_evaluation_report(
    *,
    model_version: str,
    dataset_version: str,
    metrics: Mapping[str, Any],
    test_samples: int,
    prediction_rows: int,
    error_rows: int,
    generated_at: str | None = None,
) -> dict[str, Any]:
    return {
        "contract_version": ARTIFACT_CONTRACT_VERSION,
        "model_version": model_version,
        "dataset_version": dataset_version,
        "generated_at": generated_at or utcnow_iso(),
        "metrics": dict(metrics),
        "test_samples": int(test_samples),
        "prediction_rows": int(prediction_rows),
        "error_samples": int(error_rows),
        "artifacts": {
            "metrics": "metrics.json",
            "confusion_matrix": "confusion_matrix.json",
            "predictions": "predictions.csv",
            "error_samples": "error_samples.json",
            "report": "report.json",
        },
    }


def write_evaluation_report(report: Mapping[str, Any], output_path: str | Path) -> dict[str, Any]:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return dict(report)


def generate_report(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return build_evaluation_report(*args, **kwargs)


__all__ = ["build_evaluation_report", "generate_report", "write_evaluation_report"]
