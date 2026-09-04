from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


ARTIFACT_CONTRACT_VERSION = "evaluation-artifact-v1"
PREDICTION_FIELDS = (
    "image_id",
    "true_species",
    "pred_species",
    "confidence",
    "correct",
    "model_version",
    "dataset_version",
)
ERROR_SAMPLE_FIELDS = (
    "image_id",
    "true_species",
    "pred_species",
    "confidence",
    "error_group",
    "model_version",
)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class MetricsArtifact:
    """Stable envelope for metrics while allowing future metric keys."""

    model_version: str
    dataset_version: str
    test_samples: int
    metrics: dict[str, Any] = field(default_factory=dict)
    generated_at: str = ""
    contract_version: str = ARTIFACT_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if not result["generated_at"]:
            result["generated_at"] = utcnow_iso()
        return result


@dataclass(frozen=True)
class ConfusionMatrixArtifact:
    labels: list[str]
    matrix: list[list[int]]
    model_version: str = ""
    dataset_version: str = ""
    generated_at: str = ""
    contract_version: str = ARTIFACT_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if not result["generated_at"]:
            result["generated_at"] = utcnow_iso()
        return result


@dataclass(frozen=True)
class EvaluationArtifact:
    """Paths and counts returned after writing a complete artifact set."""

    model_version: str
    dataset_version: str
    artifact_root: str
    metrics_path: str
    confusion_matrix_path: str
    predictions_path: str
    error_samples_path: str
    report_path: str
    test_samples: int
    prediction_rows: int
    error_rows: int
    # Added in Evaluation Artifact Contract v1.1.  A default keeps callers
    # that construct the v1 dataclass directly source-compatible; the builder
    # always writes and returns the JSONL path.
    prediction_rows_path: str = ""
    generated_at: str = ""
    contract_version: str = ARTIFACT_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if not result["generated_at"]:
            result["generated_at"] = utcnow_iso()
        return result
