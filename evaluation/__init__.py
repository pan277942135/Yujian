"""Evaluation Artifact Contract v1 helpers.

The package is deliberately post-evaluation only: it serializes metrics and
test predictions that already exist, without changing model training or
Dataset Freeze state.
"""

from .artifact_builder import (
    EvaluationArtifactBuilder,
    build_evaluation_artifact,
    build_evaluation_artifacts,
    write_evaluation_artifacts,
)
from .artifact_schema import (
    ARTIFACT_CONTRACT_VERSION,
    ERROR_SAMPLE_FIELDS,
    PREDICTION_FIELDS,
)
from .confusion_exporter import (
    build_confusion_matrix_artifact,
    export_confusion_matrix,
    write_confusion_matrix,
)
from .prediction_exporter import (
    build_error_samples,
    normalize_prediction_rows,
    write_error_samples,
    write_predictions_csv,
)

__all__ = [
    "ARTIFACT_CONTRACT_VERSION",
    "ERROR_SAMPLE_FIELDS",
    "EvaluationArtifactBuilder",
    "PREDICTION_FIELDS",
    "build_confusion_matrix_artifact",
    "build_error_samples",
    "build_evaluation_artifact",
    "build_evaluation_artifacts",
    "export_confusion_matrix",
    "normalize_prediction_rows",
    "write_confusion_matrix",
    "write_error_samples",
    "write_evaluation_artifacts",
    "write_predictions_csv",
]
