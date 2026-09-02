from __future__ import annotations

import csv
import json

from evaluation.artifact_builder import EvaluationArtifactBuilder, build_evaluation_artifacts
from evaluation.artifact_schema import ERROR_SAMPLE_FIELDS, PREDICTION_FIELDS


def test_builder_writes_contract_v1_files_and_preserves_metrics(tmp_path):
    report = {
        "model_version": "MODEL_M1_v0.3",
        "dataset_version": "DS_M1_v0.3",
        "test": {
            "count": 107,
            "accuracy": 0.645,
            "top3_accuracy": 0.879,
            "macro_precision": 0.662,
            "macro_recall": 0.599,
            "macro_f1": 0.601,
            "confusion_matrix": [[6, 3, 2], [3, 7, 0], [1, 0, 6]],
        },
        "classes": ["grass_carp", "common_carp", "black_carp"],
    }
    predictions = [
        {"image_id": "P500001", "true_species": "grass_carp", "pred_species": "common_carp", "confidence": 0.62},
        {"image_id": "P500002", "true_species": "common_carp", "pred_species": "grass_carp", "confidence": 0.71},
        {"image_id": "P500003", "true_species": "grass_carp", "pred_species": "black_carp", "confidence": 0.55},
        {"image_id": "P500004", "true_species": "grass_carp", "pred_species": "grass_carp", "confidence": 0.91},
    ]

    result = build_evaluation_artifacts(report, predictions, tmp_path)
    root = tmp_path / "MODEL_M1_v0.3"
    expected = {"metrics.json", "confusion_matrix.json", "predictions.csv", "prediction_rows.jsonl", "error_samples.json", "report.json"}
    assert {path.name for path in root.iterdir()} == expected
    assert result["test_samples"] == 107
    assert result["error_rows"] == 3

    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["model_version"] == "MODEL_M1_v0.3"
    assert metrics["dataset_version"] == "DS_M1_v0.3"
    assert metrics["test_samples"] == 107
    assert metrics["metrics"] == {
        "top1_accuracy": 0.645,
        "top3_accuracy": 0.879,
        "macro_precision": 0.662,
        "macro_recall": 0.599,
        "macro_f1": 0.601,
        "count": 107,
    }
    confusion = json.loads((root / "confusion_matrix.json").read_text(encoding="utf-8"))
    assert confusion["labels"] == ["grass_carp", "common_carp", "black_carp"]
    assert confusion["matrix"][0] == [6, 3, 2]

    with (root / "predictions.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == list(PREDICTION_FIELDS)
    assert rows[0]["correct"] == "False"

    errors = json.loads((root / "error_samples.json").read_text(encoding="utf-8"))
    assert len(errors) == 3
    assert set(ERROR_SAMPLE_FIELDS).issubset(errors[0])
    assert errors[0]["error_group"] == "grass_carp_vs_common_carp"
    prediction_lines = (root / "prediction_rows.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(prediction_lines) == 4
    assert '"predicted_species":"common_carp"' in prediction_lines[0]


def test_builder_accepts_class_indices_and_object_facade(tmp_path):
    builder = EvaluationArtifactBuilder(tmp_path, model_version="MODEL_TEST", dataset_version="DS_TEST")
    result = builder(
        {"test_samples": 2, "metrics": {"top1_accuracy": 0.5}, "confusion_matrix": [[1, 1], [0, 0]]},
        [
            {"image_id": "a", "true_index": 0, "pred_index": 1, "confidence": 0.2},
            {"image_id": "b", "true_index": 0, "pred_index": 0, "confidence": 0.8},
        ],
        class_map=["grass_carp", "common_carp"],
    )
    assert result["model_version"] == "MODEL_TEST"
    assert result["dataset_version"] == "DS_TEST"
    assert result["error_rows"] == 1
    confusion = json.loads((tmp_path / "MODEL_TEST" / "confusion_matrix.json").read_text(encoding="utf-8"))
    assert confusion["labels"] == ["grass_carp", "common_carp"]
