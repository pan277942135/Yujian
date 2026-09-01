from __future__ import annotations

import json

from app.intelligence.confusion_analyzer import (
    ConfusionAnalyzer,
    build_confusion_report,
    write_confusion_report,
)


def test_confusion_matrix_marks_three_errors_as_p0():
    evaluation = {
        "model_version": "MODEL_M1_v0.3",
        "test": {
            "confusion_matrix": [[7, 3], [1, 8]],
        },
        "classes": [
            {"class_index": 0, "species_key": "grass_carp"},
            {"class_index": 1, "species_key": "common_carp"},
        ],
    }

    report = build_confusion_report(evaluation, generated_at="2026-09-01T00:00:00+00:00")

    assert report["model_version"] == "MODEL_M1_v0.3"
    assert report["top_confusions"][0]["true_species"] == "grass_carp"
    assert report["top_confusions"][0]["pred_species"] == "common_carp"
    assert report["top_confusions"][0]["error_count"] == 3
    assert report["top_confusions"][0]["error_rate"] == 0.3
    assert report["top_confusions"][0]["priority"] == "P0"


def test_sample_level_evaluation_and_priority_score_are_supported(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps(
            {
                "model_version": "MODEL_M1_v0.3",
                "samples": [
                    {"image_id": "a", "true_species": "grass_carp", "prediction": "common_carp"},
                    {"image_id": "b", "true_species": "grass_carp", "prediction": "common_carp"},
                    {"image_id": "c", "true_species": "grass_carp", "prediction": "grass_carp"},
                    {"image_id": "d", "true_species": "common_carp", "prediction": "grass_carp"},
                ],
            }
        ),
        encoding="utf-8",
    )

    report = ConfusionAnalyzer(species_importance={"grass_carp": 2})(path)
    pair = next(row for row in report["top_confusions"] if row["true_species"] == "grass_carp")
    assert pair["error_count"] == 2
    assert pair["error_rate"] == 2 / 3
    assert pair["priority"] == "P1"
    assert pair["priority_score"] == round(2 * (2 / 3) * 2, 6)


def test_write_confusion_report_creates_json(tmp_path):
    destination = tmp_path / "reports" / "confusion_report.json"
    write_confusion_report(
        {"model_version": "MODEL_M1_v0.3", "confusion_matrix": [[1, 0], [0, 1]]},
        destination,
    )
    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert saved["model_version"] == "MODEL_M1_v0.3"
    assert saved["top_confusions"] == []
