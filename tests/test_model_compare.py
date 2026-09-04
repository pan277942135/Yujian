from __future__ import annotations

from evaluation.model_compare import COMPARE_REPORT_TYPE, compare_model_artifacts


def _bundle(model: str, matrix: list[list[int]], f1: list[float]) -> dict:
    return {
        "model_version": model,
        "metrics": {
            "top1_accuracy": 0.60 if "CROP" not in model else 0.70,
            "top3_accuracy": 0.85 if "CROP" not in model else 0.92,
            "macro_precision": 0.58 if "CROP" not in model else 0.68,
            "macro_recall": 0.55 if "CROP" not in model else 0.66,
            "macro_f1": 0.56 if "CROP" not in model else 0.67,
        },
        "labels": ["crucian_carp", "common_carp", "silver_carp", "bighead_carp"],
        "confusion_matrix": matrix,
        "per_class": [
            {"class_index": index, "precision": value, "recall": value, "f1": value, "support": 10}
            for index, value in enumerate(f1)
        ],
    }


def test_model_compare_reports_metric_and_crop_gain_deltas():
    baseline = _bundle(
        "MODEL_M1_v0.5",
        [[7, 3, 0, 0], [4, 6, 0, 0], [0, 0, 8, 2], [0, 0, 3, 7]],
        [0.5, 0.55, 0.6, 0.65],
    )
    candidate = _bundle(
        "MODEL_CROP_M1_v0.1",
        [[9, 1, 0, 0], [1, 9, 0, 0], [0, 0, 9, 1], [0, 0, 1, 9]],
        [0.8, 0.75, 0.7, 0.72],
    )
    report = compare_model_artifacts(baseline, candidate)
    assert report["report_type"] == COMPARE_REPORT_TYPE
    assert report["baseline_model_version"] == "MODEL_M1_v0.5"
    assert report["candidate_model_version"] == "MODEL_CROP_M1_v0.1"
    assert report["metrics"]["top1_accuracy"]["delta"] == 0.1
    assert report["crop_gain"]["improved_count"] == 4
    pair = next(row for row in report["focus_pairs"] if row["species"] == ["crucian_carp", "common_carp"])
    assert pair["baseline_errors"] == 7
    assert pair["candidate_errors"] == 2


def test_compare_accepts_legacy_model_metrics_directory(tmp_path):
    root = tmp_path / "legacy"
    root.mkdir()
    (root / "metrics.json").write_text(
        '{"model_version":"MODEL_M1_v0.5","test":{"accuracy":0.5,"confusion_matrix":[[2,1],[1,2]]}}',
        encoding="utf-8",
    )
    (root / "class_map.json").write_text(
        '{"classes":[{"class_index":0,"species_key":"crucian_carp"},{"class_index":1,"species_key":"common_carp"}]}',
        encoding="utf-8",
    )
    report = compare_model_artifacts(root, root)
    assert report["baseline_model_version"] == "MODEL_M1_v0.5"
    assert report["confusion_matrix"]["baseline"]["labels"] == ["crucian_carp", "common_carp"]
