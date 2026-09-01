from __future__ import annotations

import csv

from app.intelligence.hard_case_miner import HARD_CASE_FIELDS, mine_hard_cases


def test_hard_case_miner_copies_error_image_and_writes_manifest(tmp_path):
    source = tmp_path / "source.jpg"
    source.write_bytes(b"fake-image")
    report = {
        "model_version": "MODEL_M1_v0.3",
        "top_confusions": [
            {
                "true_species": "grass_carp",
                "pred_species": "common_carp",
                "error_count": 3,
                "error_rate": 0.273,
                "priority": "P0",
            }
        ],
    }
    result = mine_hard_cases(
        [
            {
                "image_id": "P500123",
                "file_name": "original.jpg",
                "local_path": str(source),
                "true_species": "grass_carp",
                "pred_species": "common_carp",
                "confidence": 0.62,
                "hard_case_type": "carp_family_boundary",
            }
        ],
        report,
        tmp_path / "hard_cases",
    )

    assert result["row_count"] == 1
    assert result["copied_count"] == 1
    output_image = tmp_path / "hard_cases" / "MODEL_M1_v0.3" / "grass_carp_vs_common_carp" / "images" / "original.jpg"
    assert output_image.read_bytes() == b"fake-image"
    with open(result["manifest_path"], encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == list(HARD_CASE_FIELDS)
    assert rows[0]["image_id"] == "P500123"
    assert rows[0]["hard_case_type"] == "carp_family_boundary"


def test_hard_case_miner_keeps_unavailable_uri_for_operator_follow_up(tmp_path):
    result = mine_hard_cases(
        [
            {
                "image_id": "missing-1",
                "file_name": "missing.jpg",
                "gcs_uri": "gs://bucket/eval/missing.jpg",
                "true": "grass_carp",
                "pred": "common_carp",
                "confidence": "0.4",
            }
        ],
        {"model_version": "MODEL_M1_v0.3", "top_confusions": []},
        tmp_path / "hard_cases",
    )
    assert result["row_count"] == 1
    assert result["missing_image_count"] == 1
    assert (tmp_path / "hard_cases" / "MODEL_M1_v0.3" / "metadata" / "hard_case_manifest.csv").exists()
