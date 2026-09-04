from __future__ import annotations

from app.intelligence.data_gap_analyzer import analyze_data_gaps
from app.intelligence.task_generator import generate_collection_task


def test_data_gap_analyzer_reports_count_and_scene_gaps():
    rows = [
        {"species_key": "grass_carp", "scene": "river", "angle": "side", "image_quality": "good"}
        for _ in range(48)
    ]
    report = analyze_data_gaps(
        rows,
        {"targets": {"grass_carp": 300}, "required_scenes": ["river", "night", "fish_net"]},
    )
    assert report["species_gaps"] == [{"species": "grass_carp", "current": 48, "target": 300, "gap": 252}]
    assert report["scene_gaps"][0]["missing_scenes"] == ["night", "fish_net"]


def test_task_generator_uses_confusion_pair_and_target_counts():
    confusion = {
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
    gaps = {
        "species_gaps": [
            {"species": "grass_carp", "current": 48, "target": 300, "gap": 252},
            {"species": "common_carp", "current": 90, "target": 300, "gap": 210},
        ],
        "recommended_scenes": ["night", "fish_net"],
    }
    task = generate_collection_task(confusion, gaps, generated_at="2026-09-01T00:00:00+00:00")
    assert task["task_id"] == "TASK_20260901_001"
    assert task["task_type"] == "HARD_CASE_COLLECTION"
    assert task["reason"][0]["errors"] == 3
    assert [(row["name"], row["count"]) for row in task["requirements"]["species"]] == [
        ("grass_carp", 300),
        ("common_carp", 300),
    ]
    assert task["requirements"]["scenes"] == ["night", "fish_net"]
    assert task["batch_suggestion"] == {
        "batch_id": "BATCH_HARDCASE_20260901_001",
        "source": "MODEL_ERROR_DRIVEN",
    }
    assert task["safety"]["creates_batch"] is False
