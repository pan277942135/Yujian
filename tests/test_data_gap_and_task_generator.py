from __future__ import annotations

from app.intelligence.data_gap_analyzer import analyze_data_gaps
from app.intelligence.task_generator import generate_collection_task, generate_collection_tasks


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
    assert report["quantity_gaps"] == report["species_gaps"]
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
                "test_support": 11,
                "priority_score": 9.1,
            }
        ],
    }
    gaps = {
        "species_gaps": [
            {"species": "grass_carp", "current": 48, "target": 300, "gap": 252},
            {"species": "common_carp", "current": 90, "target": 300, "gap": 210},
        ],
        "quantity_gaps": [
            {"species": "grass_carp", "current": 48, "target": 300, "gap": 252},
            {"species": "common_carp", "current": 90, "target": 300, "gap": 210},
        ],
        "scene_gaps": [{"species": "grass_carp", "missing_scenes": ["night", "fish_net"]}],
        "recommended_scenes": ["night", "fish_net"],
    }
    task = generate_collection_task(confusion, gaps, generated_at="2026-09-01T00:00:00+00:00")
    assert task["task_id"] == "TASK_20260901_001"
    assert task["task_type"] == "HARD_CASE_COLLECTION"
    assert task["true_species"] == "grass_carp"
    assert task["confused_species"] == "common_carp"
    assert task["priority"] == "P0"
    assert task["reason"][0]["errors"] == 3
    assert [(row["name"], row["count"]) for row in task["requirements"]["species"]] == [("grass_carp", 100)]
    assert task["requirements"]["scenes"] == ["night", "fish_net"]
    assert task["batch_suggestion"]["batch_id"] == "BATCH_HARDCASE_20260901_001"
    assert task["batch_suggestion"]["batch_type"] == "HARD_CASE_COLLECTION"
    assert task["batch_suggestion"]["metadata"]["target_species"] == ["grass_carp"]
    assert task["batch_suggestion"]["metadata"]["reason"] == "grass_carp_to_common_carp"
    assert task["batch_suggestion"]["upload_url"] == "/platform/data/import"
    assert task["safety"]["creates_batch"] is False


def test_collection_tasks_keep_hard_case_first_and_separate_other_gaps():
    tasks = generate_collection_tasks(
        {
            "model_version": "MODEL_CROP_M1_v0.1",
            "top_confusions": [
                {
                    "true_species": "silver_carp",
                    "pred_species": "bighead_carp",
                    "error_count": 5,
                    "error_rate": 0.263,
                    "priority": "P0",
                    "test_support": 19,
                }
            ],
        },
        {
            "quantity_gaps": [{"species": "mandarin_fish", "current": 194, "target": 250, "gap": 56}],
            "scene_gaps": [{"species": "grass_carp", "missing_scenes": ["night", "fish_net"]}],
        },
        generated_at="2026-09-15T00:00:00+00:00",
    )

    assert [task["task_type"] for task in tasks] == [
        "HARD_CASE_COLLECTION",
        "DATA_BALANCE_COLLECTION",
        "SCENE_GAP_COLLECTION",
    ]
    assert tasks[0]["requirements"]["species"] == [{"name": "silver_carp", "count": 100, "priority": "P0"}]
    assert tasks[1]["requirements"]["species"] == [{"name": "mandarin_fish", "count": 56}]
    assert tasks[2]["requirements"]["scenes"] == ["night", "fish_net"]
