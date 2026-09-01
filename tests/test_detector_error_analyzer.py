from app.intelligence.detector_error_analyzer import analyze_detector_errors


def test_detector_error_analyzer_reports_miss_and_multiple_fish_without_auto_actions():
    report = analyze_detector_errors(
        [
            {"image_id": "miss", "detector_version": "DET_FISH_v0.1", "human_present": True},
            {
                "image_id": "multi",
                "detector": {"detector_version": "DET_FISH_v0.1", "detections": [
                    {"candidate_bbox": [0.1, 0.1, 0.3, 0.3]},
                    {"candidate_bbox": [0.5, 0.1, 0.3, 0.3]},
                ]},
                "accepted_bbox": [0.1, 0.1, 0.7, 0.7],
            },
        ]
    )
    assert report["error_counts"]["missed_detection"] == 1
    assert report["error_counts"]["multiple_fish"] == 1
    assert report["improvement_task"]["safety"]["creates_batch"] is False


def test_detector_error_analyzer_uses_reference_for_box_size():
    report = analyze_detector_errors(
        [{
            "image_id": "large",
            "detection": {"candidate_bbox": [0.0, 0.0, 0.9, 0.9], "detector_version": "DET_FISH_v0.1"},
            "accepted_bbox": [0.2, 0.2, 0.3, 0.3],
        }]
    )
    assert "bbox_too_large" in report["error_counts"]
