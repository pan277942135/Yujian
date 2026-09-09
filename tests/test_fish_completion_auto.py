import numpy as np

from app.fish_completion_auto import _bbox_to_pixels, analyze_completion, analyze_completion_details


class Box:
    def __init__(self, x1, y1, x2, y2):
        self.values = (x1, y1, x2, y2)

    def normalized(self):
        return type("Normalized", (), dict(zip(("x1", "y1", "x2", "y2"), self.values)))()


def test_normalized_bbox_to_pixels():
    assert _bbox_to_pixels(Box(.1, .2, .8, .9), 1000, 500) == (100, 100, 800, 450)


def test_zero_normalized_bbox_remains_invalid():
    assert _bbox_to_pixels(Box(0, 0, 0, 0), 1000, 500) == (0, 0, 0, 0)


def test_auto_completion_stays_inside_primary_bbox():
    mask = np.zeros((40, 60), dtype=bool)
    mask[10:30, 15:45] = True
    result, candidate = analyze_completion(mask, (15, 10, 45, 30))
    assert candidate.shape == mask.shape
    assert not candidate[:10].any()
    assert not candidate[30:].any()
    assert 0 <= result["completion_ratio"] <= 1
    assert result["region_count"] <= 2 or result["severity"] == "NOT_ELIGIBLE"


def test_complete_mask_does_not_create_outer_contour_completion():
    mask = np.zeros((40, 60), dtype=bool)
    mask[10:30, 15:45] = True
    result, candidate = analyze_completion(mask, (10, 5, 50, 35))
    assert not candidate.any()
    assert result["completion_required"] is False
    assert result["severity"] == "COMPLETION_NOT_REQUIRED"
    assert result["reason"] == ["visible_mask_sufficient"]
    assert result["detection_method"] == "enclosed_hole_fill"


def test_enclosed_gap_is_completion_candidate_not_outer_border():
    mask = np.zeros((40, 60), dtype=bool)
    mask[10:30, 15:45] = True
    mask[18:22, 25:35] = False
    result, candidate = analyze_completion(mask, (10, 5, 50, 35))
    assert candidate[19, 30]
    assert not candidate[10:15, 15:45].any()
    assert result["completion_required"] is True
    assert "internal_gap_detected" in result["reason"]


def test_many_tiny_internal_holes_are_rejected_without_accepted_mask():
    mask = np.zeros((80, 80), dtype=bool)
    mask[10:70, 10:70] = True
    for index in range(17):
        y = 15 + (index // 5) * 8
        x = 15 + (index % 5) * 8
        mask[y, x] = False
    result, candidate, accepted, _envelope = analyze_completion_details(mask, (5, 5, 75, 75))
    assert result["severity"] == "NOT_ELIGIBLE"
    assert result["reason"] == ["candidate_too_fragmented"]
    assert candidate.any()
    assert not accepted.any()


def test_large_completion_is_experimental_not_ratio_blocked():
    mask = np.zeros((60, 60), dtype=bool)
    mask[8:52, 8:52] = True
    mask[18:42, 18:42] = False
    result, candidate, accepted, _envelope = analyze_completion_details(mask, (5, 5, 55, 55))
    assert result["severity"] == "LARGE_EXPERIMENTAL"
    assert result["completion_required"] is True
    assert result["completion_percent"] > 20
    assert "large_completion_area" in result["reason"]
    assert "completion_ratio_exceeded" not in result["reason"]
    assert accepted.sum() == candidate.sum()


def test_auto_completion_invalid_box_is_not_eligible():
    result, candidate = analyze_completion(np.zeros((10, 10), dtype=bool), (4, 4, 4, 8))
    assert not result["completion_required"]
    assert result["severity"] == "NOT_ELIGIBLE"
    assert not candidate.any()


from PIL import Image

from app.completion_worker_client import check_completion_worker
from app.fish_completion_auto import _auto_progress, _protected_compose


def test_protected_compose_does_not_change_visible_pixels():
    original = Image.new("RGB", (4, 4), (10, 10, 10))
    visible = np.ones((4, 4), dtype=bool)
    completion = np.zeros((4, 4), dtype=bool)
    completion[1:3, 1:3] = True
    visible[1:3, 1:3] = False
    generated = Image.new("RGB", (2, 2), (200, 20, 20))

    _, compose, _ = _protected_compose(
        original,
        visible,
        completion,
        generated,
        {"box": [1, 1, 3, 3], "original_size": [2, 2]},
    )

    assert compose["generated_pixels"] == 4
    assert compose["visible_changed_pixels"] == 0
    assert compose["visible_pixel_change_ratio"] == 0


def test_auto_progress_exposes_worker_and_timing_fields():
    progress = _auto_progress(
        {
            "timings": {"worker_ms": 123.4},
            "detector": {"bbox_pixel": [1, 2, 3, 4], "confidence": 0.9},
            "segmentation": {"quality": "GOOD"},
            "auto_completion": {"completion_required": True, "completion_ratio": 0.04},
            "completion_mask": {"area_pixels": 20},
            "worker": {
                "status": "WORKER_EXECUTED",
                "endpoint_configured": True,
                "health_status": 200,
                "inference_time_ms": 111,
            },
            "composition": {
                "generated_pixels": 20,
                "visible_changed_pixels": 0,
                "visible_pixel_change_ratio": 0,
            },
            "assets": {
                "fish_clean.png": "gs://clean",
                "fish_gold_outline.png": "gs://gold",
                "fish_black_outline.png": "gs://black",
                "final_asset": "gs://final",
            },
        }
    )

    worker_card = next(item for item in progress if item["stage"] == "powerpaint")
    assert worker_card["status"] == "WORKER_EXECUTED"
    assert worker_card["elapsed_ms"] == 123.4
    assert worker_card["result"]["inference_time_ms"] == 111


def test_worker_status_is_explicit_when_endpoint_missing(monkeypatch):
    monkeypatch.delenv("FISH_COMPLETION_WORKER_URL", raising=False)
    result = check_completion_worker()
    assert result["endpoint_configured"] is False
    assert result["status"] == "WORKER_UNAVAILABLE"
