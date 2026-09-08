import numpy as np

from app.fish_completion_auto import analyze_completion


def test_auto_completion_stays_inside_primary_bbox():
    mask = np.zeros((40, 60), dtype=bool)
    mask[10:30, 15:45] = True
    result, candidate = analyze_completion(mask, (15, 10, 45, 30))
    assert candidate.shape == mask.shape
    assert not candidate[:10].any()
    assert not candidate[30:].any()
    assert 0 <= result["completion_ratio"] <= 1
    assert result["region_count"] <= 2 or result["severity"] == "NOT_ELIGIBLE"


def test_auto_completion_invalid_box_is_not_eligible():
    result, candidate = analyze_completion(np.zeros((10, 10), dtype=bool), (4, 4, 4, 8))
    assert not result["completion_required"]
    assert result["severity"] == "NOT_ELIGIBLE"
    assert not candidate.any()
