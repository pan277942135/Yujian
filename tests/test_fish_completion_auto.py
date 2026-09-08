import numpy as np

from app.fish_completion_auto import _bbox_to_pixels, analyze_completion


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


def test_auto_completion_invalid_box_is_not_eligible():
    result, candidate = analyze_completion(np.zeros((10, 10), dtype=bool), (4, 4, 4, 8))
    assert not result["completion_required"]
    assert result["severity"] == "NOT_ELIGIBLE"
    assert not candidate.any()
