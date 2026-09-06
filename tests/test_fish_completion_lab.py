import numpy as np
import pytest

from app.fish_completion_lab import _stats


def test_completion_mask_must_be_subset_of_occluder():
    raw = np.zeros((4, 4), dtype=bool)
    add = np.zeros((4, 4), dtype=bool)
    remove = np.zeros((4, 4), dtype=bool)
    occluder = np.zeros((4, 4), dtype=bool)
    completion = np.zeros((4, 4), dtype=bool)
    completion[0, 0] = True
    result = _stats(raw, add, remove, occluder, completion)
    assert result["completion_mask_valid"] is False
    assert result["illegal_completion_pixels"] == 1


@pytest.mark.parametrize(("area", "level"), [(0, "LIGHT"), (5, "LIGHT"), (6, "MEDIUM"), (12, "HEAVY")])
def test_generated_ratio_levels(area, level):
    raw = np.ones((10, 10), dtype=bool)
    completion = np.zeros((10, 10), dtype=bool)
    completion.flat[:area] = True
    result = _stats(raw, np.zeros_like(raw), np.zeros_like(raw), np.zeros_like(raw), completion)
    assert result["completion_level"] == level


def test_completion_regions_over_two_are_blocked():
    raw = np.ones((20, 20), dtype=bool)
    completion = np.zeros((20, 20), dtype=bool)
    completion[1, 1] = completion[1, 3] = completion[1, 5] = True
    occluder = completion.copy()
    result = _stats(raw, np.zeros_like(raw), np.zeros_like(raw), occluder, completion)
    assert result["completion_region_count"] == 3
    assert result["eligible_for_v0_1"] is False
