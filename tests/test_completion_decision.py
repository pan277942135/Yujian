import inspect

import numpy as np
import pytest

from app.completion_decision import AUTO_COMPLETION, decide_completion
from app.fish_completion_lab import _stats


def _fish(width=240, height=120):
    mask = np.zeros((height, width), dtype=bool)
    mask[45:75, 20:220] = True
    return mask


def _internal_gap(start=100, end=120):
    mask = _fish()
    mask[45:75, start:end] = False
    return mask


def _decision(mask, assessment="", quality="GOOD"):
    return decide_completion(mask, (10, 30, 230, 90), mask.shape, detector_assessment=assessment, segmentation_quality=quality)


def test_complete_fish_is_not_required_without_case_label():
    decision = _decision(_fish())
    assert decision.mode == AUTO_COMPLETION
    assert decision.status == "NOT_REQUIRED"
    assert decision.case_class == "COMPLETE"
    assert not decision.completion_required
    assert not decision.completion_mask.any()


def test_local_missing_generates_local_auto_mask():
    mask = _fish()
    mask[45:58, 100:116] = False
    decision = _decision(mask)
    assert decision.status in {"MASK_READY", "LARGE_EXPERIMENTAL"}
    assert decision.completion_required
    assert decision.completion_mask.any()
    assert decision.ring_candidate_rejected
    assert decision.completion_mask.sum() < mask.sum()
    assert not np.any(decision.completion_mask & decision.clean_visible_mask)


def test_internal_structural_occlusion_uses_geometry_not_label():
    decision = _decision(_internal_gap(), assessment="incomplete_fish")
    assert decision.status in {"MASK_READY", "LARGE_EXPERIMENTAL"}
    assert decision.case_class in {"STRUCTURAL_OCCLUSION", "LARGE_MISSING"}
    assert decision.completion_required
    assert decision.occlusion_detected
    assert decision.occluder_mask.sum() == 0
    assert decision.decision_confidence >= 0.70


def test_high_confidence_missing_over_twenty_percent_is_experimental_and_allowed():
    decision = _decision(_internal_gap(78, 145), assessment="incomplete_fish")
    assert decision.completion_ratio > 0.20
    assert decision.status == "LARGE_EXPERIMENTAL"
    assert decision.execution_allowed
    assert decision.severity in {"LARGE_EXPERIMENTAL", "VERY_LARGE_EXPERIMENTAL"}


def test_very_large_structural_missing_is_not_rejected_by_ratio_alone():
    decision = _decision(_internal_gap(65, 155), assessment="incomplete_fish")
    assert decision.completion_ratio > 0.35
    assert decision.status in {"LARGE_EXPERIMENTAL", "REVIEW_REQUIRED"}
    assert "VISIBLE_FISH_AREA_INSUFFICIENT" not in decision.reason


def test_severe_ambiguous_missing_requires_review():
    mask = np.zeros((120, 240), dtype=bool)
    mask[52:68, 24:65] = True
    decision = _decision(mask, assessment="incomplete_fish", quality="WARNING")
    assert decision.status == "REVIEW_REQUIRED"
    assert not decision.completion_required
    assert "VISIBLE_FISH_AREA_INSUFFICIENT" not in decision.reason


def test_low_sam_bbox_fill_is_debug_metric_only():
    mask = np.zeros((120, 240), dtype=bool)
    mask[45:75, 80:160] = True
    decision = decide_completion(mask, (10, 20, 230, 100), mask.shape, segmentation_quality="GOOD")
    assert decision.sam_bbox_fill_ratio < 0.50
    assert decision.status != "NOT_ELIGIBLE"
    assert "LOW_SAM_BBOX_FILL_DEBUG_ONLY" in decision.reason


def test_case_label_is_not_a_decision_input():
    signature = inspect.signature(decide_completion)
    assert "case_label" not in signature.parameters
    empty = _decision(_internal_gap(), assessment="incomplete_fish")
    labelled = _decision(_internal_gap(), assessment="incomplete_fish")
    assert empty.as_dict() == labelled.as_dict()


def test_auto_completion_mask_and_occluder_semantics_are_separate():
    decision = _decision(_internal_gap(), assessment="incomplete_fish")
    assert int((decision.completion_mask & decision.clean_visible_mask).sum()) == 0
    assert int(decision.occluder_mask.sum()) == 0
    assert decision.as_dict()["visible_overlap_pixels"] == 0


def test_stats_allows_more_than_two_regions_and_twenty_percent():
    raw = _fish()
    completion = np.zeros_like(raw)
    completion[20:25, 35:40] = True
    completion[20:25, 80:85] = True
    completion[20:25, 130:135] = True
    stats = _stats(raw, np.zeros_like(raw), np.zeros_like(raw), np.zeros_like(raw), completion, mode=AUTO_COMPLETION, decision_confidence=0.90)
    assert stats["completion_region_count"] == 3
    assert stats["completion_percent"] > 0
    assert stats["execution_allowed"]
    assert stats["mask_geometrically_valid"]


def test_stats_rejects_auto_visible_overlap_but_not_ratio():
    raw = _fish()
    completion = np.zeros_like(raw)
    completion[50:55, 100:110] = True
    stats = _stats(raw, np.zeros_like(raw), np.zeros_like(raw), np.zeros_like(raw), completion, mode=AUTO_COMPLETION, decision_confidence=0.90)
    assert stats["visible_overlap_pixels"] == int(completion.sum())
    assert not stats["mask_geometrically_valid"]
    assert not stats["execution_allowed"]


def test_stats_zero_ratio_is_not_required():
    raw = _fish()
    stats = _stats(raw, np.zeros_like(raw), np.zeros_like(raw), np.zeros_like(raw), np.zeros_like(raw), mode=AUTO_COMPLETION)
    assert stats["completion_level"] == "NONE"
    assert stats["execution_reason"] == "NOT_REQUIRED"
