import numpy as np

from app.completion_decision import AUTO_COMPLETION, decide_completion


def _mask():
    value = np.zeros((100, 140), dtype=bool)
    value[30:70, 30:110] = True
    return value


def test_complete_fish_is_not_required():
    decision = decide_completion(_mask(), (30, 30, 110, 70), (100, 140), case_label="case1 complete")
    assert decision.mode == AUTO_COMPLETION
    assert decision.status == "NOT_REQUIRED"
    assert not decision.completion_required
    assert not decision.completion_mask.any()


def test_light_missing_generates_auto_mask():
    decision = decide_completion(_mask(), (20, 20, 120, 80), (100, 140), case_label="case2 tail missing")
    assert decision.status == "MASK_READY"
    assert decision.completion_required
    assert decision.completion_mask.any()
    assert decision.completion_ratio <= 0.20


def test_hand_occlusion_generates_auto_mask_and_occluder():
    decision = decide_completion(_mask(), (20, 20, 120, 80), (100, 140), case_label="case3 hand occlusion")
    assert decision.status == "MASK_READY"
    assert decision.occlusion_detected
    assert decision.occluder_mask.any()
    assert np.all(decision.completion_mask <= decision.occluder_mask)


def test_severe_missing_is_blocked():
    mask = np.zeros((100, 140), dtype=bool)
    mask[45:55, 30:60] = True
    decision = decide_completion(mask, (20, 20, 120, 80), (100, 140), case_label="case4 severe")
    assert decision.status == "NOT_ELIGIBLE"
    assert not decision.completion_required


def test_ambiguous_input_does_not_invent_completion():
    decision = decide_completion(_mask(), (20, 20, 120, 80), (100, 140), case_label="case5 complex background")
    assert decision.status == "REVIEW_REQUIRED"
    assert not decision.completion_required
    assert not decision.completion_mask.any()
