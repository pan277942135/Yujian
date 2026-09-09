import inspect

import numpy as np

from app import powerpaint_direct_lab as direct
from app.recognition_pipeline import BBox


def test_direct_lab_uses_v02_prompt_and_isolated_scope():
    assert direct.DIRECT_VERSION == "POWERPAINT_DIRECT_V0.2"
    assert direct.PROMPT_ID == direct.DIRECT_VERSION
    assert "Reconstruct one complete realistic fish" in direct.DIRECT_PROMPT
    assert "Do not change the fish species." in direct.DIRECT_PROMPT
    assert "case_label" not in inspect.getsource(direct)
    assert "completion_decision" not in inspect.getsource(direct)
    assert "fish_completion_lab" not in inspect.getsource(direct)


def test_protect_visible_mask_is_full_frame_non_empty_and_disjoint():
    visible = np.zeros((100, 160), dtype=bool)
    visible[35:65, 50:110] = True
    edit = direct.build_worker_edit_mask(visible, BBox(0.31, 0.35, 0.69, 0.65), "SAM_PROTECT_VISIBLE")
    assert edit.shape == visible.shape
    assert edit.any()
    assert not edit.all()
    assert int(np.logical_and(edit, visible).sum()) == 0


def test_repaint_visible_mask_is_raw_sam_mask():
    visible = np.zeros((20, 30), dtype=bool)
    visible[5:12, 7:19] = True
    edit = direct.build_worker_edit_mask(visible, BBox(0.2, 0.2, 0.8, 0.8), "SAM_REPAINT_VISIBLE")
    assert np.array_equal(edit, visible)


def test_prepare_state_uri_is_persistent_not_process_memory():
    assert "DIRECT_TASKS" not in inspect.getsource(direct)
    assert direct._state_uri("PPD_TEST").endswith("PPD_TEST/prepare.json")


def test_progress_contains_only_direct_stages():
    report = {
        "input": {"filename": "IMG00103"},
        "detector": {"model": "DET_FISH_v0.1"},
        "sam": {"quality": "GOOD"},
        "direct_generation": {"mask_strategy": "SAM_PROTECT_VISIBLE"},
        "worker": {"status": "WORKER_EXECUTED", "result_uri": "gs://bucket/result.png"},
        "timings": {"input_decode_ms": 1, "detector_ms": 2, "sam_ms": 3, "worker_ms": 4},
    }
    stages = direct._progress(report)
    assert [stage["stage"] for stage in stages] == ["input", "detector", "sam", "worker_edit_mask", "powerpaint"]
    assert stages[-1]["status"] == "WORKER_EXECUTED"

