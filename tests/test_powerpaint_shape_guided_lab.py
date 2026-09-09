import inspect

import numpy as np

import app.powerpaint_shape_guided_lab as lab


def test_shape_guided_route_and_contract():
    paths = {route.path for route in lab.router.routes}
    assert "/debug/powerpaint-shape-guided-lab" in paths
    assert "/api/debug/powerpaint-shape-guided-lab/run" in paths
    assert lab.TASK_MODE == "SHAPE_GUIDED"
    assert lab.FITTING_DEGREES == (0.6, 0.8, 0.95)
    assert lab.PROMPT_ID == "FIXED_FISH_SHAPE_GUIDED_V0.3"


def test_completion_mask_is_disjoint_and_capped():
    visible = np.zeros((40, 80), dtype=bool)
    visible[10:30, 10:70] = True
    visible[18:23, 38:45] = False
    mask = lab.build_completion_mask(visible)
    validation = lab.validate_completion_mask(mask, visible)
    assert validation["valid"] is True
    assert validation["visible_overlap_pixels"] == 0
    assert validation["completion_area_ratio"] <= 0.20


def test_fitting_degree_validation_and_report_schema():
    assert lab.normalize_fitting_degrees("0.95,0.6,0.95") == [0.6, 0.95]
    try:
        lab.normalize_fitting_degrees([0.7])
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported fitting degree must fail")
    report = lab.shape_guided_report_entry(
        fitting_degree=0.95,
        visible_pixel_change_ratio=0.0,
        completion_area_ratio=0.12,
        generated_area_pixels=123,
        status="SUCCESS",
    )
    assert set(report) == {
        "task_mode", "fitting_degree", "visible_pixel_change_ratio",
        "completion_area_ratio", "generated_area_pixels", "fish_identity_check",
        "background_change", "status",
    }
    assert report["task_mode"] == "SHAPE_GUIDED"


def test_shape_guided_module_is_independent():
    source = inspect.getsource(lab)
    assert "completion_decision" not in source
    assert "fish_completion_lab" not in source
    assert "completion_worker_client" not in source
