import inspect
import io

import numpy as np
from PIL import Image

import app.powerpaint_shape_guided_lab as lab


def test_shape_guided_route_and_contract():
    paths = {route.path for route in lab.router.routes}
    assert "/debug/powerpaint-shape-guided-lab" in paths
    assert "/api/debug/powerpaint-shape-guided-lab/run" in paths
    assert lab.TASK_MODE == "SHAPE_GUIDED"
    assert lab.FITTING_DEGREES == (0.6, 0.8, 0.95)
    assert lab.PROMPT_ID == "FIXED_FISH_SHAPE_GUIDED_V0.3.2"
    assert lab.PROMPT.startswith("Restore the missing part of the same fish.")
    assert lab.NEGATIVE_PROMPT_STATUS == "NEGATIVE_PROMPT_NOT_SUPPORTED"
    assert lab.VISIBLE_FISH_INPUT_TYPE == "RGB_CANVAS"


def test_refined_visible_input_is_rgb_canvas():
    crop = Image.new("RGB", (4, 3), (20, 40, 60))
    visible = np.zeros((3, 4), dtype=bool)
    visible[1, 1:3] = True
    data = lab._visible_fish_input_png(crop, visible)
    with Image.open(io.BytesIO(data)) as image:
        assert image.mode == "RGB"
        assert image.getpixel((0, 0)) == (255, 255, 255)
        assert image.getpixel((1, 1)) == (20, 40, 60)


def test_fish_subject_output_is_rgba_and_transparent():
    crop = Image.new("RGB", (4, 3), (20, 40, 60))
    subject = np.zeros((3, 4), dtype=bool)
    subject[1, 1:3] = True
    data = lab._fish_subject_png(crop, subject)
    with Image.open(io.BytesIO(data)) as image:
        assert image.mode == "RGBA"
        assert image.getpixel((0, 0))[3] == 0
        assert image.getpixel((1, 1)) == (20, 40, 60, 255)


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


def test_powerpaint_request_uses_refined_visible_input():
    source = inspect.getsource(lab.run)
    assert '"image_uri": report["assets"]["refined_visible_fish_input"]' in source
    assert 'image_uri=report["assets"]["refined_visible_fish_input"]' in source
    assert '"mask_uri": report["assets"]["completion_mask"]' in source
    assert '_fish_subject_png(generated_image, subject_mask)' in source
    assert 'FISH_SUBJECT_OUTPUT_TYPE' in source
    assert '_compose(crop, generated_image, completion)' not in source


def test_shape_guided_module_is_independent():
    source = inspect.getsource(lab)
    assert "completion_decision" not in source
    assert "fish_completion_lab" not in source
    assert "completion_worker_client" not in source
