from app.recognition_pipeline import (
    BBox,
    CROP_EDGE_NEAR,
    Detection,
    PipelineStatus,
    SOURCE_EDGE_NEAR,
    assess_detections,
    crop_box_pixels,
)


def det(conf: float, x1: float, y1: float, x2: float, y2: float) -> Detection:
    return Detection(confidence=conf, box=BBox(x1, y1, x2, y2))


def test_no_fish_when_nothing_above_weak_threshold():
    result = assess_detections([])
    assert result.status == PipelineStatus.NO_FISH
    assert result.primary is None


def test_weak_detection_is_uncertain_not_no_fish():
    result = assess_detections([det(0.25, 0.2, 0.2, 0.8, 0.8)])
    assert result.status == PipelineStatus.UNCERTAIN
    assert result.primary is not None


def test_single_complete_fish_is_ready_and_expanded():
    result = assess_detections([det(0.92, 0.2, 0.25, 0.8, 0.75)])
    assert result.status == PipelineStatus.READY
    assert result.crop_box is not None
    assert result.crop_box.x1 < 0.2
    assert result.crop_box.y1 < 0.25
    assert result.crop_box.x2 > 0.8
    assert result.crop_box.y2 > 0.75


def test_touching_image_edge_is_non_blocking_boundary_warning():
    result = assess_detections([det(0.93, 0.0, 0.2, 0.75, 0.8)])
    assert result.status == PipelineStatus.READY
    assert result.crop_box is not None
    assert result.boundary.source_edge_near is True
    assert result.boundary.crop_edge_near is True
    assert result.boundary.reason == SOURCE_EDGE_NEAR
    assert result.boundary.hard_block is False


def test_crop_edge_warning_is_distinct_from_source_edge_warning():
    # The detector box is just inside the source margin, while the expanded
    # classifier crop reaches the source raster edge.
    result = assess_detections([det(0.93, 0.03, 0.2, 0.75, 0.8)])
    assert result.status == PipelineStatus.READY
    assert result.boundary.source_edge_near is False
    assert result.boundary.crop_edge_near is True
    assert result.boundary.reason == CROP_EDGE_NEAR


def test_normal_bbox_is_good_and_classifier_allowed():
    result = assess_detections([det(0.93, 0.2, 0.25, 0.8, 0.75)])
    assert result.status == PipelineStatus.READY
    assert result.boundary.source_edge_near is False
    assert result.boundary.crop_edge_near is False
    assert result.boundary.reason is None


def test_invalid_bbox_geometry_is_a_hard_failure():
    result = assess_detections([det(0.93, 0.7, 0.2, 0.7, 0.8)])
    assert result.status == PipelineStatus.INVALID_BBOX
    assert result.crop_box is None
    assert result.boundary.hard_block is False


def test_small_fish_is_rejected_before_classifier():
    result = assess_detections([det(0.91, 0.40, 0.40, 0.58, 0.58)])
    assert result.status == PipelineStatus.FISH_TOO_SMALL


def test_multiple_strong_fish_is_explicit_status():
    result = assess_detections([
        det(0.95, 0.1, 0.2, 0.45, 0.7),
        det(0.88, 0.55, 0.2, 0.9, 0.7),
    ])
    assert result.status == PipelineStatus.MULTIPLE_FISH
    assert len(result.strong_detections) == 2


def test_crop_pixel_rounding_contract_is_floor_left_ceil_right():
    assert crop_box_pixels(BBox(0.101, 0.201, 0.799, 0.899), 100, 200) == (10, 40, 80, 180)
