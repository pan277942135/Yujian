from app.recognition_pipeline import BBox, Detection, PipelineStatus, assess_detections, crop_box_pixels


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


def test_touching_image_edge_is_incomplete():
    result = assess_detections([det(0.93, 0.0, 0.2, 0.75, 0.8)])
    assert result.status == PipelineStatus.INCOMPLETE_FISH
    assert result.crop_box is None


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
