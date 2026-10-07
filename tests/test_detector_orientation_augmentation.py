from __future__ import annotations

import numpy as np
import pytest

from trainer.detector_orientation_augmentation import (
    DiscreteQuarterTurnTrainTransform,
    SUPPORTED_QUARTER_TURNS,
    rotate_image_and_xyxy_targets,
)


@pytest.mark.parametrize(
    ("degrees", "expected_shape", "expected_box"),
    [
        (90, (3, 2, 3), [1.0, 0.0, 2.0, 2.0]),
        (180, (2, 3, 3), [1.0, 1.0, 3.0, 2.0]),
        (270, (3, 2, 3), [0.0, 1.0, 1.0, 3.0]),
    ],
)
def test_discrete_quarter_turn_rotates_pixels_and_xyxy_labels(degrees, expected_shape, expected_box):
    image = np.arange(18, dtype=np.uint8).reshape((2, 3, 3))
    targets = np.array([[0.0, 0.0, 2.0, 1.0, 0.0]], dtype=np.float32)

    rotated_image, rotated_targets = rotate_image_and_xyxy_targets(image, targets, degrees)

    assert rotated_image.shape == expected_shape
    np.testing.assert_allclose(rotated_targets[0, :4], expected_box)
    assert rotated_targets[0, 4] == 0.0
    np.testing.assert_array_equal(targets, [[0.0, 0.0, 2.0, 1.0, 0.0]])


def test_v02_augmentation_explicitly_selects_only_right_angle_turns():
    class Rng:
        def random(self):
            return 0.0

        def choice(self, values):
            assert values == (90, 180, 270)
            return 270

    captured = {}

    def base(image, targets, input_dim):
        captured["shape"] = image.shape
        captured["targets"] = targets
        captured["input_dim"] = input_dim
        return image, targets

    transform = DiscreteQuarterTurnTrainTransform(base, probability=1.0, rng=Rng())
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    targets = np.array([[0.0, 0.0, 2.0, 1.0, 0.0]], dtype=np.float32)

    transform(image, targets, (416, 416))

    assert SUPPORTED_QUARTER_TURNS == (90, 180, 270)
    assert captured["shape"] == (3, 2, 3)
    np.testing.assert_allclose(captured["targets"][0, :4], [0.0, 1.0, 1.0, 3.0])
    assert captured["input_dim"] == (416, 416)


def test_discrete_transform_rejects_non_quarter_turn_angle():
    with pytest.raises(ValueError, match="unsupported discrete rotation"):
        rotate_image_and_xyxy_targets(np.zeros((2, 3, 3)), np.zeros((0, 5)), 15)
