from __future__ import annotations

import random
from typing import Any, Callable

import numpy as np


SUPPORTED_QUARTER_TURNS = (90, 180, 270)


def rotate_image_and_xyxy_targets(
    image: np.ndarray,
    targets: np.ndarray,
    degrees: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a BGR image and pixel xyxy+class targets by an exact quarter-turn."""
    if degrees not in SUPPORTED_QUARTER_TURNS:
        raise ValueError(f"unsupported discrete rotation: {degrees}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected HxWx3 image")
    if targets.ndim != 2 or targets.shape[1] < 5:
        raise ValueError("expected Nx5+ targets in xyxy+class format")

    height, width = image.shape[:2]
    k = {90: 3, 180: 2, 270: 1}[degrees]  # numpy uses counter-clockwise turns
    rotated_image = np.ascontiguousarray(np.rot90(image, k=k))
    rotated_targets = targets.copy()
    if len(rotated_targets) == 0:
        return rotated_image, rotated_targets

    x1, y1, x2, y2 = (targets[:, index].copy() for index in range(4))
    if degrees == 90:  # clockwise
        rotated_targets[:, 0] = height - y2
        rotated_targets[:, 1] = x1
        rotated_targets[:, 2] = height - y1
        rotated_targets[:, 3] = x2
    elif degrees == 180:
        rotated_targets[:, 0] = width - x2
        rotated_targets[:, 1] = height - y2
        rotated_targets[:, 2] = width - x1
        rotated_targets[:, 3] = height - y1
    else:  # counter-clockwise
        rotated_targets[:, 0] = y1
        rotated_targets[:, 1] = width - x2
        rotated_targets[:, 2] = y2
        rotated_targets[:, 3] = width - x1
    return rotated_image, rotated_targets


class DiscreteQuarterTurnTrainTransform:
    """Apply explicit 90/180/270-degree train augmentation before YOLOX transforms."""

    def __init__(
        self,
        base_transform: Callable,
        probability: float = 0.75,
        rng: Any = random,
    ) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        self.base_transform = base_transform
        self.probability = probability
        self.rng = rng

    def __call__(self, image: np.ndarray, targets: np.ndarray, input_dim: tuple[int, int]):
        if self.rng.random() < self.probability:
            degrees = self.rng.choice(SUPPORTED_QUARTER_TURNS)
            image, targets = rotate_image_and_xyxy_targets(image, targets, degrees)
        return self.base_transform(image, targets, input_dim)
