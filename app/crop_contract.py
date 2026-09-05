"""Canonical crop contract shared by review preview and training dataset build."""

from __future__ import annotations

import math
from dataclasses import dataclass
from io import BytesIO
from typing import Sequence

from PIL import Image, ImageOps

DEFAULT_EXPAND_RATIO = 0.15
JPEG_QUALITY = 92


@dataclass(frozen=True)
class CanonicalCrop:
    jpeg_bytes: bytes
    pixel_box: tuple[int, int, int, int]
    width: int
    height: int


def _normalized_bbox(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError("accepted bbox must contain [x, y, width, height]")
    try:
        x, y, width, height = (float(value) for value in bbox)
    except (TypeError, ValueError) as exc:
        raise ValueError("accepted bbox must contain numeric values") from exc
    values = (x, y, width, height)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("accepted bbox values must be finite")
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("accepted bbox must be positive and normalized")
    if x + width > 1.00001 or y + height > 1.00001:
        raise ValueError("accepted bbox must stay within normalized image bounds")
    return x, y, width, height


def crop_pixel_box(
    bbox: Sequence[float],
    image_width: int,
    image_height: int,
    *,
    expand_ratio: float = DEFAULT_EXPAND_RATIO,
) -> tuple[int, int, int, int]:
    """Convert a normalized accepted bbox into the canonical integer pixel box."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("source image dimensions must be positive")
    if not math.isfinite(expand_ratio) or not 0.0 <= expand_ratio <= 1.0:
        raise ValueError("expand_ratio must be between 0 and 1")

    x, y, width, height = _normalized_bbox(bbox)
    x1 = max(0.0, x - width * expand_ratio)
    y1 = max(0.0, y - height * expand_ratio)
    x2 = min(1.0, x + width + width * expand_ratio)
    y2 = min(1.0, y + height + height * expand_ratio)

    left = max(0, min(image_width - 1, math.floor(x1 * image_width)))
    top = max(0, min(image_height - 1, math.floor(y1 * image_height)))
    right = max(left + 1, min(image_width, math.ceil(x2 * image_width)))
    bottom = max(top + 1, min(image_height, math.ceil(y2 * image_height)))
    return left, top, right, bottom


def canonical_crop(
    data: bytes,
    bbox: Sequence[float],
    *,
    expand_ratio: float = DEFAULT_EXPAND_RATIO,
    jpeg_quality: int = JPEG_QUALITY,
) -> CanonicalCrop:
    """Return the one canonical crop artifact for an accepted normalized bbox.

    Contract order is fixed: decode -> EXIF transpose -> RGB -> expand bbox ->
    floor left/top + ceil right/bottom -> clamp -> crop -> JPEG quality 92.
    Both review preview and training must consume this function's output.
    """

    if jpeg_quality != JPEG_QUALITY:
        raise ValueError(f"canonical JPEG quality must remain {JPEG_QUALITY}")

    with Image.open(BytesIO(data)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        pixel_box = crop_pixel_box(
            bbox,
            image.width,
            image.height,
            expand_ratio=expand_ratio,
        )
        crop = image.crop(pixel_box)
        output = BytesIO()
        crop.save(output, format="JPEG", quality=JPEG_QUALITY)
        width, height = crop.size

    return CanonicalCrop(
        jpeg_bytes=output.getvalue(),
        pixel_box=pixel_box,
        width=width,
        height=height,
    )


__all__ = [
    "CanonicalCrop",
    "DEFAULT_EXPAND_RATIO",
    "JPEG_QUALITY",
    "canonical_crop",
    "crop_pixel_box",
]
