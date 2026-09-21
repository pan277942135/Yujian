from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


ALPHA_THRESHOLD = 16
MAX_LONG_EDGE = 1600
PADDING_RATIO = 0.08


class BsideVisualError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ImageArtifact:
    data: bytes
    metadata: dict[str, Any]
    media_type: str = "image/png"


def _open_source(data: bytes) -> Image.Image:
    try:
        source = Image.open(io.BytesIO(data))
        bands = source.getbands()
        has_alpha = "A" in bands or "transparency" in source.info or source.mode in {"LA", "PA", "RGBA"}
        if not has_alpha:
            raise BsideVisualError("INVALID_TRANSPARENT_FISH", "源图片没有透明 Alpha 通道")
        image = source.convert("RGBA")
        source.close()
    except BsideVisualError:
        raise
    except Exception as exc:
        raise BsideVisualError("INVALID_TRANSPARENT_FISH", "源图片不是可读取的透明 PNG") from exc
    alpha = np.asarray(image, dtype=np.uint8)[:, :, 3]
    if int(np.count_nonzero(alpha >= ALPHA_THRESHOLD)) == 0:
        raise BsideVisualError("INVALID_TRANSPARENT_FISH", "源图片没有有效的非透明鱼体像素")
    return image


def validate_transparent_fish(data: bytes) -> dict[str, Any]:
    image = _open_source(data)
    alpha = np.asarray(image, dtype=np.uint8)[:, :, 3]
    return {
        "width": int(image.width),
        "height": int(image.height),
        "has_alpha": True,
        "nontransparent_pixels": int(np.count_nonzero(alpha >= ALPHA_THRESHOLD)),
        "alpha_threshold": ALPHA_THRESHOLD,
    }


def _orientation(mask: np.ndarray) -> tuple[float, float, float, str]:
    """Return auto rotation, axis ratio, confidence and a human message."""

    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    eroded = np.asarray(mask_image.filter(ImageFilter.MinFilter(3)), dtype=np.uint8) >= 128
    if int(eroded.sum()) < max(24, int(mask.sum() * 0.08)):
        eroded = mask
    yx = np.column_stack(np.nonzero(eroded))
    if len(yx) < 8:
        return 0.0, 1.0, 0.0, "有效像素过少，保留原始方向"

    xy = yx[:, [1, 0]].astype(np.float64)
    centered = xy - xy.mean(axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)
    minor = max(float(eigenvalues[order[0]]), 1e-9)
    major = max(float(eigenvalues[order[-1]]), minor)
    axis_ratio = math.sqrt(major / minor)
    confidence = max(0.0, min(1.0, (axis_ratio - 1.0) / 2.0))
    if axis_ratio < 1.15:
        return 0.0, axis_ratio, confidence, "方向置信度较低，未强制旋转"

    vector = eigenvectors[:, order[-1]]
    axis_angle = math.degrees(math.atan2(float(vector[1]), float(vector[0])))
    # PCA has a 180-degree ambiguity.  Normalize the rotation to the smallest
    # correction without flipping the image or changing the head direction.
    axis_angle = ((axis_angle + 90.0) % 180.0) - 90.0
    rotation = 0.0 if abs(axis_angle) < 3.0 else -axis_angle
    return rotation, axis_ratio, confidence, "依据主体主轴完成水平归一化"


def _rotate_premultiplied(image: Image.Image, angle: float) -> Image.Image:
    rgba = np.asarray(image, dtype=np.float32) / 255.0
    alpha = rgba[:, :, 3:4]
    premultiplied = np.clip(rgba[:, :, :3] * alpha, 0.0, 1.0)
    rgb_image = Image.fromarray(np.round(premultiplied * 255.0).astype(np.uint8), mode="RGB")
    alpha_image = image.getchannel("A")
    if abs(angle) >= 1e-6:
        rgb_image = rgb_image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(0, 0, 0))
        alpha_image = alpha_image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=0)
    rgb = np.asarray(rgb_image, dtype=np.float32)
    alpha_array = np.asarray(alpha_image, dtype=np.float32)
    alpha_fraction = alpha_array / 255.0
    restored = np.zeros_like(rgb)
    np.divide(rgb, alpha_fraction[:, :, None], out=restored, where=alpha_fraction[:, :, None] > 1e-5)
    restored = np.clip(restored * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(np.dstack((restored, alpha_array.astype(np.uint8))), mode="RGBA")


def _tight_crop(image: Image.Image) -> Image.Image:
    alpha = np.asarray(image, dtype=np.uint8)[:, :, 3]
    yx = np.column_stack(np.nonzero(alpha >= ALPHA_THRESHOLD))
    if len(yx) == 0:
        raise BsideVisualError("INVALID_TRANSPARENT_FISH", "旋转后没有有效鱼体像素")
    top, left = yx.min(axis=0)
    bottom, right = yx.max(axis=0) + 1
    width = int(right - left)
    height = int(bottom - top)
    padding = max(1, round(max(width, height) * PADDING_RATIO))
    left = max(0, int(left) - padding)
    top = max(0, int(top) - padding)
    right = min(image.width, int(right) + padding)
    bottom = min(image.height, int(bottom) + padding)
    return image.crop((left, top, right, bottom))


def standardize(source_fish: bytes, manual_rotation_offset_deg: float = 0.0) -> ImageArtifact:
    """Standardize a transparent Qwen fish PNG without segmentation or generation."""

    image = _open_source(source_fish)
    source_width, source_height = image.size
    alpha = np.asarray(image, dtype=np.uint8)[:, :, 3]
    mask = alpha >= ALPHA_THRESHOLD
    auto_rotation, axis_ratio, confidence, confidence_message = _orientation(mask)
    offset = float(manual_rotation_offset_deg)
    if not -15.0 <= offset <= 15.0 or abs(offset * 2 - round(offset * 2)) > 1e-6:
        raise BsideVisualError("ROTATION_OFFSET_INVALID", "手动旋转偏移必须在 -15 到 +15 度之间，步进 0.5 度")
    actual_rotation = auto_rotation + offset if confidence >= 0.075 else offset
    if abs(actual_rotation) < 3.0:
        actual_rotation = 0.0
    rotated = _rotate_premultiplied(image, actual_rotation)
    cropped = _tight_crop(rotated)
    pre_resize_width, pre_resize_height = cropped.size
    long_edge = max(cropped.size)
    if long_edge > MAX_LONG_EDGE:
        scale = MAX_LONG_EDGE / long_edge
        cropped = cropped.resize(
            (max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale))),
            Image.Resampling.LANCZOS,
        )
    output = io.BytesIO()
    cropped.save(output, format="PNG", optimize=True)
    metadata = {
        "rotation_deg": round(actual_rotation, 3),
        "auto_rotation_deg": round(auto_rotation, 3),
        "manual_rotation_offset_deg": round(offset, 3),
        "axis_ratio": round(axis_ratio, 4),
        "orientation_confidence": round(confidence, 4),
        "orientation_message": confidence_message,
        "source_width": source_width,
        "source_height": source_height,
        "pre_resize_width": pre_resize_width,
        "pre_resize_height": pre_resize_height,
        "output_width": cropped.width,
        "output_height": cropped.height,
        "alpha_threshold": ALPHA_THRESHOLD,
        "direction_flipped": False,
    }
    return ImageArtifact(output.getvalue(), metadata)
