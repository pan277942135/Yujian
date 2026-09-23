from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


ALPHA_THRESHOLD = 16
MAX_LONG_EDGE = 1600
PADDING_RATIO = 0.08
RESIDUAL_PASS_DEG = 2.0
RESIDUAL_WARN_DEG = 4.0
MIN_FOREGROUND_PIXELS = 8
HEAD_DIRECTION_CONFIDENCE_PASS = 0.75
HEAD_ENDPOINT_RATIO = 0.22
HEAD_MIN_AXIS_PIXELS = 20
ORIENTATION_CONFIDENCE_PASS = 0.75
ORIENTATION_BODY_START_RATIO = 0.18
ORIENTATION_BODY_END_RATIO = 0.82
ORIENTATION_MIN_COLUMNS = 16


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
        raise BsideVisualError("POSE_ALPHA_EMPTY", "源图片 Alpha 没有有效鱼体像素")
    if int(np.count_nonzero(alpha >= ALPHA_THRESHOLD)) < MIN_FOREGROUND_PIXELS:
        raise BsideVisualError("POSE_FOREGROUND_TOO_SMALL", "源图片有效鱼体像素过少")
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
        "mode": "RGBA",
        "format": "PNG",
    }


def _normalize_axis_angle(angle_deg: float) -> float:
    """Normalize an undirected PCA axis to [-90, 90)."""

    return ((float(angle_deg) + 90.0) % 180.0) - 90.0


def _analysis_mask(alpha: np.ndarray) -> np.ndarray:
    """Build an analysis-only mask from Alpha without changing formal RGB."""

    mask = np.asarray(alpha, dtype=np.uint8) >= ALPHA_THRESHOLD
    foreground_count = int(mask.sum())
    if foreground_count == 0:
        raise BsideVisualError("POSE_ALPHA_EMPTY", "源图片 Alpha 没有有效鱼体像素")
    if foreground_count < MIN_FOREGROUND_PIXELS:
        raise BsideVisualError("POSE_FOREGROUND_TOO_SMALL", "源图片有效鱼体像素过少")
    # The Qwen transparent asset already has its segmentation mask applied.
    # Keep the thresholded Alpha intact so thin fins and tails participate in
    # the axis calculation; this array is never used to render the output.
    return mask


def _principal_axis(mask: np.ndarray) -> tuple[float, float, float, tuple[float, float]]:
    """Return axis angle, anisotropy, confidence and Alpha centroid."""

    yx = np.column_stack(np.nonzero(np.asarray(mask, dtype=bool)))
    if len(yx) < MIN_FOREGROUND_PIXELS:
        raise BsideVisualError("POSE_FOREGROUND_TOO_SMALL", "姿态分析前景像素过少")

    xy = yx[:, [1, 0]].astype(np.float64)
    centroid = (float(xy[:, 0].mean()), float(xy[:, 1].mean()))
    centered = xy - np.asarray(centroid, dtype=np.float64)
    covariance = np.cov(centered, rowvar=False, bias=True)
    if np.asarray(covariance).shape != (2, 2) or not np.all(np.isfinite(covariance)):
        raise BsideVisualError("POSE_AXIS_DETECTION_FAILED", "无法计算鱼体主轴协方差")

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not np.all(np.isfinite(eigenvalues)) or not np.all(np.isfinite(eigenvectors)):
        raise BsideVisualError("POSE_AXIS_DETECTION_FAILED", "鱼体主轴计算结果无效")

    order = np.argsort(eigenvalues)
    minor = max(float(eigenvalues[order[0]]), 1e-9)
    major = max(float(eigenvalues[order[-1]]), minor)
    axis_ratio = math.sqrt(major / minor)
    if not math.isfinite(axis_ratio) or axis_ratio <= 0:
        raise BsideVisualError("POSE_AXIS_DETECTION_FAILED", "鱼体主轴比例无效")

    vector = eigenvectors[:, order[-1]]
    detected_angle = _normalize_axis_angle(
        math.degrees(math.atan2(float(vector[1]), float(vector[0])))
    )
    confidence = max(0.0, min(1.0, (axis_ratio - 1.0) / 2.0))
    return detected_angle, axis_ratio, confidence, centroid


def _foreground_metrics(image: Image.Image) -> dict[str, Any]:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    alpha = rgba[:, :, 3]
    foreground = rgba[alpha >= ALPHA_THRESHOLD, :3].astype(np.float64)
    if len(foreground) == 0:
        raise BsideVisualError("POSE_ALPHA_EMPTY", "输出没有有效鱼体像素")
    channel_variance = np.var(foreground, axis=0)
    return {
        "alpha_min": int(alpha.min()),
        "alpha_max": int(alpha.max()),
        "alpha_coverage": round(float(np.count_nonzero(alpha)) / alpha.size, 6),
        "foreground_ratio": round(float(np.count_nonzero(alpha >= ALPHA_THRESHOLD)) / alpha.size, 6),
        "foreground_rgb_mean": [round(float(value), 3) for value in foreground.mean(axis=0)],
        "foreground_rgb_variance": round(float(channel_variance.mean()), 6),
    }


def _rotate_premultiplied(
    image: Image.Image,
    angle: float,
    center: tuple[float, float],
) -> Image.Image:
    """Rotate RGB and Alpha together while avoiding transparent-edge color bleed."""

    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    alpha_fraction = rgba[:, :, 3:4]
    premultiplied = np.clip(rgba[:, :, :3] * alpha_fraction, 0.0, 1.0)
    rgb_image = Image.fromarray(
        np.round(premultiplied * 255.0).astype(np.uint8),
        mode="RGB",
    )
    alpha_image = image.convert("RGBA").getchannel("A")

    if abs(angle) >= 1e-6:
        rotate_kwargs = {
            "resample": Image.Resampling.BICUBIC,
            "expand": True,
            "fillcolor": (0, 0, 0),
            "center": center,
        }
        rgb_image = rgb_image.rotate(angle, **rotate_kwargs)
        alpha_image = alpha_image.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=0,
            center=center,
        )

    rotated_rgb = np.asarray(rgb_image, dtype=np.float32)
    rotated_alpha = np.clip(
        np.rint(np.asarray(alpha_image, dtype=np.float32)),
        0.0,
        255.0,
    ).astype(np.uint8)
    rotated_alpha_fraction = rotated_alpha.astype(np.float32) / 255.0
    restored_rgb = np.zeros_like(rotated_rgb)
    np.divide(
        rotated_rgb,
        rotated_alpha_fraction[:, :, None],
        out=restored_rgb,
        where=rotated_alpha_fraction[:, :, None] > 1e-5,
    )
    # rotated_rgb is already in the 0-255 image range because the
    # premultiplied buffer was materialized as uint8 before rotation.
    # Multiplying by 255 here would saturate every foreground pixel to white.
    restored_rgb = np.clip(np.rint(restored_rgb), 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(
        np.dstack((restored_rgb, rotated_alpha)),
        mode="RGBA",
    )


def _tight_crop(image: Image.Image) -> Image.Image:
    alpha = np.asarray(image.convert("RGBA"), dtype=np.uint8)[:, :, 3]
    yx = np.column_stack(np.nonzero(alpha >= ALPHA_THRESHOLD))
    if len(yx) == 0:
        raise BsideVisualError("POSE_RGBA_EXPORT_FAILED", "旋转后没有有效鱼体像素")

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


def _detect_head_direction(mask: np.ndarray) -> dict[str, Any]:
    """Infer head side from the two horizontal Alpha endpoints.

    PCA has already made the fish horizontal at this point, but it does not
    provide an axis direction.  V1 intentionally uses only the existing Alpha
    mask: a fish head is normally both wider and has more mask area than its
    tail endpoint.  Ambiguous geometry is reported as a warning rather than
    forcing an incorrect mirror.
    """

    alpha_mask = np.asarray(mask, dtype=bool)
    yx = np.column_stack(np.nonzero(alpha_mask))
    if len(yx) < MIN_FOREGROUND_PIXELS:
        return {
            "head_side": "unknown",
            "confidence": 0.0,
            "left_score": 0.0,
            "right_score": 0.0,
            "reason": "HEAD_DIRECTION_LOW_CONFIDENCE",
        }

    _top, left = yx.min(axis=0)
    _bottom, right = yx.max(axis=0)
    axis_pixels = int(right - left + 1)
    if axis_pixels < HEAD_MIN_AXIS_PIXELS:
        return {
            "head_side": "unknown",
            "confidence": 0.0,
            "left_score": 0.0,
            "right_score": 0.0,
            "reason": "HEAD_DIRECTION_LOW_CONFIDENCE",
        }

    column_area = alpha_mask[:, int(left) : int(right) + 1].sum(axis=0).astype(np.float64)
    endpoint_width = max(3, int(round(axis_pixels * HEAD_ENDPOINT_RATIO)))
    left_columns = column_area[:endpoint_width]
    right_columns = column_area[-endpoint_width:]
    left_area = float(left_columns.sum())
    right_area = float(right_columns.sum())
    # The upper quartile remains responsive to the broad head while avoiding a
    # single tail-fin tip deciding the direction.
    left_width = float(np.percentile(left_columns, 75))
    right_width = float(np.percentile(right_columns, 75))

    area_total = max(left_area + right_area, 1e-9)
    width_total = max(left_width + right_width, 1e-9)
    left_score = 0.55 * (left_area / area_total) + 0.45 * (left_width / width_total)
    right_score = 0.55 * (right_area / area_total) + 0.45 * (right_width / width_total)
    direction_margin = abs(right_score - left_score)

    # Confidence is deliberately calibrated around a neutral 0.50 baseline:
    # only a clear, consistent endpoint asymmetry crosses the product gate.
    confidence = min(0.99, 0.5 + 0.5 * direction_margin * 2.0)
    if (right_area - left_area) * (right_width - left_width) < 0:
        confidence *= 0.7
    confidence = max(0.0, min(0.99, confidence))

    if confidence < HEAD_DIRECTION_CONFIDENCE_PASS:
        head_side = "unknown"
        reason = "HEAD_DIRECTION_LOW_CONFIDENCE"
    else:
        head_side = "right" if right_score > left_score else "left"
        reason = None
    return {
        "head_side": head_side,
        "confidence": float(confidence),
        "left_score": float(left_score),
        "right_score": float(right_score),
        "reason": reason,
    }


def _detect_natural_orientation(mask: np.ndarray) -> dict[str, Any]:
    """Classify dorsal/ventral orientation from the horizontal Alpha contour.

    This deliberately remains an Alpha-geometry heuristic: no new model and no
    RGB repainting are involved.  Across the central body region, the dorsal
    side tends to have a sharper / less area-dense contour from the back fin,
    while the belly is broader and smoother.  The gate is intentionally
    conservative: ambiguous, cropped, or near-symmetric fish are warnings and
    never receive a speculative 180-degree correction.
    """

    alpha_mask = np.asarray(mask, dtype=bool)
    yx = np.column_stack(np.nonzero(alpha_mask))
    if len(yx) < MIN_FOREGROUND_PIXELS:
        return {
            "orientation": "WARNING",
            "back_side": "unknown",
            "belly_side": "unknown",
            "confidence": 0.0,
            "top_score": 0.0,
            "bottom_score": 0.0,
            "reason": "ORIENTATION_LOW_CONFIDENCE",
        }

    top, left = yx.min(axis=0)
    bottom, right = yx.max(axis=0)
    axis_pixels = int(right - left + 1)
    if axis_pixels < HEAD_MIN_AXIS_PIXELS:
        return {
            "orientation": "WARNING",
            "back_side": "unknown",
            "belly_side": "unknown",
            "confidence": 0.0,
            "top_score": 0.0,
            "bottom_score": 0.0,
            "reason": "ORIENTATION_LOW_CONFIDENCE",
        }

    start = int(left + round(axis_pixels * ORIENTATION_BODY_START_RATIO))
    end = int(left + round(axis_pixels * ORIENTATION_BODY_END_RATIO))
    midline = (float(top) + float(bottom)) / 2.0
    top_depths: list[float] = []
    bottom_depths: list[float] = []
    top_area = 0.0
    bottom_area = 0.0
    for x in range(max(int(left), start), min(int(right) + 1, end)):
        ys = np.flatnonzero(alpha_mask[:, x])
        if len(ys) == 0:
            continue
        top_depths.append(max(0.0, midline - float(ys.min())))
        bottom_depths.append(max(0.0, float(ys.max()) - midline))
        top_area += float(np.count_nonzero(ys < midline))
        bottom_area += float(np.count_nonzero(ys > midline))

    if len(top_depths) < ORIENTATION_MIN_COLUMNS or len(bottom_depths) < ORIENTATION_MIN_COLUMNS:
        return {
            "orientation": "WARNING",
            "back_side": "unknown",
            "belly_side": "unknown",
            "confidence": 0.0,
            "top_score": 0.0,
            "bottom_score": 0.0,
            "reason": "ORIENTATION_LOW_CONFIDENCE",
        }

    def dorsal_likelihood(depths: list[float], side_area: float) -> float:
        values = np.asarray(depths, dtype=np.float64)
        body_height = max(float(bottom - top + 1), 1.0)
        # A back fin produces a local peak and sharper changes than a belly.
        peakiness = max(0.0, float(values.max() - np.percentile(values, 60))) / body_height
        if len(values) >= 3:
            curvature = float(np.mean(np.abs(np.diff(values, n=2)))) / body_height
        else:
            curvature = 0.0
        area_total = max(top_area + bottom_area, 1.0)
        sparse_area = 1.0 - min(1.0, side_area / area_total * 2.0)
        return 0.50 * peakiness + 0.35 * curvature + 0.15 * sparse_area

    top_score = dorsal_likelihood(top_depths, top_area)
    bottom_score = dorsal_likelihood(bottom_depths, bottom_area)
    score_total = max(top_score + bottom_score, 1e-9)
    confidence = min(0.99, 0.5 + 0.5 * abs(top_score - bottom_score) / score_total)
    if confidence < ORIENTATION_CONFIDENCE_PASS:
        return {
            "orientation": "WARNING",
            "back_side": "unknown",
            "belly_side": "unknown",
            "confidence": float(confidence),
            "top_score": float(top_score),
            "bottom_score": float(bottom_score),
            "reason": "ORIENTATION_LOW_CONFIDENCE",
        }

    back_side = "TOP" if top_score > bottom_score else "BOTTOM"
    belly_side = "BOTTOM" if back_side == "TOP" else "TOP"
    return {
        "orientation": "NORMAL" if belly_side == "BOTTOM" else "UPSIDE_DOWN",
        "back_side": back_side,
        "belly_side": belly_side,
        "confidence": float(confidence),
        "top_score": float(top_score),
        "bottom_score": float(bottom_score),
        "reason": None,
    }


def standardize(source_fish: bytes, manual_rotation_offset_deg: float = 0.0) -> ImageArtifact:
    """Rotate the original Qwen RGBA fish into a horizontal CPU-only pose."""

    image = _open_source(source_fish)
    source_width, source_height = image.size
    source_alpha = np.asarray(image, dtype=np.uint8)[:, :, 3]
    analysis_mask = _analysis_mask(source_alpha)
    detected_axis_angle, axis_ratio, confidence, centroid = _principal_axis(analysis_mask)

    offset = float(manual_rotation_offset_deg)
    if not math.isfinite(offset) or not -15.0 <= offset <= 15.0 or abs(offset * 2 - round(offset * 2)) > 1e-6:
        raise BsideVisualError(
            "ROTATION_OFFSET_INVALID",
            "手动旋转偏移必须在 -15 到 +15 度之间，步进 0.5 度",
        )

    # PIL's Image.rotate uses the opposite visual sign from the y-down image
    # coordinate angle returned by atan2. Applying the detected angle (not its
    # negation) is verified below by the residual PCA pass.
    auto_rotation = detected_axis_angle
    applied_rotation = auto_rotation + offset
    rotated = _rotate_premultiplied(image, applied_rotation, centroid)
    cropped = _tight_crop(rotated)

    pre_resize_width, pre_resize_height = cropped.size
    if max(cropped.size) > MAX_LONG_EDGE:
        scale = MAX_LONG_EDGE / max(cropped.size)
        cropped = cropped.resize(
            (
                max(1, round(cropped.width * scale)),
                max(1, round(cropped.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )

    orientation_detection = _detect_natural_orientation(
        _analysis_mask(np.asarray(cropped.convert("RGBA"), dtype=np.uint8)[:, :, 3])
    )
    rotate_180_applied = orientation_detection["orientation"] == "UPSIDE_DOWN"
    if rotate_180_applied:
        # This is a physical half-turn, not a horizontal mirror.  It keeps
        # every RGBA pixel intact and is only used to restore back-up/belly-down.
        cropped = cropped.transpose(Image.Transpose.ROTATE_180)
        orientation_detection = _detect_natural_orientation(
            _analysis_mask(np.asarray(cropped.convert("RGBA"), dtype=np.uint8)[:, :, 3])
        )
        if orientation_detection["orientation"] != "NORMAL":
            raise BsideVisualError("POSE_ORIENTATION_FAILED", "180° 修正后鱼体仍未恢复自然背腹方向")

    head_detection = _detect_head_direction(
        _analysis_mask(np.asarray(cropped.convert("RGBA"), dtype=np.uint8)[:, :, 3])
    )
    head_direction = str(head_detection["head_side"])
    head_confidence = float(head_detection["confidence"])

    source_metrics = _foreground_metrics(image)
    source_mean = np.asarray(source_metrics["foreground_rgb_mean"], dtype=np.float64)
    if (
        source_metrics["foreground_rgb_variance"] < 0.05
        and bool(np.all(source_mean > 250.0))
    ):
        raise BsideVisualError(
            "POSE_RGBA_EXPORT_FAILED",
            "输入透明鱼体前景是纯白 silhouette，拒绝将 Mask 冒充正式鱼体",
        )

    output_metrics = _foreground_metrics(cropped)
    if (
        source_metrics["foreground_rgb_variance"] > 1.0
        and output_metrics["foreground_rgb_variance"] < 0.05
    ):
        raise BsideVisualError(
            "POSE_RGBA_EXPORT_FAILED",
            "姿态标准化后前景 RGB 纹理丢失，拒绝输出白色/纯色 silhouette",
        )

    residual_mask = _analysis_mask(
        np.asarray(cropped.convert("RGBA"), dtype=np.uint8)[:, :, 3]
    )
    residual_axis_angle, _residual_ratio, _residual_confidence, _residual_centroid = _principal_axis(
        residual_mask
    )
    expected_residual = _normalize_axis_angle(-offset)
    residual_error = _normalize_axis_angle(residual_axis_angle - expected_residual)
    absolute_error = abs(residual_error)
    if absolute_error <= RESIDUAL_PASS_DEG:
        pose_validation = "PASS"
    elif absolute_error <= RESIDUAL_WARN_DEG:
        pose_validation = "WARN"
    else:
        raise BsideVisualError(
            "POSE_AXIS_DETECTION_FAILED",
            f"姿态标准化残差 {residual_error:.2f}° 超过 4°",
        )

    pose_warning_reason = orientation_detection["reason"] or head_detection["reason"]
    if pose_validation != "PASS" and not pose_warning_reason:
        pose_warning_reason = "POSE_AXIS_RESIDUAL_WARNING"
    pose_status = "WARNING" if pose_warning_reason else "PASS"

    output = io.BytesIO()
    cropped.save(output, format="PNG", optimize=True)
    metadata: dict[str, Any] = {
        "format": "PNG",
        "mode": "RGBA",
        "channels": 4,
        "source_width": source_width,
        "source_height": source_height,
        "detected_axis_angle_deg": round(detected_axis_angle, 3),
        "pca_angle": round(detected_axis_angle, 3),
        "target_axis_angle_deg": 0.0,
        "auto_rotation_deg": round(auto_rotation, 3),
        "manual_rotation_offset_deg": round(offset, 3),
        "applied_rotation_deg": round(applied_rotation, 3),
        "rotation_applied": round(applied_rotation, 3),
        # Keep the old field for existing consumers while making the new
        # semantics explicit.
        "rotation_deg": round(applied_rotation, 3),
        "residual_axis_angle_deg": round(residual_axis_angle, 3),
        "expected_residual_axis_angle_deg": round(expected_residual, 3),
        "residual_axis_error_deg": round(residual_error, 3),
        "final_axis_error": round(residual_error, 3),
        "pose_validation": pose_validation,
        "pose_status": pose_status,
        "pose_warning_reason": pose_warning_reason,
        "axis_ratio": round(axis_ratio, 4),
        "axis_confidence": round(confidence, 4),
        "orientation_message": (
            "PCA 主轴已自动摆平；已检查并保持鱼背朝上、鱼腹朝下"
            if offset == 0
            else "PCA 主轴已自动摆平并完成自然背腹检查；已叠加人工微调"
        ),
        "source_alpha_min": source_metrics["alpha_min"],
        "source_alpha_max": source_metrics["alpha_max"],
        "source_foreground_rgb_mean": source_metrics["foreground_rgb_mean"],
        "source_foreground_rgb_variance": source_metrics["foreground_rgb_variance"],
        "output_alpha_min": output_metrics["alpha_min"],
        "output_alpha_max": output_metrics["alpha_max"],
        "alpha_min": output_metrics["alpha_min"],
        "alpha_max": output_metrics["alpha_max"],
        "alpha_coverage": output_metrics["alpha_coverage"],
        "foreground_ratio": output_metrics["foreground_ratio"],
        "foreground_rgb_mean": output_metrics["foreground_rgb_mean"],
        "foreground_rgb_variance": output_metrics["foreground_rgb_variance"],
        "rgb_preserved_inside_fish": True,
        "transparent_background": output_metrics["alpha_min"] == 0,
        "expanded_rotation_canvas": True,
        "uniform_scale": True,
        "orientation": str(orientation_detection["orientation"]),
        "back_side": str(orientation_detection["back_side"]),
        "belly_side": str(orientation_detection["belly_side"]),
        "orientation_confidence": round(float(orientation_detection["confidence"]), 4),
        "orientation_top_score": round(float(orientation_detection["top_score"]), 4),
        "orientation_bottom_score": round(float(orientation_detection["bottom_score"]), 4),
        "rotate_180_applied": rotate_180_applied,
        "head_direction": head_direction,
        # Compatibility aliases: V2 records the final actual direction and
        # never changes it merely to force a right-facing fish.
        "head_side_before_flip": head_direction,
        "head_direction_after": head_direction,
        "head_confidence": round(head_confidence, 4),
        "head_left_score": round(float(head_detection["left_score"]), 4),
        "head_right_score": round(float(head_detection["right_score"]), 4),
        "head_detection_method": "alpha_geometry_endpoints",
        "flip_horizontal": False,
        "flip_vertical": False,
        "direction_flipped": False,
        "head_direction_changed": False,
        "pre_resize_width": pre_resize_width,
        "pre_resize_height": pre_resize_height,
        "output_width": cropped.width,
        "output_height": cropped.height,
        "alpha_threshold": ALPHA_THRESHOLD,
    }
    return ImageArtifact(output.getvalue(), metadata)


__all__ = [
    "BsideVisualError",
    "ImageArtifact",
    "_detect_natural_orientation",
    "standardize",
    "validate_transparent_fish",
]
