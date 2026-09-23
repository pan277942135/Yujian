from __future__ import annotations

import io
import math
from typing import Any

import numpy as np
from PIL import Image, ImageColor, ImageFilter

from .standardizer import BsideVisualError, ImageArtifact
from .style_registry import OutlineStyle


def _rgba_over(base: np.ndarray, layer: np.ndarray) -> np.ndarray:
    base_alpha = base[:, :, 3:4] / 255.0
    layer_alpha = layer[:, :, 3:4] / 255.0
    output_alpha = layer_alpha + base_alpha * (1.0 - layer_alpha)
    output_rgb = np.zeros_like(base[:, :, :3], dtype=np.float32)
    numerator = layer[:, :, :3] * layer_alpha + base[:, :, :3] * base_alpha * (1.0 - layer_alpha)
    np.divide(numerator, output_alpha, out=output_rgb, where=output_alpha > 1e-6)
    return np.dstack((np.clip(output_rgb, 0, 255), np.clip(output_alpha * 255.0, 0, 255)))


def _color_layer(color: str, alpha: np.ndarray) -> np.ndarray:
    rgb = np.asarray(ImageColor.getrgb(color), dtype=np.float32)
    layer = np.zeros((alpha.shape[0], alpha.shape[1], 4), dtype=np.float32)
    layer[:, :, :3] = rgb
    layer[:, :, 3] = alpha
    return layer


def _direction_angle(direction: str) -> float:
    """Return a screen-space angle for the small V1 rim-light vocabulary."""

    vectors = {
        "UPPER_LEFT": (-1.0, -1.0),
        "UPPER_RIGHT": (1.0, -1.0),
        "LOWER_LEFT": (-1.0, 1.0),
        "LOWER_RIGHT": (1.0, 1.0),
        "LEFT": (-1.0, 0.0),
        "RIGHT": (1.0, 0.0),
        "TOP": (0.0, -1.0),
        "BOTTOM_CENTER": (0.0, 1.0),
    }
    dx, dy = vectors.get(str(direction or "").upper(), vectors["UPPER_LEFT"])
    return math.atan2(dy, dx)


def _break_pattern(x: np.ndarray, y: np.ndarray, period: float) -> np.ndarray:
    """Create stable, sparse gaps so a rim never reads as a closed stroke."""

    cell = max(2.0, float(period))
    sectors = np.floor((x * 0.83 + y * 0.37) / cell).astype(np.int32)
    return (sectors % 9) != 0


def _effect_mask(edge_alpha: np.ndarray, alpha: np.ndarray, style: OutlineStyle) -> np.ndarray:
    """Limit the expanded-alpha effect to the selected local light region."""

    edge = np.asarray(edge_alpha, dtype=np.float32)
    if style.mode == "none" or style.opacity <= 0.0:
        return np.zeros_like(edge)
    edge_pixels = edge > 0.0
    if style.mode == "surrounding" or style.coverage_ratio >= 0.999:
        return edge

    y, x = np.indices(edge.shape, dtype=np.float32)
    visible = np.asarray(alpha, dtype=np.float32) > 0.0
    points = np.column_stack(np.nonzero(visible))
    if len(points) == 0:
        return np.zeros_like(edge)
    min_y, min_x = points.min(axis=0).astype(np.float32)
    max_y, max_x = points.max(axis=0).astype(np.float32)
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    width = max(1.0, max_x - min_x)
    height = max(1.0, max_y - min_y)

    if style.mode == "directional_rim":
        # Normalize the ellipse before calculating the angle so wide fish do
        # not turn an upper-left light into a mostly horizontal band.
        angles = np.arctan2((y - center_y) / height, (x - center_x) / width)
        direction = _direction_angle(style.light_direction)
        delta = np.arctan2(np.sin(angles - direction), np.cos(angles - direction))
        # The profile ratio describes the desired visible rim coverage. The
        # ellipse-angle projection covers less than that on a wide fish, so a
        # small deterministic expansion keeps the rendered result in the
        # requested 25–40% range while the break pattern preserves gaps.
        coverage = min(0.55, max(0.28, float(style.coverage_ratio or 0.32) * 1.45))
        in_arc = np.abs(delta) <= math.pi * coverage
        return np.where(
            edge_pixels & in_arc & _break_pattern(x, y, max(4.0, width * 0.035)),
            edge,
            0.0,
        )

    if style.mode == "bottom_water_glow":
        edge_y = y[edge_pixels]
        # A quantile makes the coverage proportional to the actual fish shape
        # and includes the lower jaw, belly, fins, and tail rather than using
        # a code-specific rectangle.
        quantile = min(0.72, max(0.55, 1.0 - float(style.coverage_ratio or 0.28) * 1.4))
        threshold = float(np.quantile(edge_y, quantile)) if len(edge_y) else max_y
        lower = np.clip((y - threshold) / max(1.0, max_y - threshold), 0.0, 1.0)
        mask = edge * (0.55 + 0.45 * lower)
        mask = np.where(
            edge_pixels & (y >= threshold) & _break_pattern(x, y, max(4.0, width * 0.05)),
            mask,
            0.0,
        )
        return mask

    return edge


def outline(standardized_fish: bytes, style: OutlineStyle) -> ImageArtifact:
    """Render a deterministic outline/glow while keeping source RGB pixels on top."""

    try:
        source = Image.open(io.BytesIO(standardized_fish)).convert("RGBA")
    except Exception as exc:
        raise BsideVisualError("STANDARDIZED_FISH_UNREADABLE", "标准姿态鱼图片不可读取") from exc
    rgba = np.asarray(source, dtype=np.uint8)
    alpha = rgba[:, :, 3]
    if int(np.count_nonzero(alpha)) == 0:
        raise BsideVisualError("STANDARDIZED_FISH_EMPTY", "标准姿态鱼没有有效 Alpha")

    actual_long_edge = max(source.size)
    scale = actual_long_edge / 1600.0
    outline_width = max(1, round(style.base_outline_px * scale))
    max_filter_size = max(3, outline_width * 2 + 1)
    if max_filter_size % 2 == 0:
        max_filter_size += 1
    dilated = np.asarray(
        Image.fromarray(alpha, mode="L").filter(ImageFilter.MaxFilter(max_filter_size)),
        dtype=np.float32,
    )
    edge_alpha = np.clip(dilated - alpha.astype(np.float32), 0.0, 255.0)
    blur_radius = max(0.5, float(style.blur_px) * scale)
    effect_alpha = _effect_mask(edge_alpha, alpha, style)
    glow_alpha = np.asarray(
        Image.fromarray(np.clip(effect_alpha, 0.0, 255.0).astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(blur_radius)),
        dtype=np.float32,
    ) * float(style.glow_opacity)
    solid_alpha = effect_alpha * float(style.opacity)
    base = np.zeros((source.height, source.width, 4), dtype=np.float32)
    base = _rgba_over(base, _color_layer(style.color, glow_alpha))
    base = _rgba_over(base, _color_layer(style.color, solid_alpha))
    # Keep every source fish RGBA value untouched at its original pixel. This
    # makes the outline a surrounding effect, never a recolor or regeneration.
    base[alpha > 0] = rgba[alpha > 0]
    output = io.BytesIO()
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGBA").save(output, format="PNG", optimize=True)
    metadata: dict[str, Any] = {
        "style_id": style.style_id,
        "style_color": style.color,
        "outline_width_px": outline_width,
        "blur_px": round(blur_radius, 3),
        "opacity": style.opacity,
        "glow_opacity": style.glow_opacity,
        "outline_mode": style.mode,
        "coverage_ratio": style.coverage_ratio,
        "light_direction": style.light_direction,
        "effect_edge_ratio": round(
            float(np.count_nonzero(effect_alpha > 0.0))
            / max(1, int(np.count_nonzero(edge_alpha > 0.0))),
            4,
        ),
        "width": source.width,
        "height": source.height,
        "source_rgb_preserved": True,
    }
    return ImageArtifact(output.getvalue(), metadata)
