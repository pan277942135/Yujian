from __future__ import annotations

import io
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
    glow_alpha = np.asarray(
        Image.fromarray(edge_alpha.astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(blur_radius)),
        dtype=np.float32,
    ) * float(style.glow_opacity)
    solid_alpha = edge_alpha * float(style.opacity)
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
        "width": source.width,
        "height": source.height,
        "source_rgb_preserved": True,
    }
    return ImageArtifact(output.getvalue(), metadata)
