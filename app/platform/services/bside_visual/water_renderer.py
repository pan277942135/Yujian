from __future__ import annotations

import io
import math
from typing import Any

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFilter

from .outline_renderer import outline
from .standardizer import BsideVisualError, ImageArtifact
from .style_registry import OutlineStyle
from .template_registry import WaterTemplate


def _gradient(template: WaterTemplate) -> Image.Image:
    width, height = template.canvas_width, template.canvas_height
    top = np.asarray(ImageColor.getrgb(template.top_color), dtype=np.float32)
    bottom = np.asarray(ImageColor.getrgb(template.bottom_color), dtype=np.float32)
    ratio = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    rows = np.repeat(top[None, None, :] * (1.0 - ratio) + bottom[None, None, :] * ratio, width, axis=1)
    image = Image.fromarray(np.clip(rows, 0, 255).astype(np.uint8), mode="RGB").convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    # Deterministic wave bands. No external image and no random seed are used.
    for index in range(16):
        y = int(height * (0.14 + index * 0.055))
        points = []
        for x in range(-40, width + 50, 24):
            points.append((x, y + int(8 * math.sin(x / 84.0 + index * 0.73))))
        draw.line(points, fill=(235, 248, 244, 18 if index % 3 else 24), width=2)
    for index in range(7):
        y = int(height * (0.30 + index * 0.09))
        draw.line((0, y, width, y + 18), fill=(25, 77, 81, 10), width=1)
    return image


def _canvas_layer(data: bytes, template: WaterTemplate, label: str) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data)).convert("RGBA")
    except Exception as exc:
        raise BsideVisualError("BSIDE_ASSET_UNREADABLE", f"{label} 资产不可读取") from exc
    if image.size != (template.canvas_width, template.canvas_height):
        raise BsideVisualError(
            "BSIDE_ASSET_DIMENSION_INVALID",
            (
                f"{label} 尺寸必须为 {template.canvas_width}×{template.canvas_height}，"
                f"实际为 {image.width}×{image.height}"
            ),
        )
    return image


def _solid_layer(color: str, alpha: np.ndarray) -> np.ndarray:
    rgb = np.asarray(ImageColor.getrgb(color), dtype=np.float32)
    layer = np.zeros((alpha.shape[0], alpha.shape[1], 4), dtype=np.float32)
    layer[:, :, :3] = rgb
    layer[:, :, 3] = alpha
    return layer


def _over_array(base: np.ndarray, layer: np.ndarray) -> np.ndarray:
    base_alpha = base[:, :, 3:4] / 255.0
    layer_alpha = layer[:, :, 3:4] / 255.0
    output_alpha = layer_alpha + base_alpha * (1.0 - layer_alpha)
    numerator = layer[:, :, :3] * layer_alpha + base[:, :, :3] * base_alpha * (1.0 - layer_alpha)
    output_rgb = np.zeros_like(base[:, :, :3], dtype=np.float32)
    np.divide(numerator, output_alpha, out=output_rgb, where=output_alpha > 1e-6)
    return np.dstack((output_rgb, output_alpha * 255.0))


def _fit_fish(image: Image.Image, template: WaterTemplate) -> tuple[Image.Image, float]:
    max_width = max(1, round(template.canvas_width * template.max_width_ratio))
    max_height = max(1, round(template.canvas_height * template.max_height_ratio))
    scale = min(max_width / image.width, max_height / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    return resized, scale


def compose_bside(
    standardized_fish: bytes,
    style: OutlineStyle,
    template: WaterTemplate,
    *,
    outlined_fish: bytes | None = None,
    background_bytes: bytes | None = None,
    foreground_bytes: bytes | None = None,
    light_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Compose a transparent fish into a deterministic water template."""

    source_asset = "outlined_fish_rgba" if outlined_fish is not None else "standardized_fish_rgba"
    fish_bytes = outlined_fish if outlined_fish is not None else standardized_fish
    try:
        fish = Image.open(io.BytesIO(fish_bytes)).convert("RGBA")
    except Exception as exc:
        raise BsideVisualError("STANDARDIZED_FISH_UNREADABLE", "标准姿态鱼图片不可读取") from exc
    fish_alpha = np.asarray(fish, dtype=np.uint8)[:, :, 3]
    if int(np.count_nonzero(fish_alpha)) == 0:
        raise BsideVisualError("STANDARDIZED_FISH_EMPTY", "标准姿态鱼没有有效 Alpha")

    fitted_fish, scale = _fit_fish(fish, template)
    if outlined_fish is None:
        outline_artifact = outline(standardized_fish, style)
        outlined = Image.open(io.BytesIO(outline_artifact.data)).convert("RGBA")
    else:
        outlined = fish.copy()
    outlined = outlined.resize(fitted_fish.size, Image.Resampling.LANCZOS)
    canvas = (
        _canvas_layer(background_bytes, template, "Background")
        if background_bytes is not None
        else _gradient(template)
    )
    if foreground_bytes is not None:
        canvas.alpha_composite(_canvas_layer(foreground_bytes, template, "Foreground"))
    left = round(template.canvas_width * template.anchor_x - fitted_fish.width / 2)
    top = round(template.canvas_height * template.anchor_y - fitted_fish.height / 2)
    left = max(0, min(template.canvas_width - fitted_fish.width, left))
    top = max(0, min(template.canvas_height - fitted_fish.height, top))

    if template.shadow_enabled:
        shadow_alpha = fitted_fish.getchannel("A").filter(ImageFilter.GaussianBlur(template.shadow_blur_px))
        shadow = Image.new("RGBA", fitted_fish.size, (12, 35, 34, 0))
        shadow.putalpha(shadow_alpha.point(lambda value: round(value * template.shadow_opacity)))
        canvas.alpha_composite(
            shadow,
            (left + template.shadow_offset_x, top + template.shadow_offset_y),
        )

    canvas.alpha_composite(outlined, (left, top))

    # Add extremely weak clipped water optics. The fish RGB remains the primary
    # layer; these overlays are bounded by its alpha and never create a new fish.
    fish_arr = np.asarray(fitted_fish, dtype=np.uint8)
    canvas_arr = np.asarray(canvas, dtype=np.float32).copy()
    tint_alpha = fish_arr[:, :, 3].astype(np.float32) * template.water_tint_opacity
    tint = _solid_layer(template.water_tint, tint_alpha)
    region = canvas_arr[top : top + fitted_fish.height, left : left + fitted_fish.width]
    region[:] = _over_array(region, tint)
    caustic = Image.new("RGBA", fitted_fish.size, (0, 0, 0, 0))
    caustic_draw = ImageDraw.Draw(caustic, "RGBA")
    for index in range(5):
        y = int(fitted_fish.height * (0.18 + index * 0.16))
        caustic_draw.arc((-fitted_fish.width // 3, y - 24, fitted_fish.width + fitted_fish.width // 3, y + 30), 8, 172, fill=(239, 252, 244, 28), width=2)
    caustic_arr = np.asarray(caustic, dtype=np.uint8).copy()
    caustic_arr[:, :, 3] = np.minimum(caustic_arr[:, :, 3], (fish_arr[:, :, 3].astype(np.float32) * template.caustics_opacity).astype(np.uint8))
    region[:] = _over_array(region, caustic_arr.astype(np.float32))
    canvas = Image.fromarray(np.clip(canvas_arr, 0, 255).astype(np.uint8), mode="RGBA")

    # A restrained light overlay provides the template's final depth without
    # introducing foreground objects, text, logos, or a second subject.
    light = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
    light_draw = ImageDraw.Draw(light, "RGBA")
    light_draw.ellipse((-canvas.width * 0.2, -canvas.height * 0.12, canvas.width * 0.9, canvas.height * 0.35), fill=(255, 255, 245, 12))
    canvas.alpha_composite(light)
    if light_bytes is not None:
        canvas.alpha_composite(_canvas_layer(light_bytes, template, "Light"))

    master = io.BytesIO()
    canvas.save(master, format="PNG", optimize=True)
    preview = io.BytesIO()
    canvas.convert("RGB").save(preview, format="WEBP", quality=88, method=6)
    metadata = {
        "template_id": template.template_id,
        "style_id": style.style_id,
        "width": template.canvas_width,
        "height": template.canvas_height,
        "fish_width": fitted_fish.width,
        "fish_height": fitted_fish.height,
        "anchor_x": template.anchor_x,
        "anchor_y": template.anchor_y,
        "fit_scale": round(scale, 6),
        "foreground_object_count": 0,
        "source_asset": source_asset,
        "real_fish_rgba_preserved": True,
    }
    return {"master": master.getvalue(), "preview": preview.getvalue(), "metadata": metadata}
