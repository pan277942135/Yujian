from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class WaterTemplate:
    template_id: str
    name: str
    description: str
    canvas_width: int
    canvas_height: int
    anchor_x: float
    anchor_y: float
    max_width_ratio: float
    max_height_ratio: float
    shadow_enabled: bool
    shadow_opacity: float
    shadow_blur_px: float
    shadow_offset_x: int
    shadow_offset_y: int
    water_tint: str
    water_tint_opacity: float
    caustics_opacity: float
    top_color: str
    bottom_color: str
    foreground_color: str | None = None


TEMPLATES: tuple[WaterTemplate, ...] = (
    WaterTemplate(
        "lake_dawn_01", "湖面晨光", "确定性水面模板，适合默认验证。", 1080, 1350,
        0.50, 0.52, 0.72, 0.42, True, 0.10, 18, 0, 5, "#91AAA7", 0.05, 0.05,
        "#D8E6E0", "#6D9A96", "#4C7776",
    ),
    WaterTemplate(
        "lake_mist_01", "湖面薄雾", "低对比雾感水体，突出鱼体轮廓。", 1080, 1350,
        0.50, 0.54, 0.72, 0.42, True, 0.09, 20, 0, 5, "#A9C4C0", 0.05, 0.045,
        "#D7E3E0", "#668D8A", "#426B6D",
    ),
    WaterTemplate(
        "river_shallow_01", "浅滩河流", "浅色河床与缓流纹理。", 1080, 1350,
        0.50, 0.50, 0.72, 0.42, True, 0.10, 17, 0, 5, "#8EADA2", 0.05, 0.055,
        "#DDE5D9", "#769B89", "#527361",
    ),
    WaterTemplate(
        "reservoir_deep_01", "深水库湾", "深青水面与低亮度高光。", 1080, 1350,
        0.50, 0.53, 0.72, 0.42, True, 0.11, 19, 0, 5, "#6D9290", 0.05, 0.045,
        "#B9D1CB", "#3F6E73", "#234C55",
    ),
    WaterTemplate(
        "night_fishing_01", "夜钓水面", "克制的夜色水体，不加入文字或装饰。", 1080, 1350,
        0.50, 0.52, 0.72, 0.42, True, 0.12, 18, 0, 5, "#607F87", 0.04, 0.035,
        "#526B78", "#203F54", "#152E43",
    ),
    WaterTemplate(
        "after_rain_01", "雨后湖面", "雨后偏冷的水面渐变与微弱波纹。", 1080, 1350,
        0.50, 0.51, 0.72, 0.42, True, 0.10, 18, 0, 5, "#91A9A8", 0.05, 0.06,
        "#C4D9D5", "#587F83", "#365C65",
    ),
)

_BY_ID = {item.template_id: item for item in TEMPLATES}


def get_template(template_id: str) -> WaterTemplate:
    key = str(template_id or "").strip()
    try:
        return _BY_ID[key]
    except KeyError as exc:
        raise ValueError(f"未知水体模板：{key or '<empty>'}") from exc


def list_templates() -> list[dict[str, object]]:
    return [asdict(item) for item in TEMPLATES]
