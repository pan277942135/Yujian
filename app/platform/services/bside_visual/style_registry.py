from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class OutlineStyle:
    style_id: str
    name: str
    color: str
    base_outline_px: int
    opacity: float
    blur_px: float
    glow_opacity: float
    description: str
    # Registry-backed V1 styles can constrain the effect to a local part of
    # the outer alpha. Legacy styles keep the original surrounding behavior.
    mode: str = "surrounding"
    coverage_ratio: float = 1.0
    light_direction: str = "UPPER_LEFT"


STYLES: tuple[OutlineStyle, ...] = (
    OutlineStyle("lake_mist", "湖雾青", "#A9D0CC", 4, 0.65, 10, 0.22, "低饱和湖水感，默认推荐"),
    OutlineStyle("soft_gold", "柔金", "#D7C489", 4, 0.68, 12, 0.20, "温暖收藏展示感"),
    OutlineStyle("mist_white", "雾白", "#F2F4F1", 3, 0.80, 7, 0.14, "明亮、克制的白色描边"),
    OutlineStyle("deep_teal", "深青", "#648C8C", 3, 0.62, 9, 0.17, "深水背景下的青色描边"),
)

_BY_ID = {item.style_id: item for item in STYLES}


def get_style(style_id: str) -> OutlineStyle:
    key = str(style_id or "").strip()
    try:
        return _BY_ID[key]
    except KeyError as exc:
        raise ValueError(f"未知描边样式：{key or '<empty>'}") from exc


def list_styles() -> list[dict[str, object]]:
    return [asdict(item) for item in STYLES]
