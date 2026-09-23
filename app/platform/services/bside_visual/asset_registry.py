"""Database-backed B-side style selection and renderer adapters."""

from __future__ import annotations

import json
import random
import secrets
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.platform.models import (
    BsideBackground,
    BsideBackgroundOutlineProfile,
    BsideOutlineStyle,
    BsideVisualSession,
)
from app.platform.services.bside_assets import (
    B_SIDE_CANVAS_V1,
    BsideAssetError,
    background_activation_errors,
)

from .outline_renderer import OutlineStyle
from .template_registry import WaterTemplate


class BsideStylePlanError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def get_active_bside_backgrounds(db: Session) -> list[BsideBackground]:
    """Return only ACTIVE backgrounds with a stored base image.

    Activation validation is performed when an operator changes status. The
    URI check here is intentionally cheap and protects the renderer from rows
    created by older/manual database operations.
    """

    return list(
        db.scalars(
            select(BsideBackground)
            .where(
                BsideBackground.status == "ACTIVE",
                BsideBackground.background_uri.is_not(None),
            )
            .order_by(BsideBackground.id)
        ).all()
    )


def get_outline_profiles(db: Session, background_id: int) -> list[BsideBackgroundOutlineProfile]:
    """Return enabled profiles for a background in stable order."""

    return list(
        db.scalars(
            select(BsideBackgroundOutlineProfile)
            .where(
                BsideBackgroundOutlineProfile.background_id == int(background_id),
                BsideBackgroundOutlineProfile.enabled.is_(True),
            )
            .order_by(BsideBackgroundOutlineProfile.id)
        ).all()
    )


def _weighted_choice(rng: random.Random, profiles: list[BsideBackgroundOutlineProfile]) -> BsideBackgroundOutlineProfile:
    total = sum(max(0, int(profile.weight or 0)) for profile in profiles)
    if not profiles or total != 100:
        raise BsideStylePlanError(
            "OUTLINE_WEIGHT_TOTAL_INVALID",
            f"启用描边权重总和必须为 100，当前为 {total}",
        )
    cursor = rng.uniform(0, total)
    running = 0.0
    for profile in profiles:
        running += max(0, int(profile.weight or 0))
        if cursor <= running:
            return profile
    return profiles[-1]


def _width_ratio(background: BsideBackground, seed: int) -> float:
    """Derive a deterministic fish width from the persisted style seed."""

    minimum = float(background.fish_width_min)
    maximum = float(background.fish_width_max)
    if maximum < minimum:
        minimum, maximum = maximum, minimum
    ratio_rng = random.Random(int(seed) ^ 0xB51DE)
    return round(minimum + (maximum - minimum) * ratio_rng.random(), 6)


def select_active_bside_style_plan(db: Session, *, style_seed: int | None = None) -> dict[str, Any]:
    """Select one weighted plan from the formal ACTIVE asset pool.

    Consumers such as the fish-memory worker persist the returned ids on their
    own durable job.  This keeps the database registry as the only random
    source without manufacturing a Qwen-Lab session for an unrelated flow.
    """

    backgrounds = get_active_bside_backgrounds(db)
    if not backgrounds:
        raise BsideStylePlanError(
            "BSIDE_ASSET_POOL_EMPTY",
            "没有可用的 ACTIVE B 面背景，请先上传资产并启用背景",
        )
    seed = int(style_seed or 0) or secrets.randbelow(2**31 - 1) + 1
    rng = random.Random(seed)
    background = backgrounds[rng.randrange(len(backgrounds))]
    profile = _weighted_choice(rng, get_outline_profiles(db, int(background.id)))
    outline_style = db.get(BsideOutlineStyle, int(profile.outline_style_id))
    if outline_style is None:
        raise BsideStylePlanError("OUTLINE_STYLE_MISSING", "描边样式不存在")
    return {
        "background": background,
        "outline_style": outline_style,
        "profile": profile,
        "style_seed": seed,
        "fish_width_ratio": _width_ratio(background, seed),
    }


def get_bside_style_plan(session: BsideVisualSession, db: Session) -> dict[str, Any]:
    """Get or persist one weighted B-side plan for a session.

    The first call chooses from the ACTIVE database pool. Every later call
    resolves the persisted ids, so a refresh never re-rolls the background or
    outline. Existing sessions without asset ids continue to be handled by the
    legacy B-side route until a DB-backed plan is requested.
    """

    if session.background_id and session.outline_style_id and session.outline_profile_id:
        background = db.get(BsideBackground, int(session.background_id))
        outline_style = db.get(BsideOutlineStyle, int(session.outline_style_id))
        profile = db.get(BsideBackgroundOutlineProfile, int(session.outline_profile_id))
        if background is not None and outline_style is not None and profile is not None:
            seed = int(session.style_seed or 0)
            if not seed:
                seed = secrets.randbelow(2**31 - 1) + 1
                session.style_seed = seed
            return {
                "background": background,
                "outline_style": outline_style,
                "profile": profile,
                "style_seed": seed,
                "fish_width_ratio": _width_ratio(background, seed),
            }

    selected = select_active_bside_style_plan(db, style_seed=session.style_seed)
    background = selected["background"]
    outline_style = selected["outline_style"]
    profile = selected["profile"]
    seed = int(selected["style_seed"])
    session.background_id = background.id
    session.outline_style_id = outline_style.id
    session.outline_profile_id = profile.id
    session.style_seed = seed
    return {
        "background": background,
        "outline_style": outline_style,
        "profile": profile,
        "style_seed": seed,
        "fish_width_ratio": _width_ratio(background, seed),
    }


def _profile_params(profile: BsideBackgroundOutlineProfile) -> dict[str, Any]:
    try:
        value = json.loads(str(profile.render_params_json or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        value = {}
    return value if isinstance(value, dict) else {}


def outline_renderer_style(
    outline_style: BsideOutlineStyle,
    profile: BsideBackgroundOutlineProfile,
) -> OutlineStyle:
    """Adapt the small DB parameter object to the existing outline renderer."""

    params = _profile_params(profile)
    code = str(outline_style.code or "")
    defaults = {
        "none": {
            "color": "#000000",
            "opacity": 0.0,
            "width_ratio": 0.001,
            "coverage_ratio": 0.0,
            "mode": "none",
        },
        "directional_rim": {
            "color": "#D5E1DC",
            "opacity": 0.42,
            "width_ratio": 0.003,
            "coverage_ratio": 0.32,
            "mode": "directional_rim",
        },
        "bottom_water_glow": {
            "color": "#C4E4DD",
            "opacity": 0.36,
            "width_ratio": 0.004,
            "coverage_ratio": 0.28,
            "mode": "bottom_water_glow",
        },
    }.get(code, {"color": "#D5E1DC", "opacity": 0.42, "width_ratio": 0.003, "coverage_ratio": 0.32})
    color = str(params.get("color") or defaults["color"])
    opacity = min(1.0, max(0.0, float(params.get("opacity", defaults["opacity"]))))
    width_ratio = max(0.0005, float(params.get("width_ratio", defaults["width_ratio"])))
    coverage_ratio = min(1.0, max(0.0, float(params.get("coverage_ratio", defaults["coverage_ratio"]))))
    mode = str(params.get("mode") or defaults.get("mode") or (code if code in {"directional_rim", "bottom_water_glow"} else "surrounding"))
    if code == "none":
        mode = "none"
    light_direction = str(params.get("light_direction") or defaults.get("light_direction") or "UPPER_LEFT")
    base_outline_px = max(1, round(width_ratio * 1600))
    return OutlineStyle(
        style_id=code,
        name=outline_style.name,
        color=color,
        base_outline_px=base_outline_px,
        opacity=opacity,
        blur_px=0.5 if code == "none" else 10 if code == "directional_rim" else 12,
        glow_opacity=0.0 if code == "none" else min(0.35, opacity * max(coverage_ratio, 0.25)),
        description=outline_style.description,
        mode=mode,
        coverage_ratio=coverage_ratio,
        light_direction=light_direction,
    )


def background_water_template(
    background: BsideBackground,
    *,
    fish_width_ratio: float | None = None,
) -> WaterTemplate:
    """Adapt a DB background's placement fields to the existing compose API."""

    return WaterTemplate(
        template_id=str(background.code),
        name=str(background.name),
        description=str(background.description or ""),
        canvas_width=int(B_SIDE_CANVAS_V1["width"]),
        canvas_height=int(B_SIDE_CANVAS_V1["height"]),
        anchor_x=float(background.fish_anchor_x),
        anchor_y=float(background.fish_anchor_y),
        max_width_ratio=float(fish_width_ratio or background.fish_width_max),
        max_height_ratio=0.42,
        shadow_enabled=True,
        shadow_opacity=0.10,
        shadow_blur_px=18,
        shadow_offset_x=0,
        shadow_offset_y=5,
        water_tint="#D5E1DC",
        water_tint_opacity=0.025,
        caustics_opacity=0.035,
        top_color="#D8E6E0",
        bottom_color="#6D9A96",
        foreground_color="#4C7776",
    )


def activation_errors(db: Session, background: BsideBackground) -> list[dict[str, Any]]:
    return background_activation_errors(db, background)


__all__ = [
    "BsideStylePlanError",
    "activation_errors",
    "background_water_template",
    "get_active_bside_backgrounds",
    "get_bside_style_plan",
    "get_outline_profiles",
    "outline_renderer_style",
    "select_active_bside_style_plan",
]
