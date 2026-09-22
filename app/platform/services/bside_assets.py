"""Small B-side visual asset registry helpers.

This module deliberately owns only the V1 asset contract: three image slots,
canvas validation, GCS/local persistence, and the initial database seed. It is
not a general-purpose CMS or storage abstraction.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import mimetypes
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

from google.cloud import storage
from PIL import Image, UnidentifiedImageError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.platform.models import (
    BsideBackground,
    BsideBackgroundOutlineProfile,
    BsideOutlineStyle,
)


B_SIDE_CANVAS_V1 = {
    "width": 1080,
    "height": 1350,
    "aspect_ratio": 1080 / 1350,
    "name": "B_SIDE_CANVAS_V1",
}
B_SIDE_ASSET_MAX_BYTES = 50 * 1024 * 1024
_ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
_ALLOWED_SLOTS = {"background", "foreground", "light"}
_ALPHA_SLOTS = {"foreground", "light"}
_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{1,127}$")


class BsideAssetError(ValueError):
    """A user-facing validation/storage error for one B-side asset."""

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def _image_details(data: bytes) -> dict[str, Any]:
    if not data:
        raise BsideAssetError("EMPTY_FILE", "图片文件为空")
    if len(data) > B_SIDE_ASSET_MAX_BYTES:
        raise BsideAssetError(
            "FILE_TOO_LARGE",
            "图片不能超过 50 MiB",
            details={"size_bytes": len(data), "max_bytes": B_SIDE_ASSET_MAX_BYTES},
        )
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            image_format = (image.format or "").upper()
            has_alpha = "A" in image.getbands() or "transparency" in image.info
            image.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise BsideAssetError("UNSUPPORTED_FORMAT", "仅支持 PNG、JPG 或 WEBP 图片") from exc
    if image_format not in _ALLOWED_FORMATS:
        raise BsideAssetError("UNSUPPORTED_FORMAT", "仅支持 PNG、JPG 或 WEBP 图片")
    if width <= 0 or height <= 0:
        raise BsideAssetError("INVALID_DIMENSIONS", "图片尺寸非法")
    return {
        "width": int(width),
        "height": int(height),
        "format": image_format,
        "has_alpha": bool(has_alpha),
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _check_canvas(details: dict[str, Any], *, slot: str) -> None:
    expected_width = int(B_SIDE_CANVAS_V1["width"])
    expected_height = int(B_SIDE_CANVAS_V1["height"])
    actual_width = int(details["width"])
    actual_height = int(details["height"])
    actual_ratio = actual_width / actual_height
    expected_ratio = float(B_SIDE_CANVAS_V1["aspect_ratio"])
    if (
        actual_width != expected_width
        or actual_height != expected_height
        or abs(actual_ratio - expected_ratio) > 1e-6
    ):
        raise BsideAssetError(
            "CANVAS_DIMENSION_MISMATCH",
            (
                f"{slot} 必须使用 {expected_width}×{expected_height}（比例 "
                f"{expected_ratio:.6f}），实际为 {actual_width}×{actual_height}"
            ),
            details={
                "slot": slot,
                "actual": {
                    "width": actual_width,
                    "height": actual_height,
                    "aspect_ratio": round(actual_ratio, 6),
                },
                "expected": {
                    "width": expected_width,
                    "height": expected_height,
                    "aspect_ratio": round(expected_ratio, 6),
                },
            },
        )


def normalize_bside_asset(data: bytes, slot: str) -> dict[str, Any]:
    """Validate one upload and return canonical bytes plus inspectable metadata."""

    normalized_slot = str(slot or "").strip().lower()
    if normalized_slot not in _ALLOWED_SLOTS:
        raise BsideAssetError("INVALID_SLOT", "slot 必须是 background、foreground 或 light")
    details = _image_details(data)
    _check_canvas(details, slot=normalized_slot)
    if normalized_slot in _ALPHA_SLOTS and not details["has_alpha"]:
        raise BsideAssetError(
            "ALPHA_REQUIRED",
            f"{normalized_slot} 必须包含 Alpha 通道",
            details={"slot": normalized_slot},
        )

    try:
        with Image.open(io.BytesIO(data)) as source:
            if normalized_slot == "background":
                # A background is an opaque base layer. Flatten a source alpha
                # channel onto white before the canonical WebP conversion.
                if "A" in source.getbands() or "transparency" in source.info:
                    rgba = source.convert("RGBA")
                    normalized = Image.new("RGB", rgba.size, (255, 255, 255))
                    normalized.paste(rgba, mask=rgba.getchannel("A"))
                else:
                    normalized = source.convert("RGB")
                output = io.BytesIO()
                normalized.save(output, format="WEBP", quality=92, method=6)
                preview_image = normalized.copy()
                preview_image.thumbnail((540, 675), Image.Resampling.LANCZOS)
                preview = io.BytesIO()
                preview_image.save(preview, format="WEBP", quality=88, method=6)
                canonical = output.getvalue()
                preview_bytes = preview.getvalue()
                content_type = "image/webp"
                filename = "background.webp"
            else:
                normalized = source.convert("RGBA")
                output = io.BytesIO()
                normalized.save(output, format="PNG", optimize=True)
                canonical = output.getvalue()
                preview_bytes = None
                content_type = "image/png"
                filename = f"{normalized_slot}.png"
    except (OSError, ValueError) as exc:
        raise BsideAssetError("NORMALIZE_FAILED", "图片无法转换为 B 面资产格式") from exc

    return {
        **details,
        "slot": normalized_slot,
        "content_type": content_type,
        "filename": filename,
        "data": canonical,
        "preview_data": preview_bytes,
        "stored_size_bytes": len(canonical),
    }


def _safe_code(code: str) -> str:
    value = str(code or "").strip()
    if not _CODE_RE.fullmatch(value):
        raise BsideAssetError("INVALID_CODE", "code 必须是小写字母开头的 snake_case")
    return value


def _object_name(code: str, slot: str) -> str:
    code = _safe_code(code)
    if slot == "background":
        return f"bside-assets/{code}/background.webp"
    if slot == "foreground":
        return f"bside-assets/{code}/foreground.png"
    if slot == "light":
        return f"bside-assets/{code}/light.png"
    if slot == "preview":
        return f"bside-assets/{code}/preview.webp"
    raise BsideAssetError("INVALID_SLOT", f"不支持的 B 面资产 slot：{slot}")


def store_bside_asset(code: str, slot: str, data: bytes, content_type: str) -> str:
    """Persist a canonical asset in the existing GCS bucket or test fallback."""

    object_name = _object_name(code, slot)
    bucket_name = os.getenv("GCS_BUCKET", "").strip()
    if bucket_name:
        try:
            blob = storage.Client().bucket(bucket_name).blob(object_name)
            blob.upload_from_string(data, content_type=content_type)
        except Exception as exc:
            raise BsideAssetError("STORAGE_FAILED", "B 面资产上传到 GCS 失败") from exc
        return f"gs://{bucket_name}/{object_name}"

    path = Path("/tmp") / "yujian" / "bside-assets" / code / object_name.rsplit("/", 1)[-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def read_bside_uri(uri: str) -> tuple[bytes, str]:
    """Read a managed URI for renderer/media use without introducing a new store."""

    value = str(uri or "").strip()
    if value.startswith("data:"):
        try:
            header, encoded = value.split(",", 1)
            media_type = header[5:].split(";", 1)[0].strip().lower() or "application/octet-stream"
            data = (
                base64.b64decode(encoded, validate=True)
                if ";base64" in header.lower()
                else encoded.encode("utf-8")
            )
            return data, media_type
        except (TypeError, ValueError) as exc:
            raise FileNotFoundError(value) from exc
    if value.startswith("gs://"):
        try:
            bucket_name, object_name = value[5:].split("/", 1)
        except ValueError as exc:
            raise FileNotFoundError(value) from exc
        blob = storage.Client().bucket(bucket_name).blob(object_name)
        return blob.download_as_bytes(timeout=120), mimetypes.guess_type(object_name)[0] or "application/octet-stream"
    if value.startswith(("http://", "https://")):
        with urllib.request.urlopen(value, timeout=120) as response:
            return response.read(), response.headers.get_content_type() or "application/octet-stream"
    if value.startswith("local://"):
        relative = value[len("local://") :].lstrip("/")
        root = Path.cwd().resolve()
        path = (root / relative).resolve()
        if path != root and root not in path.parents:
            raise ValueError("local URI escapes the application workspace")
    else:
        path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(value)
    return path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def image_metadata_from_uri(uri: str) -> dict[str, Any]:
    data, _media_type = read_bside_uri(uri)
    try:
        with Image.open(io.BytesIO(data)) as image:
            return {
                "width": int(image.width),
                "height": int(image.height),
                "size_bytes": len(data),
                "format": (image.format or "").upper(),
            }
    except (OSError, ValueError) as exc:
        raise BsideAssetError("ASSET_UNREADABLE", "已保存的 B 面资产无法读取") from exc


def background_activation_errors(db: Session, background: BsideBackground) -> list[dict[str, Any]]:
    """Return explicit reasons why one background cannot become ACTIVE."""

    errors: list[dict[str, Any]] = []
    if not str(background.background_uri or "").strip():
        errors.append({"code": "BACKGROUND_REQUIRED", "message": "请先上传 Background"})
    else:
        try:
            details = image_metadata_from_uri(str(background.background_uri))
            if details["width"] != B_SIDE_CANVAS_V1["width"] or details["height"] != B_SIDE_CANVAS_V1["height"]:
                errors.append(
                    {
                        "code": "BACKGROUND_DIMENSION_MISMATCH",
                        "message": (
                            f"Background 尺寸必须为 {B_SIDE_CANVAS_V1['width']}×{B_SIDE_CANVAS_V1['height']}，"
                            f"实际为 {details['width']}×{details['height']}"
                        ),
                    }
                )
        except BsideAssetError as exc:
            errors.append({"code": exc.code, "message": exc.message})
        except (FileNotFoundError, OSError) as exc:
            errors.append({"code": "BACKGROUND_UNREADABLE", "message": "Background 资产暂时不可读取"})

    profiles = db.scalars(
        select(BsideBackgroundOutlineProfile).where(
            BsideBackgroundOutlineProfile.background_id == background.id
        )
    ).all()
    enabled = [profile for profile in profiles if bool(profile.enabled)]
    if not enabled:
        errors.append({"code": "OUTLINE_PROFILE_REQUIRED", "message": "至少启用一个描边组合规则"})
    total = sum(max(0, int(profile.weight or 0)) for profile in enabled)
    if total != 100:
        errors.append(
            {
                "code": "OUTLINE_WEIGHT_TOTAL_INVALID",
                "message": f"当前启用描边权重总和为 {total}，请调整为 100",
                "weight_total": total,
            }
        )
    return errors


BACKGROUND_SEEDS: tuple[dict[str, Any], ...] = (
    {
        "code": "lake_dawn_01",
        "name": "清晨湖面",
        "description": "清晨湖面水域背景。",
        "fish_anchor_x": 0.50,
        "fish_anchor_y": 0.52,
        "fish_width_min": 0.68,
        "fish_width_max": 0.74,
    },
    {
        "code": "shallow_stream_01",
        "name": "浅溪清流",
        "description": "浅溪清流水域背景。",
        "fish_anchor_x": 0.50,
        "fish_anchor_y": 0.50,
        "fish_width_min": 0.72,
        "fish_width_max": 0.78,
    },
    {
        "code": "reservoir_deep_01",
        "name": "深水水库",
        "description": "深水水库水域背景。",
        "fish_anchor_x": 0.50,
        "fish_anchor_y": 0.46,
        "fish_width_min": 0.65,
        "fish_width_max": 0.72,
    },
)

OUTLINE_SEEDS: tuple[dict[str, str], ...] = (
    {"code": "none", "name": "原生", "description": "保留鱼体原生边缘，不添加描边。"},
    {"code": "directional_rim", "name": "侧向轮廓光", "description": "低饱和侧向轮廓光，突出鱼体边缘。"},
    {"code": "bottom_water_glow", "name": "下托水光", "description": "从下方托起鱼体的克制水光。"},
)

PROFILE_WEIGHTS: dict[str, dict[str, int]] = {
    "lake_dawn_01": {"directional_rim": 50, "bottom_water_glow": 40, "none": 10},
    "shallow_stream_01": {"bottom_water_glow": 55, "directional_rim": 30, "none": 15},
    "reservoir_deep_01": {"directional_rim": 50, "bottom_water_glow": 45, "none": 5},
}

PROFILE_PARAMS: dict[str, dict[str, Any]] = {
    "none": {},
    "directional_rim": {
        "color": "#D5E1DC",
        "opacity": 0.42,
        "width_ratio": 0.003,
        "coverage_ratio": 0.32,
        "light_direction": "UPPER_LEFT",
    },
    "bottom_water_glow": {
        "color": "#C4E4DD",
        "opacity": 0.36,
        "width_ratio": 0.004,
        "coverage_ratio": 0.28,
        "light_direction": "BOTTOM_CENTER",
    },
}


def seed_bside_asset_registry(db: Session) -> None:
    """Insert only missing V1 rows; never overwrite operator changes."""

    backgrounds: dict[str, BsideBackground] = {}
    for values in BACKGROUND_SEEDS:
        row = db.scalar(select(BsideBackground).where(BsideBackground.code == values["code"]))
        if row is None:
            row = BsideBackground(status="DRAFT", **values)
            db.add(row)
        backgrounds[str(values["code"])] = row

    outlines: dict[str, BsideOutlineStyle] = {}
    for values in OUTLINE_SEEDS:
        row = db.scalar(select(BsideOutlineStyle).where(BsideOutlineStyle.code == values["code"]))
        if row is None:
            row = BsideOutlineStyle(status="ACTIVE", **values)
            db.add(row)
        outlines[str(values["code"])] = row

    db.flush()
    for background_code, weights in PROFILE_WEIGHTS.items():
        ensure_background_profiles(db, backgrounds[background_code], weights=weights)
    db.commit()


def ensure_background_profiles(
    db: Session,
    background: BsideBackground,
    *,
    weights: dict[str, int] | None = None,
) -> None:
    """Create the three V1 combination rows for a newly added background."""

    default_weights = weights or {"directional_rim": 50, "bottom_water_glow": 40, "none": 10}
    outline_rows = db.scalars(
        select(BsideOutlineStyle).where(BsideOutlineStyle.status == "ACTIVE").order_by(BsideOutlineStyle.id)
    ).all()
    for outline_style in outline_rows:
        if outline_style.code not in default_weights:
            continue
        existing = db.scalar(
            select(BsideBackgroundOutlineProfile).where(
                BsideBackgroundOutlineProfile.background_id == background.id,
                BsideBackgroundOutlineProfile.outline_style_id == outline_style.id,
            )
        )
        if existing is None:
            db.add(
                BsideBackgroundOutlineProfile(
                    background_id=background.id,
                    outline_style_id=outline_style.id,
                    enabled=True,
                    weight=int(default_weights[outline_style.code]),
                    render_params_json=json.dumps(
                        PROFILE_PARAMS.get(outline_style.code, {}),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                )
            )


__all__ = [
    "BACKGROUND_SEEDS",
    "B_SIDE_ASSET_MAX_BYTES",
    "B_SIDE_CANVAS_V1",
    "BsideAssetError",
    "OUTLINE_SEEDS",
    "PROFILE_PARAMS",
    "PROFILE_WEIGHTS",
    "background_activation_errors",
    "ensure_background_profiles",
    "image_metadata_from_uri",
    "normalize_bside_asset",
    "read_bside_uri",
    "seed_bside_asset_registry",
    "store_bside_asset",
]
