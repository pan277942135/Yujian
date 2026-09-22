from __future__ import annotations

import json
import mimetypes
import re
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import AliasChoices, BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import get_db
from app.platform.models import (
    BsideBackground,
    BsideBackgroundOutlineProfile,
    BsideOutlineStyle,
)
from app.platform.services.bside_assets import (
    B_SIDE_ASSET_MAX_BYTES,
    B_SIDE_CANVAS_V1,
    BsideAssetError,
    background_activation_errors,
    ensure_background_profiles,
    image_metadata_from_uri,
    normalize_bside_asset,
    read_bside_uri,
    store_bside_asset,
)


router = APIRouter(prefix="/api/platform/assets/bside", tags=["bside-assets"])

_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{1,127}$")
_SLOTS = {"background", "foreground", "light"}


class BsideBackgroundCreate(BaseModel):
    name: str = Field(min_length=1, max_length=256)
    code: str = Field(min_length=2, max_length=128)
    description: str = Field(default="", max_length=2000)
    fish_anchor_x: float = Field(default=0.50, ge=0.0, le=1.0)
    fish_anchor_y: float = Field(default=0.50, ge=0.0, le=1.0)
    fish_width_min: float = Field(default=0.68, ge=0.0, le=1.0)
    fish_width_max: float = Field(default=0.74, ge=0.0, le=1.0)

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        result = value.strip()
        if not _CODE_RE.fullmatch(result):
            raise ValueError("code 必须是小写字母开头的 snake_case")
        return result

    @field_validator("name", "description")
    @classmethod
    def trim_text(cls, value: str) -> str:
        return value.strip()


class BsideBackgroundPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=256)
    code: str | None = Field(default=None, min_length=2, max_length=128)
    description: str | None = Field(default=None, max_length=2000)
    fish_anchor_x: float | None = Field(default=None, ge=0.0, le=1.0)
    fish_anchor_y: float | None = Field(default=None, ge=0.0, le=1.0)
    fish_width_min: float | None = Field(default=None, ge=0.0, le=1.0)
    fish_width_max: float | None = Field(default=None, ge=0.0, le=1.0)
    status: Literal["DRAFT", "ACTIVE"] | None = None

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str | None) -> str | None:
        if value is None:
            return None
        result = value.strip()
        if not _CODE_RE.fullmatch(result):
            raise ValueError("code 必须是小写字母开头的 snake_case")
        return result

    @field_validator("name", "description")
    @classmethod
    def trim_optional_text(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None


class BsideProfilePatch(BaseModel):
    enabled: bool | None = None
    weight: int | None = Field(default=None, ge=0, le=100)
    render_params: dict[str, Any] | None = Field(
        default=None,
        validation_alias=AliasChoices("render_params", "render_params_json"),
    )


def _profile_params(row: BsideBackgroundOutlineProfile) -> dict[str, Any]:
    try:
        value = json.loads(str(row.render_params_json or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        value = {}
    return value if isinstance(value, dict) else {}


def _asset_url(background_id: int, slot: str) -> str:
    return f"/api/platform/assets/bside/backgrounds/{background_id}/media/{slot}"


def _asset_info(background_id: int, slot: str, uri: str | None) -> dict[str, Any]:
    if not str(uri or "").strip():
        return {
            "available": False,
            "url": None,
            "filename": None,
            "width": None,
            "height": None,
            "size_bytes": None,
        }
    filename = {
        "background": "background.webp",
        "foreground": "foreground.png",
        "light": "light.png",
        "preview": "preview.webp",
    }[slot]
    try:
        metadata = image_metadata_from_uri(str(uri))
    except Exception:
        metadata = {"width": None, "height": None, "size_bytes": None}
    return {
        "available": True,
        "url": _asset_url(background_id, slot),
        "filename": filename,
        "width": metadata.get("width"),
        "height": metadata.get("height"),
        "size_bytes": metadata.get("size_bytes"),
    }


def _background_dict(db: Session, row: BsideBackground) -> dict[str, Any]:
    errors = background_activation_errors(db, row)
    assets = {
        "background": _asset_info(row.id, "background", row.background_uri),
        "foreground": _asset_info(row.id, "foreground", row.foreground_uri),
        "light": _asset_info(row.id, "light", row.light_uri),
        "preview": _asset_info(row.id, "preview", row.preview_uri),
    }
    profiles = db.scalars(
        select(BsideBackgroundOutlineProfile).where(
            BsideBackgroundOutlineProfile.background_id == row.id
        )
    ).all()
    enabled_profiles = [profile for profile in profiles if bool(profile.enabled)]
    weight_total = sum(max(0, int(profile.weight or 0)) for profile in enabled_profiles)
    return {
        "id": row.id,
        "code": row.code,
        "name": row.name,
        "description": row.description,
        "background_url": assets["background"]["url"],
        "foreground_url": assets["foreground"]["url"],
        "light_url": assets["light"]["url"],
        "preview_url": assets["preview"]["url"],
        "assets": assets,
        "fish_anchor_x": row.fish_anchor_x,
        "fish_anchor_y": row.fish_anchor_y,
        "fish_width_min": row.fish_width_min,
        "fish_width_max": row.fish_width_max,
        "status": row.status,
        "canvas": B_SIDE_CANVAS_V1,
        "weight_total": weight_total,
        "activation_ready": not errors,
        "activation_errors": errors,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def _outline_dict(row: BsideOutlineStyle) -> dict[str, Any]:
    return {
        "id": row.id,
        "code": row.code,
        "name": row.name,
        "description": row.description,
        "status": row.status,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def _profile_dict(db: Session, row: BsideBackgroundOutlineProfile) -> dict[str, Any]:
    background = db.get(BsideBackground, row.background_id)
    outline_style = db.get(BsideOutlineStyle, row.outline_style_id)
    return {
        "id": row.id,
        "background_id": row.background_id,
        "background_code": background.code if background else None,
        "background_name": background.name if background else None,
        "outline_style_id": row.outline_style_id,
        "outline_code": outline_style.code if outline_style else None,
        "outline_name": outline_style.name if outline_style else None,
        "enabled": bool(row.enabled),
        "weight": int(row.weight or 0),
        "render_params": _profile_params(row),
        "render_params_json": row.render_params_json or "{}",
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def _commit(db: Session) -> None:
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="B 面资产记录已存在或数据冲突") from exc


def _background_or_404(db: Session, background_id: int) -> BsideBackground:
    row = db.get(BsideBackground, background_id)
    if row is None:
        raise HTTPException(status_code=404, detail="B 面背景不存在")
    return row


@router.get("/canvas")
def bside_canvas() -> dict[str, Any]:
    return B_SIDE_CANVAS_V1


@router.get("/backgrounds")
def list_bside_backgrounds(db: Session = Depends(get_db)) -> dict[str, Any]:
    rows = db.scalars(select(BsideBackground).order_by(BsideBackground.id)).all()
    return {
        "canvas": B_SIDE_CANVAS_V1,
        "items": [_background_dict(db, row) for row in rows],
    }


@router.post("/backgrounds", status_code=201)
def create_bside_background(
    payload: BsideBackgroundCreate,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if payload.fish_width_max < payload.fish_width_min:
        raise HTTPException(status_code=422, detail="鱼体宽度 Max 不能小于 Min")
    row = BsideBackground(
        **payload.model_dump(),
        status="DRAFT",
    )
    db.add(row)
    db.flush()
    ensure_background_profiles(db, row)
    _commit(db)
    db.refresh(row)
    return _background_dict(db, row)


@router.patch("/backgrounds/{background_id}")
def update_bside_background(
    background_id: int,
    payload: BsideBackgroundPatch,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    row = _background_or_404(db, background_id)
    values = payload.model_dump(exclude_unset=True)
    if "code" in values and values["code"] != row.code:
        duplicate = db.scalar(
            select(BsideBackground).where(
                BsideBackground.code == values["code"],
                BsideBackground.id != row.id,
            )
        )
        if duplicate is not None:
            raise HTTPException(status_code=409, detail="背景 Code 已存在")
    minimum = float(values.get("fish_width_min", row.fish_width_min))
    maximum = float(values.get("fish_width_max", row.fish_width_max))
    if maximum < minimum:
        raise HTTPException(status_code=422, detail="鱼体宽度 Max 不能小于 Min")
    requested_status = values.pop("status", None)
    for field, value in values.items():
        setattr(row, field, value)
    if requested_status == "ACTIVE":
        errors = background_activation_errors(db, row)
        if errors:
            db.rollback()
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "BACKGROUND_ACTIVATION_BLOCKED",
                    "message": "背景未满足 ACTIVE 条件",
                    "errors": errors,
                },
            )
    if requested_status is not None:
        row.status = requested_status
    _commit(db)
    db.refresh(row)
    return _background_dict(db, row)


@router.post("/backgrounds/{background_id}/upload")
async def upload_bside_background_asset(
    background_id: int,
    slot: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    row = _background_or_404(db, background_id)
    normalized_slot = str(slot or "").strip().lower()
    if normalized_slot not in _SLOTS:
        raise HTTPException(status_code=422, detail="slot 必须是 background、foreground 或 light")
    try:
        data = await file.read(B_SIDE_ASSET_MAX_BYTES + 1)
        asset = normalize_bside_asset(data, normalized_slot)
    except BsideAssetError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": exc.code, "message": exc.message, **exc.details},
        ) from exc
    try:
        uri = store_bside_asset(row.code, normalized_slot, asset["data"], asset["content_type"])
        setattr(row, f"{normalized_slot}_uri", uri)
        preview_uri = None
        if normalized_slot == "background" and asset.get("preview_data"):
            preview_uri = store_bside_asset(row.code, "preview", asset["preview_data"], "image/webp")
            row.preview_uri = preview_uri
        _commit(db)
        db.refresh(row)
    except BsideAssetError as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail={"error": exc.code, "message": exc.message}) from exc
    result = _background_dict(db, row)
    result["uploaded"] = {
        "slot": normalized_slot,
        "filename": asset["filename"],
        "width": asset["width"],
        "height": asset["height"],
        "size_bytes": asset["size_bytes"],
        "stored_size_bytes": asset["stored_size_bytes"],
        "content_type": asset["content_type"],
        "preview_generated": bool(preview_uri),
    }
    return result


@router.get("/backgrounds/{background_id}/media/{slot}")
def bside_background_media(background_id: int, slot: str, db: Session = Depends(get_db)) -> Response:
    row = _background_or_404(db, background_id)
    normalized_slot = str(slot or "").strip().lower()
    if normalized_slot not in {"background", "foreground", "light", "preview"}:
        raise HTTPException(status_code=404, detail="B 面资产不存在")
    uri = getattr(row, f"{normalized_slot}_uri", None)
    if not uri:
        raise HTTPException(status_code=404, detail="B 面资产尚未上传")
    try:
        content, media_type = read_bside_uri(str(uri))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="B 面资产不存在") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="B 面资产暂时不可用") from exc
    return Response(
        content=content,
        media_type=media_type or mimetypes.guess_type(str(uri))[0] or "application/octet-stream",
        headers={"Cache-Control": "private, max-age=300", "X-Content-Type-Options": "nosniff"},
    )


@router.get("/outline-styles")
def list_bside_outline_styles(db: Session = Depends(get_db)) -> dict[str, Any]:
    rows = db.scalars(select(BsideOutlineStyle).order_by(BsideOutlineStyle.id)).all()
    return {"items": [_outline_dict(row) for row in rows]}


@router.get("/profiles")
def list_bside_profiles(
    background_id: int | None = None,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    query = select(BsideBackgroundOutlineProfile).order_by(BsideBackgroundOutlineProfile.background_id, BsideBackgroundOutlineProfile.id)
    if background_id is not None:
        query = query.where(BsideBackgroundOutlineProfile.background_id == background_id)
    rows = db.scalars(query).all()
    return {"items": [_profile_dict(db, row) for row in rows]}


@router.patch("/profiles/{profile_id}")
def update_bside_profile(
    profile_id: int,
    payload: BsideProfilePatch,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    row = db.get(BsideBackgroundOutlineProfile, profile_id)
    if row is None:
        raise HTTPException(status_code=404, detail="B 面组合规则不存在")
    values = payload.model_dump(exclude_unset=True)
    if "enabled" in values:
        row.enabled = values["enabled"]
    if "weight" in values:
        row.weight = values["weight"]
    if "render_params" in values:
        params = values["render_params"] or {}
        row.render_params_json = json.dumps(params, ensure_ascii=False, separators=(",", ":"))
    background = db.get(BsideBackground, row.background_id)
    if background and background.status == "ACTIVE":
        errors = background_activation_errors(db, background)
        if errors:
            db.rollback()
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "ACTIVE_BACKGROUND_PROFILE_INVALID",
                    "message": "ACTIVE 背景的启用描边权重总和必须为 100",
                    "errors": errors,
                },
            )
    _commit(db)
    db.refresh(row)
    return _profile_dict(db, row)


__all__ = ["router"]
