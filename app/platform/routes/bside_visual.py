from __future__ import annotations

import json
import mimetypes
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from google.cloud import storage
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.requests import Request

from app.db import get_db
from app.platform.models import BsideBackground, BsideOutlineStyle, BsideVisualSession, BsideVisualStep, PipelineRun
from app.platform.routes import qwen_image_edit_lab as qwen_lab
from app.platform.services.bside_visual import compose_bside, outline, standardize
from app.platform.services.bside_assets import read_bside_uri
from app.platform.services.bside_visual.asset_registry import (
    BsideStylePlanError,
    background_water_template,
    get_active_bside_backgrounds,
    get_bside_style_plan,
    outline_renderer_style,
)
from app.platform.services.bside_visual.repository import (
    create_steps,
    get_session,
    get_session_for_qwen_run,
    get_steps,
)
from app.platform.services.bside_visual.schemas import (
    COMPLETE,
    FAILED,
    NOT_STARTED,
    RUNNING,
    STALE,
    STEP_COMPOSE,
    STEP_ORDER,
    STEP_OUTLINE,
    STEP_STANDARDIZE,
    STEP_TRANSPARENT,
    ComposeRequest,
    OutlineRequest,
    StandardizeRequest,
)
from app.platform.services.bside_visual.style_registry import get_style, list_styles
from app.platform.services.bside_visual.template_registry import get_template, list_templates
from app.platform.services.qwen_output import process_qwen_output


api_router = APIRouter(prefix="/api/qwen-lab", tags=["qwen-bside-visual"])
page_router = APIRouter(tags=["qwen-bside-visual-pages"])
templates = Jinja2Templates(directory="app/templates")

SESSION_STATUS = "ACTIVE"
STEP_LABELS = {
    STEP_TRANSPARENT: "透明背景鱼体",
    STEP_STANDARDIZE: "姿态标准化",
    STEP_OUTLINE: "特色描边",
    STEP_COMPOSE: "融入水体背景",
}
ASSET_LABELS = {
    "source": "Qwen Result · RGB",
    "transparent": "透明背景鱼体",
    "standardized": "标准姿态鱼",
    "outlined": "特色描边鱼",
    "final": "B面最终视觉",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _json(value: Any, fallback: Any = None) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return fallback


def _new_session_id() -> str:
    timestamp = _utcnow().strftime("%Y%m%d_%H%M%S")
    return f"BSIDE_{timestamp}_{secrets.token_hex(5)}"


def _page_url(session_id: str) -> str:
    return f"/platform/fish-portrait/qwen-lab/bside-visual/{session_id}"


def _source_url(run_id: str) -> str:
    return f"/api/fish-portrait/qwen-lab/runs/{run_id}/media/qwen_result_rgb"


def _asset_url(session_id: str, asset: str) -> str:
    return f"/api/qwen-lab/bside-visual/{session_id}/media/{asset}"


def _store_bytes(
    session_id: str,
    step_key: str,
    version: int,
    filename: str,
    data: bytes,
    media_type: str,
) -> str:
    object_name = f"experiments/qwen_image_edit_lab/bside/{session_id}/{step_key}/v{version}/{filename}"
    bucket_name = os.getenv("GCS_BUCKET", "").strip()
    if bucket_name:
        try:
            blob = storage.Client().bucket(bucket_name).blob(object_name)
            blob.upload_from_string(data, content_type=media_type)
        except Exception as exc:
            raise RuntimeError(f"无法保存 B 面视觉资源：{exc}") from exc
        return f"gs://{bucket_name}/{object_name}"
    path = (
        Path("/tmp")
        / "yujian"
        / "qwen_image_edit_lab"
        / "bside"
        / session_id
        / step_key
        / f"v{version}"
        / filename
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def _read_uri(uri: str) -> tuple[bytes, str]:
    return qwen_lab._read_managed_uri(uri)


def _qwen_output(
    db: Session,
    run_id: str,
) -> tuple[PipelineRun, str, bytes, str | None]:
    """Resolve only the saved Qwen RGB asset; never start post-processing."""

    run = db.get(PipelineRun, str(run_id))
    if run is None or run.pipeline_type != qwen_lab.PIPELINE_TYPE:
        raise HTTPException(status_code=404, detail="Qwen Image Edit Run 不存在")
    if str(run.status or "").upper() != "SUCCESS":
        raise HTTPException(
            status_code=409,
            detail={
                "error": "QWEN_RUN_NOT_SUCCESS",
                "message": "只有成功的 Qwen Run 才能进入 B 面视觉生成",
            },
        )
    state = qwen_lab._state_for_run(run)
    result = state.get("result") if isinstance(state.get("result"), dict) else {}
    rgb_uri = str(
        result.get("qwen_result_rgb_uri") or result.get("output_image_uri") or ""
    ).strip()
    if not rgb_uri:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "QWEN_RGB_RESULT_MISSING",
                "message": "该 Qwen Run 没有可用的 RGB 结果",
            },
        )
    try:
        data, _media_type = _read_uri(rgb_uri)
        qwen_lab.normalise_qwen_rgb(data)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "QWEN_RGB_RESULT_UNREADABLE",
                "message": "Qwen RGB 结果暂时不可读取",
            },
        ) from exc
    legacy_transparent_uri = str(result.get("transparent_fish_uri") or "").strip() or None
    return run, rgb_uri, data, legacy_transparent_uri


def _legacy_transparent_uri(db: Session, run_id: str) -> str | None:
    run = db.get(PipelineRun, str(run_id))
    if run is None:
        return None
    state = qwen_lab._state_for_run(run)
    result = state.get("result") if isinstance(state.get("result"), dict) else {}
    return str(result.get("transparent_fish_uri") or "").strip() or None


def _hydrate_legacy_session(db: Session, session: BsideVisualSession) -> dict[str, BsideVisualStep]:
    """Expose pre-four-step sessions without deleting their existing assets."""

    steps = get_steps(db, session.session_id)
    run = db.get(PipelineRun, session.source_qwen_run_id)
    if run is not None:
        state = qwen_lab._state_for_run(run)
        result = state.get("result") if isinstance(state.get("result"), dict) else {}
        rgb_uri = str(
            result.get("qwen_result_rgb_uri") or result.get("output_image_uri") or ""
        ).strip()
        if not session.source_qwen_rgb_uri and rgb_uri:
            session.source_qwen_rgb_uri = rgb_uri
        transparent_uri = str(result.get("transparent_fish_uri") or "").strip() or None
    else:
        transparent_uri = None
    transparent_uri = (
        transparent_uri
        or (str(session.source_transparent_fish_uri or "").strip() or None)
    )
    transparent_step = steps[STEP_TRANSPARENT]
    if transparent_uri and not transparent_step.output_uri:
        result = qwen_lab._state_for_run(run) if run is not None else {}
        result_data = result.get("result") if isinstance(result.get("result"), dict) else {}
        transparent_step.output_uri = transparent_uri
        transparent_step.status = COMPLETE
        transparent_step.version = max(1, int(transparent_step.version or 0))
        transparent_step.metadata_json = json.dumps(
            {
                "legacy_qwen_transparent_asset": True,
                "source_qwen_rgb_uri": session.source_qwen_rgb_uri,
                "fish_mask_uri": result_data.get("fish_mask_uri"),
                "transparent_fish_rgba_uri": transparent_uri,
            },
            ensure_ascii=False,
        )
    return steps


def _qwen_rgb_uri(db: Session, session: BsideVisualSession) -> str | None:
    _hydrate_legacy_session(db, session)
    return str(session.source_qwen_rgb_uri or "").strip() or None


def _step_response(session_id: str, row: BsideVisualStep) -> dict[str, Any]:
    metadata = _json(row.metadata_json, {})
    preview_asset = "compose_preview" if row.step_key == STEP_COMPOSE and row.preview_uri else None
    return {
        "step": row.step_key,
        "label": STEP_LABELS[row.step_key],
        "status": row.status,
        "version": row.version,
        "style_id": row.style_id,
        "template_id": row.template_id,
        "metadata": metadata if isinstance(metadata, dict) else {},
        "output_url": _asset_url(session_id, row.step_key),
        "preview_url": _asset_url(session_id, preview_asset) if preview_asset else None,
        "available": bool(row.output_uri),
        "error": (
            {"code": row.error_code, "message": row.error_message}
            if row.error_code or row.error_message
            else None
        ),
    }


def _serialize_session(db: Session, session: BsideVisualSession) -> dict[str, Any]:
    steps = _hydrate_legacy_session(db, session)
    step_items = [_step_response(session.session_id, steps[key]) for key in STEP_ORDER]
    step_by_key = {item["step"]: item for item in step_items}
    step_by_key[STEP_TRANSPARENT]["can_run"] = (
        bool(_qwen_rgb_uri(db, session)) and steps[STEP_TRANSPARENT].status != RUNNING
    )
    step_by_key[STEP_STANDARDIZE]["can_run"] = (
        steps[STEP_TRANSPARENT].status == COMPLETE and steps[STEP_STANDARDIZE].status != RUNNING
    )
    step_by_key[STEP_OUTLINE]["can_run"] = (
        steps[STEP_STANDARDIZE].status == COMPLETE and steps[STEP_OUTLINE].status != RUNNING
    )
    step_by_key[STEP_COMPOSE]["can_run"] = (
        steps[STEP_OUTLINE].status == COMPLETE and steps[STEP_COMPOSE].status != RUNNING
    )
    assets = []
    source_url = _source_url(session.source_qwen_run_id)
    asset_step_map = {
        "transparent": STEP_TRANSPARENT,
        "standardized": STEP_STANDARDIZE,
        "outlined": STEP_OUTLINE,
        "final": STEP_COMPOSE,
    }
    for key, label in ASSET_LABELS.items():
        if key == "source":
            available = bool(_qwen_rgb_uri(db, session))
            url = source_url if available else None
            status = COMPLETE if available else NOT_STARTED
        else:
            step_key = asset_step_map[key]
            row = steps[step_key]
            available = bool(row.output_uri)
            url = _asset_url(session.session_id, step_key) if available else None
            status = row.status
        assets.append({"key": key, "label": label, "available": available, "url": url, "status": status})
    completed = sum(1 for row in steps.values() if row.status == COMPLETE)
    status_values = {key: steps[key].status for key in STEP_ORDER}
    return {
        "session_id": session.session_id,
        "source_qwen_run_id": session.source_qwen_run_id,
        "source_qwen_rgb_uri": session.source_qwen_rgb_uri,
        "source_qwen_rgb_url": source_url if _qwen_rgb_uri(db, session) else None,
        # Compatibility name for clients that displayed the old source field.
        "source_transparent_fish_url": _asset_url(session.session_id, STEP_TRANSPARENT)
        if steps[STEP_TRANSPARENT].output_uri
        else None,
        "page_url": _page_url(session.session_id),
        "status": session.status,
        "progress": {"completed": completed, "total": 4},
        "steps": step_items,
        "assets": assets,
        "transparent_status": status_values[STEP_TRANSPARENT],
        "standardize_status": status_values[STEP_STANDARDIZE],
        "outline_status": status_values[STEP_OUTLINE],
        "bside_status": status_values[STEP_COMPOSE],
        "style_plan": _persisted_style_plan(db, session),
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "updated_at": session.updated_at.isoformat() if session.updated_at else None,
    }


def _session_or_404(db: Session, session_id: str) -> BsideVisualSession:
    session = get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="B 面视觉生成 Session 不存在")
    return session


def _persisted_style_plan(db: Session, session: BsideVisualSession) -> dict[str, Any] | None:
    if not (session.background_id and session.outline_style_id and session.outline_profile_id):
        return None
    try:
        plan = get_bside_style_plan(session, db)
    except BsideStylePlanError:
        return None
    return _style_plan_payload(plan)


def _style_plan_payload(plan: dict[str, Any]) -> dict[str, Any]:
    """Serialize one plan identically for session state and step metadata."""

    return {
        "background_id": plan["background"].id,
        "background_code": plan["background"].code,
        "background_name": plan["background"].name,
        "outline_style_id": plan["outline_style"].id,
        "outline_code": plan["outline_style"].code,
        "outline_name": plan["outline_style"].name,
        "outline_profile_id": plan["profile"].id,
        "style_seed": plan["style_seed"],
        "fish_width_ratio": plan["fish_width_ratio"],
    }


def _legacy_preset_fallback_allowed(db: Session) -> bool:
    """Keep old isolated test/legacy databases readable before the seed runs."""

    return db.scalar(select(BsideBackground.id).limit(1)) is None


def _mark_stale(steps: dict[str, BsideVisualStep], keys: tuple[str, ...]) -> None:
    for key in keys:
        row = steps[key]
        if row.output_uri or row.status == COMPLETE:
            row.status = STALE
            row.error_code = None
            row.error_message = None


def _begin_step(db: Session, session: BsideVisualSession, row: BsideVisualStep) -> None:
    row.status = RUNNING
    row.error_code = None
    row.error_message = None
    session.status = SESSION_STATUS
    session.updated_at = _utcnow()
    db.commit()


def _fail_step(
    db: Session,
    session: BsideVisualSession,
    row: BsideVisualStep,
    code: str,
    message: str,
) -> None:
    row.status = FAILED
    row.error_code = code
    row.error_message = str(message)[:2000]
    session.status = SESSION_STATUS
    session.updated_at = _utcnow()
    db.commit()


def _step_error_status(code: str) -> int:
    if code.startswith(("QWEN_", "POSE_", "STANDARDIZED_FISH", "INVALID_")):
        return 422
    if code.startswith("BSIDE_") or code.startswith("OUTLINE_WEIGHT"):
        return 409
    return 500


@api_router.get("/bside-visual/options")
def bside_visual_options(db: Session | None = Depends(get_db)) -> dict[str, Any]:
    # Direct unit callers from the original V1 test suite invoke this function
    # without FastAPI dependency injection. Keep the legacy registry response
    # in that case; production requests receive the DB-backed options.
    actual_db = db if isinstance(db, Session) else None
    if actual_db is not None:
        registry_backgrounds = actual_db.scalars(
            select(BsideBackground).order_by(BsideBackground.id)
        ).all()
        active_backgrounds = get_active_bside_backgrounds(actual_db)
        outline_rows = actual_db.scalars(
            select(BsideOutlineStyle).where(BsideOutlineStyle.status == "ACTIVE").order_by(BsideOutlineStyle.id)
        ).all()
        if registry_backgrounds and outline_rows:
            colors = {
                "none": "#000000",
                "directional_rim": "#D5E1DC",
                "bottom_water_glow": "#C4E4DD",
            }
            return {
                "steps": [{"step": key, "label": STEP_LABELS[key]} for key in STEP_ORDER],
                "styles": [
                    {
                        "style_id": row.code,
                        "name": row.name,
                        "description": row.description,
                        "color": colors.get(row.code, "#D5E1DC"),
                    }
                    for row in outline_rows
                ],
                "templates": [
                    {
                        "template_id": row.code,
                        "name": row.name,
                        "description": row.description,
                        "canvas_width": 1080,
                        "canvas_height": 1350,
                        "anchor_x": row.fish_anchor_x,
                        "anchor_y": row.fish_anchor_y,
                        "max_width_ratio": row.fish_width_max,
                    }
                    for row in active_backgrounds
                ],
                "source_contract": {
                    "type": "QWEN_RGB_PNG",
                    "alpha_threshold": 16,
                    "gpu_required": False,
                    "step_1": "DETECTOR_SAM_ALPHA",
                    "formal_output": "RGBA_REAL_FISH",
                    "asset_source": "bside_background_and_outline_registry",
                },
            }
    return {
        "steps": [{"step": key, "label": STEP_LABELS[key]} for key in STEP_ORDER],
        "styles": list_styles(),
        "templates": list_templates(),
        "source_contract": {
            "type": "QWEN_RGB_PNG",
            "alpha_threshold": 16,
            "gpu_required": False,
            "step_1": "DETECTOR_SAM_ALPHA",
            "formal_output": "RGBA_REAL_FISH",
        },
    }


@api_router.post("/runs/{run_id}/bside-visual")
def create_or_get_bside_visual(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    existing = get_session_for_qwen_run(db, run_id)
    if existing is not None:
        return {"created": False, **_serialize_session(db, existing)}
    _run, rgb_uri, _rgb_data, legacy_transparent_uri = _qwen_output(db, run_id)
    session = BsideVisualSession(
        session_id=_new_session_id(),
        source_qwen_run_id=run_id,
        source_qwen_rgb_uri=rgb_uri,
        # Existing installations may still have this column NOT NULL.
        source_transparent_fish_uri=legacy_transparent_uri or "",
        status=SESSION_STATUS,
    )
    db.add(session)
    db.flush()
    create_steps(db, session.session_id)
    if legacy_transparent_uri:
        _hydrate_legacy_session(db, session)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        existing = get_session_for_qwen_run(db, run_id)
        if existing is None:
            raise
        return {"created": False, **_serialize_session(db, existing)}
    return {"created": True, **_serialize_session(db, session)}


@api_router.get("/bside-visual/{session_id}")
def get_bside_visual(session_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    return _serialize_session(db, _session_or_404(db, session_id))


@api_router.post("/bside-visual/{session_id}/extract-transparent")
def run_bside_transparent(
    session_id: str,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    session = _session_or_404(db, session_id)
    steps = get_steps(db, session_id)
    row = steps[STEP_TRANSPARENT]
    source_uri = _qwen_rgb_uri(db, session)
    if not source_uri:
        raise HTTPException(
            status_code=409,
            detail={"error": "QWEN_RGB_RESULT_MISSING", "message": "Qwen RGB 结果不存在"},
        )
    _begin_step(db, session, row)
    try:
        source_data, _media_type = _read_uri(source_uri)
        artifacts = process_qwen_output(source_data)
        version = row.version + 1
        raw_mask_uri = _store_bytes(
            session_id,
            STEP_TRANSPARENT,
            version,
            "fish_mask_raw.png",
            artifacts.fish_mask_raw,
            "image/png",
        )
        mask_uri = _store_bytes(
            session_id,
            STEP_TRANSPARENT,
            version,
            "fish_mask.png",
            artifacts.fish_mask,
            "image/png",
        )
        transparent_uri = _store_bytes(
            session_id,
            STEP_TRANSPARENT,
            version,
            "transparent_fish_rgba.png",
            artifacts.transparent_fish,
            "image/png",
        )
        metadata = dict(artifacts.metadata)
        metadata.update(
            {
                "source_qwen_rgb_uri": source_uri,
                "fish_mask_raw_uri": raw_mask_uri,
                "fish_mask_uri": mask_uri,
                "transparent_fish_rgba_uri": transparent_uri,
            }
        )
        row.output_uri = transparent_uri
        row.preview_uri = None
        row.metadata_json = json.dumps(metadata, ensure_ascii=False)
        row.version = version
        row.status = COMPLETE
        session.source_qwen_rgb_uri = source_uri
        session.updated_at = _utcnow()
        _mark_stale(steps, (STEP_STANDARDIZE, STEP_OUTLINE, STEP_COMPOSE))
        db.commit()
    except Exception as exc:
        code = getattr(exc, "error_code", None) or getattr(exc, "code", None) or "TRANSPARENT_FAILED"
        _fail_step(db, session, row, code, str(exc))
        raise HTTPException(
            status_code=_step_error_status(str(code)),
            detail={"error": code, "message": str(exc), "session_id": session_id},
        ) from exc
    return _serialize_session(db, session)


@api_router.post("/bside-visual/{session_id}/standardize")
def run_bside_standardize(
    session_id: str,
    payload: StandardizeRequest = Body(default=StandardizeRequest()),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    del payload
    session = _session_or_404(db, session_id)
    steps = get_steps(db, session_id)
    if steps[STEP_TRANSPARENT].status != COMPLETE:
        raise HTTPException(status_code=409, detail={"error": "STEP_LOCKED", "message": "请先完成透明背景鱼体"})
    row = steps[STEP_STANDARDIZE]
    _begin_step(db, session, row)
    try:
        source_uri = steps[STEP_TRANSPARENT].output_uri or ""
        source_data, _media_type = _read_uri(source_uri)
        # Step 2 is automatic in V1. The service's legacy offset parameter is
        # kept for direct compatibility, but the route never accepts a user angle.
        artifact = standardize(source_data, 0.0)
        artifact_metadata = dict(artifact.metadata)
        artifact_metadata["source_uri"] = source_uri
        version = row.version + 1
        row.output_uri = _store_bytes(
            session_id,
            STEP_STANDARDIZE,
            version,
            "standardized_fish_rgba.png",
            artifact.data,
            "image/png",
        )
        artifact_metadata["standardized_uri"] = row.output_uri
        row.preview_uri = None
        row.metadata_json = json.dumps(artifact_metadata, ensure_ascii=False)
        row.version = version
        row.status = COMPLETE
        session.updated_at = _utcnow()
        _mark_stale(steps, (STEP_OUTLINE, STEP_COMPOSE))
        db.commit()
    except Exception as exc:
        code = getattr(exc, "code", "STANDARDIZE_FAILED")
        _fail_step(db, session, row, code, str(exc))
        raise HTTPException(
            status_code=_step_error_status(str(code)),
            detail={"error": code, "message": str(exc), "session_id": session_id},
        ) from exc
    return _serialize_session(db, session)


@api_router.post("/bside-visual/{session_id}/outline")
def run_bside_outline(
    session_id: str,
    payload: OutlineRequest = Body(default=OutlineRequest()),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    session = _session_or_404(db, session_id)
    steps = get_steps(db, session_id)
    if steps[STEP_STANDARDIZE].status != COMPLETE:
        raise HTTPException(status_code=409, detail={"error": "STEP_LOCKED", "message": "请先完成姿态标准化"})
    row = steps[STEP_OUTLINE]
    _begin_step(db, session, row)
    try:
        try:
            plan = get_bside_style_plan(session, db)
        except BsideStylePlanError as plan_error:
            if plan_error.code == "BSIDE_ASSET_POOL_EMPTY" and _legacy_preset_fallback_allowed(db):
                plan = None
            else:
                raise
        if plan is not None:
            style = outline_renderer_style(plan["outline_style"], plan["profile"])
        else:
            style = get_style(payload.style_id)
        source_data, _media_type = _read_uri(steps[STEP_STANDARDIZE].output_uri or "")
        artifact = outline(source_data, style)
        artifact_metadata = dict(artifact.metadata)
        artifact_metadata["source_uri"] = steps[STEP_STANDARDIZE].output_uri
        if plan is not None:
            artifact_metadata["style_plan"] = _style_plan_payload(plan)
        version = row.version + 1
        row.output_uri = _store_bytes(
            session_id,
            STEP_OUTLINE,
            version,
            "outlined_fish_rgba.png",
            artifact.data,
            "image/png",
        )
        artifact_metadata["outlined_uri"] = row.output_uri
        row.preview_uri = None
        row.metadata_json = json.dumps(artifact_metadata, ensure_ascii=False)
        row.style_id = style.style_id
        row.version = version
        row.status = COMPLETE
        session.updated_at = _utcnow()
        _mark_stale(steps, (STEP_COMPOSE,))
        db.commit()
    except Exception as exc:
        code = getattr(exc, "code", "OUTLINE_FAILED")
        _fail_step(db, session, row, code, str(exc))
        raise HTTPException(
            status_code=_step_error_status(str(code)),
            detail={"error": code, "message": str(exc), "session_id": session_id},
        ) from exc
    return _serialize_session(db, session)


@api_router.post("/bside-visual/{session_id}/compose")
def run_bside_compose(
    session_id: str,
    payload: ComposeRequest = Body(default=ComposeRequest()),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    session = _session_or_404(db, session_id)
    steps = get_steps(db, session_id)
    if steps[STEP_OUTLINE].status != COMPLETE:
        raise HTTPException(status_code=409, detail={"error": "STEP_LOCKED", "message": "请先完成特色描边"})
    row = steps[STEP_COMPOSE]
    _begin_step(db, session, row)
    try:
        try:
            plan = get_bside_style_plan(session, db)
        except BsideStylePlanError as plan_error:
            if plan_error.code == "BSIDE_ASSET_POOL_EMPTY" and _legacy_preset_fallback_allowed(db):
                plan = None
            else:
                raise
        background_bytes = None
        foreground_bytes = None
        light_bytes = None
        if plan is not None:
            style = outline_renderer_style(plan["outline_style"], plan["profile"])
            template = background_water_template(
                plan["background"],
                fish_width_ratio=plan["fish_width_ratio"],
            )
            background_bytes, _ = read_bside_uri(str(plan["background"].background_uri))
            if plan["background"].foreground_uri:
                foreground_bytes, _ = read_bside_uri(str(plan["background"].foreground_uri))
            if plan["background"].light_uri:
                light_bytes, _ = read_bside_uri(str(plan["background"].light_uri))
        else:
            style = get_style(steps[STEP_OUTLINE].style_id or "lake_mist")
            template = get_template(payload.template_id)
        standardized_data, _media_type = _read_uri(steps[STEP_STANDARDIZE].output_uri or "")
        outlined_data, _outlined_media_type = _read_uri(steps[STEP_OUTLINE].output_uri or "")
        rendered = compose_bside(
            standardized_data,
            style,
            template,
            outlined_fish=outlined_data,
            background_bytes=background_bytes,
            foreground_bytes=foreground_bytes,
            light_bytes=light_bytes,
        )
        version = row.version + 1
        master_uri = _store_bytes(
            session_id,
            STEP_COMPOSE,
            version,
            "bside_result.png",
            rendered["master"],
            "image/png",
        )
        preview_uri = _store_bytes(
            session_id,
            STEP_COMPOSE,
            version,
            "bside_preview.webp",
            rendered["preview"],
            "image/webp",
        )
        rendered_metadata = dict(rendered["metadata"])
        rendered_metadata["source_uri"] = steps[STEP_OUTLINE].output_uri
        rendered_metadata["standardized_uri"] = steps[STEP_STANDARDIZE].output_uri
        rendered_metadata["outlined_uri"] = steps[STEP_OUTLINE].output_uri
        rendered_metadata["bside_result_uri"] = master_uri
        if plan is not None:
            rendered_metadata["style_plan"] = _style_plan_payload(plan)
        row.output_uri = master_uri
        row.preview_uri = preview_uri
        row.metadata_json = json.dumps(rendered_metadata, ensure_ascii=False)
        row.template_id = template.template_id
        row.style_id = style.style_id
        row.version = version
        row.status = COMPLETE
        session.updated_at = _utcnow()
        db.commit()
    except Exception as exc:
        code = getattr(exc, "code", "COMPOSE_FAILED")
        _fail_step(db, session, row, code, str(exc))
        raise HTTPException(
            status_code=_step_error_status(str(code)),
            detail={"error": code, "message": str(exc), "session_id": session_id},
        ) from exc
    return _serialize_session(db, session)


@api_router.get("/bside-visual/{session_id}/media/{asset}")
def bside_visual_media(session_id: str, asset: str, db: Session = Depends(get_db)) -> Response:
    session = _session_or_404(db, session_id)
    steps = get_steps(db, session_id)
    if asset == "source":
        uri = _qwen_rgb_uri(db, session)
    elif asset in {STEP_TRANSPARENT, STEP_STANDARDIZE, STEP_OUTLINE, STEP_COMPOSE}:
        uri = steps[asset].output_uri
    elif asset == "compose_preview":
        uri = steps[STEP_COMPOSE].preview_uri
    else:
        raise HTTPException(status_code=404, detail="B 面视觉资源不存在")
    if not uri:
        raise HTTPException(status_code=404, detail="B 面视觉资源尚未生成")
    try:
        content, media_type = _read_uri(str(uri))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="B 面视觉资源不存在") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="B 面视觉资源暂时不可用") from exc
    return Response(
        content=content,
        media_type=media_type or mimetypes.guess_type(str(uri))[0] or "application/octet-stream",
        headers={
            "Cache-Control": "private, no-cache, must-revalidate",
            "X-Content-Type-Options": "nosniff",
        },
    )


@page_router.get(
    "/platform/fish-portrait/qwen-lab/bside-visual/{session_id}",
    response_class=HTMLResponse,
    include_in_schema=False,
)
def bside_visual_page(request: Request, session_id: str) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="platform/lab/qwen_bside_visual.html",
        context={"page_title": "B面视觉生成", "session_id": session_id},
    )


__all__ = ["api_router", "bside_visual_page", "page_router", "templates"]
