from __future__ import annotations

import json
import mimetypes
import os
import secrets
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from google.cloud import storage
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.requests import Request

from app.db import get_db
from app.platform.models import BsideVisualSession, BsideVisualStep, PipelineRun
from app.platform.routes import qwen_image_edit_lab as qwen_lab
from app.platform.services.bside_visual import compose_bside, outline, standardize, validate_transparent_fish
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
    ComposeRequest,
    OutlineRequest,
    StandardizeRequest,
)
from app.platform.services.bside_visual.style_registry import get_style, list_styles
from app.platform.services.bside_visual.template_registry import get_template, list_templates


api_router = APIRouter(prefix="/api/qwen-lab", tags=["qwen-bside-visual"])
page_router = APIRouter(tags=["qwen-bside-visual-pages"])
templates = Jinja2Templates(directory="app/templates")

SESSION_STATUS = "ACTIVE"
STEP_LABELS = {
    STEP_STANDARDIZE: "姿态标准化",
    STEP_OUTLINE: "特色描边",
    STEP_COMPOSE: "融入水体背景",
}
ASSET_LABELS = {
    "source": "Qwen完整鱼体",
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
    return f"/api/fish-portrait/qwen-lab/runs/{run_id}/media/transparent_fish"


def _asset_url(session_id: str, asset: str) -> str:
    return f"/api/qwen-lab/bside-visual/{session_id}/media/{asset}"


def _store_bytes(session_id: str, step_key: str, version: int, filename: str, data: bytes, media_type: str) -> str:
    object_name = f"experiments/qwen_image_edit_lab/bside/{session_id}/{step_key}/v{version}/{filename}"
    bucket_name = os.getenv("GCS_BUCKET", "").strip()
    if bucket_name:
        try:
            blob = storage.Client().bucket(bucket_name).blob(object_name)
            blob.upload_from_string(data, content_type=media_type)
        except Exception as exc:
            raise RuntimeError(f"无法保存 B 面视觉资源：{exc}") from exc
        return f"gs://{bucket_name}/{object_name}"
    path = Path("/tmp") / "yujian" / "qwen_image_edit_lab" / "bside" / session_id / step_key / f"v{version}" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def _read_uri(uri: str) -> tuple[bytes, str]:
    return qwen_lab._read_managed_uri(uri)


def _qwen_output(db: Session, run_id: str) -> tuple[PipelineRun, str, bytes]:
    run = db.get(PipelineRun, str(run_id))
    if run is None or run.pipeline_type != qwen_lab.PIPELINE_TYPE:
        raise HTTPException(status_code=404, detail="Qwen Image Edit Run 不存在")
    if str(run.status or "").upper() != "SUCCESS":
        raise HTTPException(status_code=409, detail={"error": "QWEN_RUN_NOT_SUCCESS", "message": "只有成功的 Qwen Run 才能进入 B 面视觉生成"})
    state = qwen_lab._state_for_run(run)
    result = state.get("result") if isinstance(state.get("result"), dict) else {}
    uri = str(result.get("transparent_fish_uri") or result.get("output_image_uri") or "").strip()
    if not uri:
        raise HTTPException(status_code=409, detail={"error": "QWEN_TRANSPARENT_ASSET_MISSING", "message": "该 Qwen Run 没有可用的透明鱼输出"})
    try:
        data, _media_type = _read_uri(uri)
        validate_transparent_fish(data)
    except Exception as exc:
        if getattr(exc, "code", None) == "INVALID_TRANSPARENT_FISH":
            raise HTTPException(status_code=422, detail={"error": "INVALID_TRANSPARENT_FISH", "message": str(exc)}) from exc
        raise HTTPException(status_code=503, detail={"error": "QWEN_OUTPUT_UNREADABLE", "message": "Qwen 输出暂时不可读取"}) from exc
    return run, uri, data


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
        "error": {"code": row.error_code, "message": row.error_message} if row.error_code or row.error_message else None,
    }


def _serialize_session(db: Session, session: BsideVisualSession) -> dict[str, Any]:
    steps = get_steps(db, session.session_id)
    step_items = [_step_response(session.session_id, steps[key]) for key in STEP_ORDER]
    step_by_key = {item["step"]: item for item in step_items}
    step_by_key[STEP_STANDARDIZE]["can_run"] = True
    step_by_key[STEP_OUTLINE]["can_run"] = steps[STEP_STANDARDIZE].status == COMPLETE
    step_by_key[STEP_COMPOSE]["can_run"] = (
        steps[STEP_STANDARDIZE].status == COMPLETE and steps[STEP_OUTLINE].status == COMPLETE
    )
    assets = []
    source_url = _source_url(session.source_qwen_run_id)
    for key, label in ASSET_LABELS.items():
        if key == "source":
            available = True
            url = source_url
            status = COMPLETE
        else:
            step_key = {"standardized": STEP_STANDARDIZE, "outlined": STEP_OUTLINE, "final": STEP_COMPOSE}[key]
            row = steps[step_key]
            available = bool(row.output_uri)
            url = _asset_url(session.session_id, step_key) if available else None
            status = row.status
        assets.append({"key": key, "label": label, "available": available, "url": url, "status": status})
    completed = sum(1 for row in steps.values() if row.status == COMPLETE)
    return {
        "session_id": session.session_id,
        "source_qwen_run_id": session.source_qwen_run_id,
        "source_transparent_fish_url": source_url,
        "page_url": _page_url(session.session_id),
        "status": session.status,
        "progress": {"completed": completed, "total": 3},
        "steps": step_items,
        "assets": assets,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "updated_at": session.updated_at.isoformat() if session.updated_at else None,
    }


def _session_or_404(db: Session, session_id: str) -> BsideVisualSession:
    session = get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="B 面视觉生成 Session 不存在")
    return session


def _mark_stale(steps: dict[str, BsideVisualStep], keys: tuple[str, ...]) -> None:
    for key in keys:
        row = steps[key]
        if row.output_uri:
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


def _fail_step(db: Session, session: BsideVisualSession, row: BsideVisualStep, code: str, message: str) -> None:
    row.status = FAILED
    row.error_code = code
    row.error_message = str(message)[:2000]
    session.status = SESSION_STATUS
    session.updated_at = _utcnow()
    db.commit()


@api_router.get("/bside-visual/options")
def bside_visual_options() -> dict[str, Any]:
    return {
        "steps": [{"step": key, "label": STEP_LABELS[key]} for key in STEP_ORDER],
        "styles": list_styles(),
        "templates": list_templates(),
        "source_contract": {
            "type": "QWEN_TRANSPARENT_FISH_PNG",
            "alpha_threshold": 16,
            "gpu_required": False,
        },
    }


@api_router.post("/runs/{run_id}/bside-visual")
def create_or_get_bside_visual(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    existing = get_session_for_qwen_run(db, run_id)
    if existing is not None:
        return {"created": False, **_serialize_session(db, existing)}
    _run, source_uri, _source_data = _qwen_output(db, run_id)
    session = BsideVisualSession(
        session_id=_new_session_id(),
        source_qwen_run_id=run_id,
        source_transparent_fish_uri=source_uri,
        status=SESSION_STATUS,
    )
    db.add(session)
    db.flush()
    create_steps(db, session.session_id)
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


@api_router.post("/bside-visual/{session_id}/standardize")
def run_bside_standardize(
    session_id: str,
    payload: StandardizeRequest = Body(default=StandardizeRequest()),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    session = _session_or_404(db, session_id)
    steps = get_steps(db, session_id)
    row = steps[STEP_STANDARDIZE]
    _begin_step(db, session, row)
    try:
        source_data, _media_type = _read_uri(session.source_transparent_fish_uri)
        artifact = standardize(source_data, payload.manual_rotation_offset_deg)
        version = row.version + 1
        row.output_uri = _store_bytes(session_id, STEP_STANDARDIZE, version, "standardized_fish.png", artifact.data, "image/png")
        row.preview_uri = None
        row.metadata_json = json.dumps(artifact.metadata, ensure_ascii=False)
        row.version = version
        row.status = COMPLETE
        session.updated_at = _utcnow()
        _mark_stale(steps, (STEP_OUTLINE, STEP_COMPOSE))
        db.commit()
    except Exception as exc:
        code = getattr(exc, "code", "STANDARDIZE_FAILED")
        _fail_step(db, session, row, code, str(exc))
        status_code = 422 if code in {"INVALID_TRANSPARENT_FISH", "ROTATION_OFFSET_INVALID"} else 500
        raise HTTPException(status_code=status_code, detail={"error": code, "message": str(exc), "session_id": session_id}) from exc
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
        style = get_style(payload.style_id)
        source_data, _media_type = _read_uri(steps[STEP_STANDARDIZE].output_uri or "")
        artifact = outline(source_data, style)
        version = row.version + 1
        row.output_uri = _store_bytes(session_id, STEP_OUTLINE, version, "outlined_fish.png", artifact.data, "image/png")
        row.preview_uri = None
        row.metadata_json = json.dumps(artifact.metadata, ensure_ascii=False)
        row.style_id = style.style_id
        row.version = version
        row.status = COMPLETE
        session.updated_at = _utcnow()
        _mark_stale(steps, (STEP_COMPOSE,))
        db.commit()
    except Exception as exc:
        code = getattr(exc, "code", "OUTLINE_FAILED")
        _fail_step(db, session, row, code, str(exc))
        status_code = 422 if code.startswith("STANDARDIZED_FISH") else 500
        raise HTTPException(status_code=status_code, detail={"error": code, "message": str(exc), "session_id": session_id}) from exc
    return _serialize_session(db, session)


@api_router.post("/bside-visual/{session_id}/compose")
def run_bside_compose(
    session_id: str,
    payload: ComposeRequest = Body(default=ComposeRequest()),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    session = _session_or_404(db, session_id)
    steps = get_steps(db, session_id)
    if steps[STEP_STANDARDIZE].status != COMPLETE or steps[STEP_OUTLINE].status != COMPLETE:
        raise HTTPException(status_code=409, detail={"error": "STEP_LOCKED", "message": "请先完成姿态标准化和特色描边"})
    row = steps[STEP_COMPOSE]
    _begin_step(db, session, row)
    try:
        style = get_style(steps[STEP_OUTLINE].style_id or "lake_mist")
        template = get_template(payload.template_id)
        standardized_data, _media_type = _read_uri(steps[STEP_STANDARDIZE].output_uri or "")
        rendered = compose_bside(standardized_data, style, template)
        version = row.version + 1
        row.output_uri = _store_bytes(session_id, STEP_COMPOSE, version, "bside_final.png", rendered["master"], "image/png")
        row.preview_uri = _store_bytes(session_id, STEP_COMPOSE, version, "bside_preview.webp", rendered["preview"], "image/webp")
        row.metadata_json = json.dumps(rendered["metadata"], ensure_ascii=False)
        row.template_id = template.template_id
        row.style_id = style.style_id
        row.version = version
        row.status = COMPLETE
        session.updated_at = _utcnow()
        db.commit()
    except Exception as exc:
        code = getattr(exc, "code", "COMPOSE_FAILED")
        _fail_step(db, session, row, code, str(exc))
        status_code = 422 if code.startswith("STANDARDIZED_FISH") else 500
        raise HTTPException(status_code=status_code, detail={"error": code, "message": str(exc), "session_id": session_id}) from exc
    return _serialize_session(db, session)


@api_router.get("/bside-visual/{session_id}/media/{asset}")
def bside_visual_media(session_id: str, asset: str, db: Session = Depends(get_db)) -> Response:
    session = _session_or_404(db, session_id)
    steps = get_steps(db, session_id)
    if asset == "source":
        uri = session.source_transparent_fish_uri
    elif asset in {STEP_STANDARDIZE, STEP_OUTLINE, STEP_COMPOSE}:
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
        headers={"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"},
    )


@page_router.get("/platform/fish-portrait/qwen-lab/bside-visual/{session_id}", response_class=HTMLResponse, include_in_schema=False)
def bside_visual_page(request: Request, session_id: str) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="platform/lab/qwen_bside_visual.html",
        context={"page_title": "B面视觉生成", "session_id": session_id},
    )


__all__ = ["api_router", "bside_visual_page", "page_router", "templates"]
