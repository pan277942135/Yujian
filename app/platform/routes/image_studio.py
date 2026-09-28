"""Private Qwen Image Studio V1 API with isolated persistence."""
from __future__ import annotations

import io
import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from google.cloud import storage
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.db import get_db
from app.image_studio_worker_client import invoke_image_studio_worker
from app.platform.models import ImageStudioRun
from app.platform.services.image_studio_prompt import compile_image_studio_prompt
from app.portrait_worker_client import PortraitWorkerError, _read_image_uri

router = APIRouter(prefix="/api/image-studio/v1", tags=["image-studio"])

STORAGE_TYPE = "IMAGE_STUDIO_V1"
MODEL_ID = "qwen-image-edit-2511"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ALLOWED_MEDIA_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_run_id() -> str:
    return "IMAGE_STUDIO_" + _utcnow().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(5)


def _read_upload_sync(upload: UploadFile, label: str) -> tuple[bytes, str, str]:
    media_type = str(upload.content_type or "").split(";", 1)[0].strip().lower()
    suffix = Path(upload.filename or "").suffix.lower()
    suffix_map = {extension: media for media, extension in ALLOWED_MEDIA_TYPES.items()}
    if media_type not in ALLOWED_MEDIA_TYPES:
        media_type = suffix_map.get(suffix, "")
    if media_type not in ALLOWED_MEDIA_TYPES:
        raise HTTPException(status_code=422, detail=f"{label} 仅支持 jpg、png、webp")
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=422, detail=f"{label} 不能为空")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"{label} 不能超过 50 MiB")
    return data, media_type, ALLOWED_MEDIA_TYPES[media_type]


def _store_bytes(run_id: str, kind: str, data: bytes, media_type: str, extension: str) -> str:
    # Image Studio has a dedicated object namespace. It never writes under
    # experiments/qwen_image_edit_lab or fish/B-side storage prefixes.
    object_name = f"image_studio/v1/runs/{run_id}/{kind}{extension}"
    bucket_name = (
        os.getenv("IMAGE_STUDIO_GCS_BUCKET", "").strip()
        or os.getenv("GCS_BUCKET", "").strip()
    )
    if bucket_name:
        blob = storage.Client().bucket(bucket_name).blob(object_name)
        blob.upload_from_string(data, content_type=media_type)
        return f"gs://{bucket_name}/{object_name}"
    path = Path("/tmp") / "yujian" / "image_studio" / "v1" / "runs" / run_id / f"{kind}{extension}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def _mask_composite(base_bytes: bytes, generated_bytes: bytes, mask_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(base_bytes)) as base_source:
        base = base_source.convert("RGB")
    with Image.open(io.BytesIO(generated_bytes)) as generated_source:
        generated = generated_source.convert("RGB").resize(base.size, Image.Resampling.LANCZOS)
    with Image.open(io.BytesIO(mask_bytes)) as mask_source:
        mask = mask_source.convert("L").resize(base.size, Image.Resampling.NEAREST)
    result = Image.composite(generated, base, mask)
    output = io.BytesIO()
    result.save(output, format="PNG")
    return output.getvalue()


def _parse_roles(raw: str, mode: str, reference_count: int) -> list[str]:
    value = str(raw or "").strip()
    if value:
        try:
            roles = json.loads(value)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail="reference_roles 必须是 JSON 数组") from exc
        if not isinstance(roles, list) or not all(isinstance(item, str) for item in roles):
            raise HTTPException(status_code=422, detail="reference_roles 必须是字符串 JSON 数组")
    else:
        roles = []
    if reference_count and not roles:
        if str(mode or "").strip().upper() == "IDENTITY_LOCK":
            roles = ["IDENTITY"] + (["FACE_ANGLE"] if reference_count > 1 else [])
        else:
            roles = ["OBJECT"] * reference_count
    if len(roles) != reference_count:
        raise HTTPException(status_code=422, detail="reference_roles 数量必须与 references 一致")
    return roles


def _request_for(run: ImageStudioRun) -> dict[str, Any]:
    try:
        value = json.loads(run.request_json or "{}")
    except json.JSONDecodeError:
        value = {}
    return value if isinstance(value, dict) else {}


def _result_for(run: ImageStudioRun) -> dict[str, Any]:
    try:
        value = json.loads(run.result_json or "{}")
    except json.JSONDecodeError:
        value = {}
    return value if isinstance(value, dict) else {}


def _reference_uris_for(run: ImageStudioRun) -> list[str]:
    try:
        value = json.loads(run.reference_uris_json or "[]")
    except json.JSONDecodeError:
        value = []
    return [str(uri) for uri in value] if isinstance(value, list) else []


def _response(run: ImageStudioRun) -> dict[str, Any]:
    request = _request_for(run)
    result = _result_for(run)
    reference_uris = _reference_uris_for(run)
    error = None
    if run.error_code or run.error_message:
        error = {"code": run.error_code, "message": run.error_message}
    return {
        "run_id": run.run_id,
        "status": run.status,
        "storage_type": STORAGE_TYPE,
        "model": run.model_version,
        "mode": run.mode,
        "preservation": run.preservation,
        "reference_roles": request.get("reference_roles") or [],
        "seed": run.seed,
        "steps": run.steps,
        "compiled_prompt": request.get("compiled_prompt"),
        "negative_prompt": request.get("negative_prompt"),
        "base_image_url": f"/api/image-studio/v1/runs/{run.run_id}/media/base",
        "reference_urls": [
            f"/api/image-studio/v1/runs/{run.run_id}/media/reference_{index + 1}"
            for index, _ in enumerate(reference_uris)
        ],
        "mask_url": (
            f"/api/image-studio/v1/runs/{run.run_id}/media/mask"
            if run.mask_uri
            else None
        ),
        "output_image_url": (
            f"/api/image-studio/v1/runs/{run.run_id}/media/output"
            if run.output_image_uri
            else None
        ),
        "elapsed_ms": run.elapsed_ms,
        "worker_protocol": result.get("worker_protocol"),
        "mask_composited": result.get("mask_composited", False),
        "error": error,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
    }


@router.post("/edit")
async def edit_image(
    base_image: UploadFile = File(...),
    references: list[UploadFile] | None = File(default=None),
    mask: UploadFile | None = File(default=None),
    prompt: str = Form(...),
    negative_prompt: str = Form(default=""),
    mode: str = Form(default="BASE_EDIT"),
    preservation: str = Form(default="STRONG"),
    reference_roles: str = Form(default="[]"),
    seed: str | None = Form(default=None),
    steps: int = Form(default=25),
    resolution_mode: str = Form(default="real_768"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    reference_uploads = list(references or [])
    if len(reference_uploads) > 2:
        raise HTTPException(status_code=422, detail="Image Studio V1 最多支持 2 张参考图")
    if not 1 <= int(steps) <= 100:
        raise HTTPException(status_code=422, detail="steps 必须在 1 到 100 之间")
    if resolution_mode not in {"current", "real_768"}:
        raise HTTPException(status_code=422, detail="resolution_mode 必须是 current 或 real_768")

    try:
        seed_value = int(seed) if str(seed or "").strip() else None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="seed 必须是整数") from exc

    roles = _parse_roles(reference_roles, mode, len(reference_uploads))
    try:
        compiled = compile_image_studio_prompt(
            prompt,
            mode=mode,
            preservation=preservation,
            reference_roles=roles,
            negative_prompt=negative_prompt,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    base_bytes, base_type, base_ext = _read_upload_sync(base_image, "base_image")
    reference_payloads = [
        _read_upload_sync(upload, f"reference_{index + 1}")
        for index, upload in enumerate(reference_uploads)
    ]
    mask_payload = _read_upload_sync(mask, "mask") if mask is not None else None
    if compiled.mode == "LOCAL_EDIT" and mask_payload is None:
        raise HTTPException(status_code=422, detail="LOCAL_EDIT 必须提供 mask")

    run_id = _new_run_id()
    base_uri = _store_bytes(run_id, "base", base_bytes, base_type, base_ext)
    reference_uris = [
        _store_bytes(run_id, f"reference_{index + 1}", data, media_type, extension)
        for index, (data, media_type, extension) in enumerate(reference_payloads)
    ]
    mask_uri = (
        _store_bytes(run_id, "mask", mask_payload[0], mask_payload[1], mask_payload[2])
        if mask_payload
        else None
    )
    request_state = {
        "reference_roles": list(compiled.reference_roles),
        "compiled_prompt": compiled.prompt,
        "negative_prompt": compiled.negative_prompt,
        "resolution_mode": resolution_mode,
        "stages": [
            {"name": "input", "status": "DONE"},
            {"name": "prompt_compile", "status": "DONE"},
            {"name": "qwen_generation", "status": "RUNNING"},
        ],
    }
    run = ImageStudioRun(
        run_id=run_id,
        status="RUNNING",
        mode=compiled.mode,
        preservation=compiled.preservation,
        model_version=MODEL_ID,
        request_json=json.dumps(request_state, ensure_ascii=False),
        result_json="{}",
        base_image_uri=base_uri,
        reference_uris_json=json.dumps(reference_uris, ensure_ascii=False),
        mask_uri=mask_uri,
        seed=seed_value,
        steps=int(steps),
        started_at=_utcnow(),
    )
    db.add(run)
    db.commit()

    try:
        worker = await run_in_threadpool(
            invoke_image_studio_worker,
            base_image_uri=base_uri,
            reference_image_uris=reference_uris,
            source_run_id=run_id,
            prompt=compiled.prompt,
            negative_prompt=compiled.negative_prompt,
            steps=int(steps),
            seed=seed_value,
            resolution_mode=resolution_mode,
        )
        generated_bytes, _ = _read_image_uri(worker["result_uri"], label="image_studio_output")
        stages = request_state["stages"]
        qwen_stage = next((stage for stage in stages if stage.get("name") == "qwen_generation"), None)
        if qwen_stage is not None:
            qwen_stage["status"] = "DONE"

        if mask_payload is not None:
            stages.append({"name": "mask_composite", "status": "RUNNING"})
            final_bytes = _mask_composite(base_bytes, generated_bytes, mask_payload[0])
            stages[-1]["status"] = "DONE"
        else:
            with Image.open(io.BytesIO(generated_bytes)) as generated_source:
                rgb = generated_source.convert("RGB")
                output = io.BytesIO()
                rgb.save(output, format="PNG")
                final_bytes = output.getvalue()

        output_uri = _store_bytes(run_id, "output", final_bytes, "image/png", ".png")
        run.output_image_uri = output_uri
        run.seed = worker.get("seed", seed_value)
        run.elapsed_ms = worker.get("elapsed_ms")
        run.request_json = json.dumps(request_state, ensure_ascii=False)
        run.result_json = json.dumps(
            {
                "worker_result_uri": worker.get("result_uri"),
                "worker_protocol": worker.get("worker_protocol"),
                "reference_count": len(reference_uris),
                "mask_composited": mask_payload is not None,
            },
            ensure_ascii=False,
        )
        run.status = "SUCCESS"
        run.finished_at = _utcnow()
        db.commit()
        db.refresh(run)
        return _response(run)
    except PortraitWorkerError as exc:
        stages = request_state.get("stages") or []
        if stages:
            stages[-1]["status"] = "FAILED"
        run.status = "FAILED"
        run.error_code = exc.error_code
        run.error_message = str(exc)[:3000]
        run.finished_at = _utcnow()
        run.request_json = json.dumps(request_state, ensure_ascii=False)
        db.commit()
        raise HTTPException(
            status_code=exc.status_code or 502,
            detail={"code": exc.error_code, "message": str(exc)},
        ) from exc
    except Exception as exc:
        stages = request_state.get("stages") or []
        if stages:
            stages[-1]["status"] = "FAILED"
        safe_message = f"{exc.__class__.__name__}: {exc}"[:2000]
        run.status = "FAILED"
        run.error_code = "IMAGE_STUDIO_FAILED"
        run.error_message = safe_message
        run.finished_at = _utcnow()
        run.request_json = json.dumps(request_state, ensure_ascii=False)
        db.commit()
        raise HTTPException(
            status_code=500,
            detail={"code": "IMAGE_STUDIO_FAILED", "message": safe_message},
        ) from exc


@router.get("/runs")
def list_runs(db: Session = Depends(get_db)) -> dict[str, Any]:
    rows = list(
        db.scalars(
            select(ImageStudioRun)
            .order_by(ImageStudioRun.created_at.desc())
            .limit(50)
        )
    )
    return {"items": [_response(run) for run in rows], "count": len(rows)}


@router.get("/runs/{run_id}")
def get_run(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    run = db.get(ImageStudioRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Image Studio run 不存在")
    return _response(run)


@router.get("/runs/{run_id}/media/{kind}")
def get_media(run_id: str, kind: str, db: Session = Depends(get_db)) -> Response:
    run = db.get(ImageStudioRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Image Studio run 不存在")
    reference_uris = _reference_uris_for(run)
    uri: str | None = None
    if kind == "base":
        uri = run.base_image_uri
    elif kind == "mask":
        uri = run.mask_uri
    elif kind.startswith("reference_"):
        try:
            index = int(kind.split("_", 1)[1]) - 1
            uri = reference_uris[index]
        except (ValueError, IndexError):
            uri = None
    elif kind == "output":
        uri = run.output_image_uri
    if not uri:
        raise HTTPException(status_code=404, detail="媒体资源不存在")
    try:
        data, media_type = _read_image_uri(uri, label=f"image_studio_{kind}")
    except Exception as exc:
        raise HTTPException(status_code=404, detail="媒体资源不可读取") from exc
    return Response(content=data, media_type=media_type or "application/octet-stream")


__all__ = ["STORAGE_TYPE", "router", "_mask_composite"]
