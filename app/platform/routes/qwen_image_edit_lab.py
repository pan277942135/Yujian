"""Independent Qwen Image Edit Lab V1.

This module intentionally bypasses the Fish Portrait preserve pipeline. It stores
the uploaded original image, sends that image directly to the existing
fish-qwen-refine-worker, and records the experiment as its own PipelineRun type.
"""
from __future__ import annotations

import json
import mimetypes
import os
import secrets
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from google.cloud import storage
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.dataset_models import DatasetItem
from app.db import get_db
from app.models import DatasetVersion
from app.platform.models import PipelineRun
from app.platform.services import adapters
from app.portrait_worker_client import PortraitWorkerError, _read_image_uri
from app.qwen_refine_worker_client import (
    QWEN_MODEL_LABEL,
    invoke_qwen_refine_worker,
)

router = APIRouter(prefix="/api/fish-portrait/qwen-lab", tags=["qwen-image-edit-lab"])

PIPELINE_TYPE = "QWEN_IMAGE_EDIT_LAB"
MODEL_ID = "qwen-image-edit-2511"
WORKER_NAME = "fish-qwen-refine-worker"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ALLOWED_MEDIA_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}

DEFAULT_PROMPT = (
    "请优先严格保留原图中的真实鱼体，不要改变鱼的身份、鱼种、身体比例、体型、鳞片、鱼鳍、颜色和真实外观。\n\n"
    "仅在此基础上，对缺失、模糊、不完整或被遮挡的局部区域进行自然补全，并去除杂乱背景、无关物体、脏乱地面、塑料桶、噪点和不自然阴影，适度优化光线、清晰度和构图。\n\n"
    "输出要求为真实写实的高质量鱼类摄影效果，鱼体必须横向放置，整条鱼横向完整展示，适合鱼获收藏展示。"
)
DEFAULT_NEGATIVE_PROMPT = (
    "不要改变鱼种，不要改变鱼的身份，不要把原鱼变成另一条鱼，不要改变体型比例，不要改变鳞片纹理，不要改变鱼鳍结构，"
    "不要出现多余鱼鳍，不要缺失鱼鳍，不要出现畸形鱼体，不要出现幻想鱼，不要出现不真实颜色，不要出现塑料感纹理，"
    "不要卡通化，不要插画风，不要3D渲染风，不要CG感，不要艺术化过度，不要竖向摆放鱼体，不要只显示半条鱼。"
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _duration_ms(started: datetime | None, finished: datetime | None) -> int | None:
    if started is None or finished is None:
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    if finished.tzinfo is None:
        finished = finished.replace(tzinfo=timezone.utc)
    return max(0, int((finished - started).total_seconds() * 1000))


def _new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"QWEN_LAB_{timestamp}_{secrets.token_hex(5)}"


def _public_media_url(run_id: str, kind: str) -> str:
    return f"/api/fish-portrait/qwen-lab/runs/{run_id}/media/{kind}"


def _json(value: Any, fallback: Any = None) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return fallback


def _set_stage(state: dict[str, Any], name: str, status: str, error: str | None = None) -> None:
    stages = state.setdefault("stages", [])
    current = next((item for item in stages if item.get("name") == name), None)
    if current is None:
        current = {"name": name}
        stages.append(current)
    current["status"] = status
    if error:
        current["error"] = error


def _set_state(run: PipelineRun, state: dict[str, Any]) -> None:
    run.stage_json = json.dumps(state, ensure_ascii=False)
    run.updated_at = _utcnow()


def _normalise_text(value: str | None, default: str, label: str) -> str:
    text = str(value or "").strip() or default
    if len(text) > 4000:
        raise HTTPException(status_code=422, detail=f"{label} 不能超过 4000 个字符")
    return text


def _parse_seed(value: str | None) -> int | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        seed = int(raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="seed 必须是整数") from exc
    if not 0 <= seed <= 2**63 - 1:
        raise HTTPException(status_code=422, detail="seed 必须在 0 到 2^63-1 之间")
    return seed


def _media_type_for_upload(image: UploadFile) -> tuple[str, str]:
    media_type = str(image.content_type or "").split(";", 1)[0].strip().lower()
    suffix = Path(image.filename or "").suffix.lower()
    suffix_to_type = {suffix: media for media, suffix in ALLOWED_MEDIA_TYPES.items()}
    if media_type not in ALLOWED_MEDIA_TYPES:
        media_type = suffix_to_type.get(suffix, "")
    if media_type not in ALLOWED_MEDIA_TYPES:
        raise HTTPException(status_code=422, detail="仅支持 jpg、png、webp 图片")
    return media_type, ALLOWED_MEDIA_TYPES[media_type]


async def _read_upload(image: UploadFile) -> tuple[bytes, str, str]:
    media_type, extension = _media_type_for_upload(image)
    data = await image.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=422, detail="图片不能为空")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="图片不能超过 50 MiB")
    return data, media_type, extension


def _store_bytes(
    run_id: str,
    kind: str,
    data: bytes,
    media_type: str,
    extension: str,
) -> str:
    object_name = f"experiments/qwen_image_edit_lab/{run_id}/{kind}{extension}"
    bucket_name = os.getenv("GCS_BUCKET", "").strip()
    if bucket_name:
        try:
            blob = storage.Client().bucket(bucket_name).blob(object_name)
            blob.upload_from_string(data, content_type=media_type)
        except Exception as exc:
            raise PortraitWorkerError(
                "QWEN_LAB_STORAGE_FAILED",
                f"无法保存 Qwen Image Edit Lab {kind} 图片: {exc}",
            ) from exc
        return f"gs://{bucket_name}/{object_name}"

    path = Path("/tmp") / "yujian" / "qwen_image_edit_lab" / run_id / f"{kind}{extension}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def _materialize_output(run_id: str, worker_result: dict[str, Any]) -> str:
    output_uri = str(
        worker_result.get("result_uri")
        or worker_result.get("output_image_uri")
        or worker_result.get("output_uri")
        or ""
    ).strip()
    if not output_uri:
        raise PortraitWorkerError(
            "QWEN_LAB_OUTPUT_MISSING",
            "Qwen Worker 没有返回生成图片",
        )
    data, media_type = _read_image_uri(output_uri, label="qwen_lab_output")
    extension = mimetypes.guess_extension(media_type.split(";", 1)[0].strip()) or ".png"
    return _store_bytes(run_id, "output", data, media_type, extension)


def _read_managed_uri(uri: str) -> tuple[bytes, str]:
    value = str(uri or "").strip()
    if value.startswith("gs://"):
        try:
            bucket_name, object_name = value[5:].split("/", 1)
        except ValueError as exc:
            raise FileNotFoundError(value) from exc
        blob = storage.Client().bucket(bucket_name).blob(object_name)
        return blob.download_as_bytes(timeout=120), mimetypes.guess_type(object_name)[0] or "application/octet-stream"
    if value.startswith("local://"):
        relative = value[len("local://") :].lstrip("/")
        root = Path.cwd().resolve()
        path = (root / relative).resolve()
        if path != root and root not in path.parents:
            raise ValueError("local URI escapes the application workspace")
    elif value.startswith("/"):
        path = Path(value)
    elif value.startswith(("http://", "https://")):
        with urllib.request.urlopen(value, timeout=120) as response:
            return response.read(), response.headers.get_content_type() or "application/octet-stream"
    else:
        raise FileNotFoundError(value)
    if not path.is_file():
        raise FileNotFoundError(value)
    return path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _read_dataset_image(
    db: Session,
    dataset_id: str,
    dataset_item_id: str,
) -> tuple[bytes, str, str, dict[str, Any]]:
    dataset_key = str(dataset_id or "").strip()
    item_key = str(dataset_item_id or "").strip()
    if not dataset_key or not item_key:
        raise HTTPException(status_code=422, detail="dataset_id 和 dataset_item_id 不能为空")

    dataset = db.get(DatasetVersion, dataset_key)
    if dataset is None:
        raise HTTPException(status_code=404, detail="数据集不存在")
    if str(dataset.status or "").upper() != "FROZEN":
        raise HTTPException(status_code=409, detail="只能从已冻结 Dataset 选择图片")

    item = None
    if item_key.isdigit():
        item = db.scalar(
            select(DatasetItem).where(
                DatasetItem.dataset_version == dataset_key,
                DatasetItem.id == int(item_key),
            )
        )
    if item is None:
        item = db.scalar(
            select(DatasetItem).where(
                DatasetItem.dataset_version == dataset_key,
                DatasetItem.image_id == item_key,
            )
        )
    if item is None:
        raise HTTPException(status_code=404, detail="数据集图片不存在")
    uri = str(item.gcs_uri or "").strip()
    if not uri:
        raise HTTPException(status_code=422, detail="数据集图片没有可读取的存储地址")

    try:
        data, media_type = _read_managed_uri(uri)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="数据集图片资源不存在") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="数据集图片暂时不可读取") from exc

    if not data:
        raise HTTPException(status_code=422, detail="数据集图片为空")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="数据集图片不能超过 50 MiB")

    normalized_type = str(media_type or "").split(";", 1)[0].strip().lower()
    if normalized_type not in ALLOWED_MEDIA_TYPES:
        suffix = Path(uri.split("?", 1)[0]).suffix.lower()
        suffix_to_type = {suffix: media for media, suffix in ALLOWED_MEDIA_TYPES.items()}
        normalized_type = suffix_to_type.get(suffix, "")
    if normalized_type not in ALLOWED_MEDIA_TYPES:
        raise HTTPException(status_code=422, detail="数据集图片格式仅支持 jpg、png、webp")

    source = {
        "input_source": "DATASET",
        "dataset_id": dataset_key,
        "dataset_item_id": item.id,
        "image_id": item.image_id,
        "batch_id": item.batch_id,
        "species": item.species_name,
    }
    return data, normalized_type, ALLOWED_MEDIA_TYPES[normalized_type], source


def _state_for_run(run: PipelineRun) -> dict[str, Any]:
    state = _json(run.stage_json, {})
    return state if isinstance(state, dict) else {}


def _response(run: PipelineRun, state: dict[str, Any]) -> dict[str, Any]:
    request = state.get("request") if isinstance(state.get("request"), dict) else {}
    result = state.get("result") if isinstance(state.get("result"), dict) else {}
    input_uri = str(request.get("input_image_uri") or "").strip() or None
    output_uri = str(result.get("output_image_uri") or "").strip() or None
    return {
        "run_id": run.run_id,
        "input_image_uri": input_uri,
        "input_image_url": _public_media_url(run.run_id, "input") if input_uri else None,
        "input_source": request.get("input_source") or "LOCAL_UPLOAD",
        "dataset_id": request.get("dataset_id"),
        "dataset_item_id": request.get("dataset_item_id"),
        "image_id": request.get("image_id"),
        "species": request.get("species"),
        "prompt": request.get("prompt"),
        "negative_prompt": request.get("negative_prompt"),
        "output_image_uri": output_uri,
        "output_image_url": _public_media_url(run.run_id, "output") if output_uri else None,
        "status": str(run.status or "UNKNOWN").upper(),
        "model": MODEL_ID,
        "model_label": QWEN_MODEL_LABEL,
        "worker": WORKER_NAME,
        "seed": result.get("seed") if result.get("seed") is not None else request.get("seed"),
        "time_ms": result.get("elapsed_ms"),
        "elapsed_ms": result.get("elapsed_ms"),
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "error": state.get("error"),
    }


def _mark_failed(
    db: Session,
    run_id: str,
    *,
    stage: str,
    error_code: str,
    message: str,
) -> None:
    db.rollback()
    run = db.get(PipelineRun, run_id)
    if run is None:
        return
    state = _state_for_run(run)
    safe_message = str(message or error_code)[:3000]
    _set_stage(state, stage, "FAILED", f"{error_code}: {safe_message}")
    state["error"] = {"code": error_code, "message": safe_message}
    run.status = "FAILED"
    run.current_stage = stage
    run.error_stage = stage
    run.error_message = f"{error_code}: {safe_message}"
    run.finished_at = _utcnow()
    if run.started_at:
        run.duration_ms = _duration_ms(run.started_at, run.finished_at)
    _set_state(run, state)
    adapters.record_operation(
        db,
        "CREATE_QWEN_IMAGE_EDIT_LAB_RUN",
        "PIPELINE_RUN",
        run_id,
        status="FAILED",
        message=safe_message,
        detail={"error_code": error_code, "stage": stage, "model": MODEL_ID},
    )
    db.commit()


@router.post("/generate")
async def generate_qwen_image_edit_lab(
    image: UploadFile | None = File(default=None),
    prompt: str = Form(default=DEFAULT_PROMPT),
    negative_prompt: str = Form(default=DEFAULT_NEGATIVE_PROMPT),
    seed: str | None = Form(default=None),
    dataset_id: str | None = Form(default=None),
    dataset_item_id: str | None = Form(default=None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    dataset_id_value = dataset_id.strip() if isinstance(dataset_id, str) else ""
    dataset_item_id_value = dataset_item_id.strip() if isinstance(dataset_item_id, str) else ""
    has_upload = image is not None
    has_dataset = bool(dataset_id_value or dataset_item_id_value)
    if has_upload and has_dataset:
        raise HTTPException(status_code=422, detail="本地上传和 Dataset 图片不能同时提交")
    if not has_upload and not (dataset_id_value and dataset_item_id_value):
        raise HTTPException(status_code=422, detail="请选择本地图片或 Dataset 图片")

    if has_upload:
        data, media_type, extension = await _read_upload(image)
        source = {"input_source": "LOCAL_UPLOAD"}
    else:
        data, media_type, extension, source = _read_dataset_image(
            db,
            dataset_id_value,
            dataset_item_id_value,
        )

    prompt_value = _normalise_text(prompt, DEFAULT_PROMPT, "Prompt")
    negative_prompt_value = _normalise_text(negative_prompt, DEFAULT_NEGATIVE_PROMPT, "Negative Prompt")
    seed_value = _parse_seed(seed)

    run_id = _new_run_id()
    input_uri = _store_bytes(run_id, "input", data, media_type, extension)
    request_state = {
        "input_image_uri": input_uri,
        "prompt": prompt_value,
        "negative_prompt": negative_prompt_value,
        "seed": seed_value,
        "model": MODEL_ID,
        "worker": WORKER_NAME,
        **source,
    }
    state: dict[str, Any] = {
        "type": PIPELINE_TYPE,
        "request": request_state,
        "result": None,
        "worker": None,
        "stages": [
            {"name": "upload_input", "status": "DONE"},
            {"name": "qwen_generate", "status": "RUNNING"},
            {"name": "persist_result", "status": "PENDING"},
        ],
    }
    run = PipelineRun(
        run_id=run_id,
        pipeline_type=PIPELINE_TYPE,
        status="RUNNING",
        current_stage="qwen_generate",
        started_at=_utcnow(),
        stage_json=json.dumps(state, ensure_ascii=False),
        model_version=MODEL_ID,
    )
    db.add(run)
    adapters.record_operation(
        db,
        "CREATE_QWEN_IMAGE_EDIT_LAB_RUN",
        "PIPELINE_RUN",
        run_id,
        detail={
            "type": PIPELINE_TYPE,
            "input_image": input_uri,
            "input_source": source.get("input_source"),
            "dataset_id": source.get("dataset_id"),
            "dataset_item_id": source.get("dataset_item_id"),
            "image_id": source.get("image_id"),
            "prompt": prompt_value,
            "negative_prompt": negative_prompt_value,
            "model": MODEL_ID,
            "worker": WORKER_NAME,
        },
    )
    db.commit()

    started = time.perf_counter()
    active_stage = "qwen_generate"
    try:
        worker_result = invoke_qwen_refine_worker(
            visible_fish_refined_image_uri=input_uri,
            source_run_id=run_id,
            prompt=prompt_value,
            negative_prompt=negative_prompt_value,
            seed=seed_value,
        )
        state["worker"] = {
            "name": WORKER_NAME,
            "model": worker_result.get("worker_model") or QWEN_MODEL_LABEL,
            "status": worker_result.get("worker_status") or "WORKER_EXECUTED",
            "http_status": worker_result.get("worker_http_status"),
            "protocol": worker_result.get("worker_protocol"),
        }
        _set_stage(state, "qwen_generate", "DONE")
        _set_stage(state, "persist_result", "RUNNING")
        active_stage = "persist_result"
        run.current_stage = "persist_result"
        _set_state(run, state)
        db.commit()

        output_uri = _materialize_output(run_id, worker_result)
        elapsed_ms = worker_result.get("elapsed_ms")
        if elapsed_ms is None:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        actual_seed = worker_result.get("seed")
        state["result"] = {
            "input_image_uri": input_uri,
            "output_image_uri": output_uri,
            "seed": actual_seed if actual_seed is not None else seed_value,
            "elapsed_ms": elapsed_ms,
            "model": MODEL_ID,
            "worker": WORKER_NAME,
        }
        _set_stage(state, "persist_result", "DONE")
        run.status = "SUCCESS"
        run.current_stage = "complete"
        run.finished_at = _utcnow()
        if run.started_at:
            run.duration_ms = _duration_ms(run.started_at, run.finished_at)
        _set_state(run, state)
        adapters.record_operation(
            db,
            "CREATE_QWEN_IMAGE_EDIT_LAB_RUN",
            "PIPELINE_RUN",
            run_id,
            status="SUCCESS",
            message="Qwen Image Edit Lab 生成完成",
            detail={
                "type": PIPELINE_TYPE,
                "model": MODEL_ID,
                "worker": WORKER_NAME,
                "input_source": source.get("input_source"),
                "dataset_id": source.get("dataset_id"),
                "dataset_item_id": source.get("dataset_item_id"),
                "seed": state["result"]["seed"],
                "elapsed_ms": elapsed_ms,
            },
        )
        db.commit()
        return _response(run, state)
    except PortraitWorkerError as exc:
        _mark_failed(
            db,
            run_id,
            stage=active_stage,
            error_code=exc.error_code,
            message=str(exc),
        )
        raise HTTPException(
            status_code=502,
            detail={"error_code": exc.error_code, "message": str(exc), "run_id": run_id},
        ) from exc
    except Exception as exc:
        _mark_failed(
            db,
            run_id,
            stage=active_stage,
            error_code="QWEN_IMAGE_EDIT_LAB_FAILED",
            message=str(exc),
        )
        raise HTTPException(
            status_code=500,
            detail={"error_code": "QWEN_IMAGE_EDIT_LAB_FAILED", "message": str(exc), "run_id": run_id},
        ) from exc


@router.get("/runs")
def qwen_image_edit_lab_runs(
    page: int = Query(default=1, ge=1, le=100000),
    size: int = Query(default=10, ge=1, le=100),
    limit: int | None = Query(default=None, ge=1, le=100),
    db: Session = Depends(get_db),
) -> dict[str, Any] | list[dict[str, Any]]:
    # Keep the legacy limit query usable for existing callers while the Lab UI
    # uses the paginated response with a stable ten-row page size.
    legacy_limit = limit is not None
    if limit is not None:
        page = 1
        size = limit

    predicate = PipelineRun.pipeline_type == PIPELINE_TYPE
    total = int(
        db.scalar(
            select(func.count())
            .select_from(PipelineRun)
            .where(predicate)
        )
        or 0
    )
    rows = db.scalars(
        select(PipelineRun)
        .where(predicate)
        .order_by(PipelineRun.created_at.desc(), PipelineRun.run_id.desc())
        .offset((page - 1) * size)
        .limit(size)
    ).all()
    items = [_response(row, _state_for_run(row)) for row in rows]
    if legacy_limit:
        return items
    page_count = max(1, (total + size - 1) // size)
    return {
        "items": items,
        "total": total,
        "page": page,
        "size": size,
        "page_count": page_count,
        "has_next": page < page_count,
    }


@router.get("/runs/{run_id}")
def qwen_image_edit_lab_run(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    run = db.get(PipelineRun, run_id)
    if run is None or run.pipeline_type != PIPELINE_TYPE:
        raise HTTPException(status_code=404, detail="Qwen Image Edit Lab 记录不存在")
    return _response(run, _state_for_run(run))


@router.get("/runs/{run_id}/media/{kind}")
def qwen_image_edit_lab_media(run_id: str, kind: str, db: Session = Depends(get_db)) -> Response:
    if kind not in {"input", "output"}:
        raise HTTPException(status_code=404, detail="资源不存在")
    run = db.get(PipelineRun, run_id)
    if run is None or run.pipeline_type != PIPELINE_TYPE:
        raise HTTPException(status_code=404, detail="Qwen Image Edit Lab 记录不存在")
    state = _state_for_run(run)
    request = state.get("request") if isinstance(state.get("request"), dict) else {}
    result = state.get("result") if isinstance(state.get("result"), dict) else {}
    uri = request.get("input_image_uri") if kind == "input" else result.get("output_image_uri")
    if not uri:
        raise HTTPException(status_code=404, detail="资源不存在")
    try:
        content, media_type = _read_managed_uri(str(uri))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="资源不存在") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="资源暂时不可用") from exc
    return Response(
        content=content,
        media_type=media_type,
        headers={"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"},
    )


__all__ = [
    "DEFAULT_NEGATIVE_PROMPT",
    "DEFAULT_PROMPT",
    "MODEL_ID",
    "PIPELINE_TYPE",
    "WORKER_NAME",
    "generate_qwen_image_edit_lab",
    "qwen_image_edit_lab_media",
    "qwen_image_edit_lab_run",
    "router",
]
