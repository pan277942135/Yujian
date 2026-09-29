"""Private Qwen Image Studio V1 API with isolated persistence."""
from __future__ import annotations

import io
import json
import os
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from google.api_core.exceptions import NotFound
from google.cloud import storage
from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import SessionLocal, get_db
from app.image_studio_worker_client import IMAGE_STUDIO_IDENTITY_V2_MODE, invoke_image_studio_worker
from app.qwen_refine_worker_client import check_qwen_refine_worker
from app.platform.models import ImageStudioRun
from app.platform.services.image_studio_prompt import (
    compile_clean_frame_prompt,
    compile_image_studio_prompt,
    compile_scene_transfer_stage_prompt,
    compile_strict_head_swap_prompt,
)
from app.portrait_worker_client import PortraitWorkerError, _read_image_uri
from app.services.image_studio_identity import (
    IdentityPreprocessError,
    composite_head_roi,
    crop_png,
    measure_strict_composite,
    prepare_strict_identity_assets,
)
from app.services.image_studio_jobs import enqueue_image_studio_queue

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


def _delete_run_assets(run_id: str) -> int:
    """Delete every Image Studio-managed asset for one run.

    Storage deletion is scoped to the run namespace rather than individual DB
    columns so uploaded references, masks and any partially-written outputs are
    removed together.
    """

    safe_run_id = str(run_id or "").strip()
    if not safe_run_id.startswith("IMAGE_STUDIO_"):
        raise RuntimeError("invalid Image Studio run id")

    prefix = f"image_studio/v1/runs/{safe_run_id}/"
    bucket_name = (
        os.getenv("IMAGE_STUDIO_GCS_BUCKET", "").strip()
        or os.getenv("GCS_BUCKET", "").strip()
    )
    if bucket_name:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        deleted = 0
        for blob in client.list_blobs(bucket, prefix=prefix):
            try:
                blob.delete()
                deleted += 1
            except NotFound:
                continue
        return deleted

    run_dir = (
        Path("/tmp")
        / "yujian"
        / "image_studio"
        / "v1"
        / "runs"
        / safe_run_id
    )
    if not run_dir.exists():
        return 0
    file_count = sum(1 for path in run_dir.rglob("*") if path.is_file())
    shutil.rmtree(run_dir)
    return file_count


def _delete_failed_run(db: Session, run: ImageStudioRun) -> dict[str, Any]:
    if run.status != "FAILED":
        raise HTTPException(
            status_code=409,
            detail={
                "code": "IMAGE_STUDIO_DELETE_REQUIRES_FAILED",
                "message": "仅 FAILED 任务允许使用失败任务清理。",
                "status": run.status,
            },
        )

    try:
        deleted_assets = _delete_run_assets(run.run_id)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "code": "IMAGE_STUDIO_ASSET_DELETE_FAILED",
                "message": f"任务图片清理失败，数据库记录已保留：{exc}",
            },
        ) from exc

    run_id = run.run_id
    db.delete(run)
    db.commit()
    return {
        "deleted": True,
        "run_id": run_id,
        "deleted_assets": deleted_assets,
    }


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


def _resize_png_to_long_edge(
    image_bytes: bytes,
    *,
    long_edge: int,
    aspect_source_bytes: bytes | None = None,
) -> tuple[bytes, tuple[int, int]]:
    if int(long_edge) not in {768, 1024, 1536, 2048}:
        raise ValueError("output_long_edge must be one of 768, 1024, 1536, 2048")

    with Image.open(io.BytesIO(image_bytes)) as image_source:
        image = image_source.convert("RGB")
    if aspect_source_bytes:
        with Image.open(io.BytesIO(aspect_source_bytes)) as aspect_source:
            source_width, source_height = aspect_source.size
    else:
        source_width, source_height = image.size

    if source_width <= 0 or source_height <= 0:
        raise ValueError("invalid source image size")

    if source_width >= source_height:
        target_width = int(long_edge)
        target_height = max(1, int(round(source_height * float(long_edge) / source_width)))
    else:
        target_height = int(long_edge)
        target_width = max(1, int(round(source_width * float(long_edge) / source_height)))

    if image.size != (target_width, target_height):
        image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue(), (target_width, target_height)


def _require_worker_ready() -> dict[str, Any]:
    """Reject transient GPU warmup before creating an Image Studio run."""

    try:
        snapshot = check_qwen_refine_worker()
    except PortraitWorkerError as exc:
        raise HTTPException(
            status_code=exc.status_code or 503,
            detail={"code": exc.error_code, "message": str(exc)},
        ) from exc

    health = snapshot.get("health") if isinstance(snapshot.get("health"), dict) else {}
    worker_status = str(health.get("status") or "").strip().lower()
    model_loaded = health.get("model_loaded") is True
    if worker_status not in {"ready", "busy"} or not model_loaded:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "QWEN_WORKER_NOT_READY",
                "status": worker_status or "unavailable",
                "model_loaded": model_loaded,
                "message": "Qwen 模型正在启动或预热，Ready 后再开始生成。",
            },
        )
    return health


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
        mode_value = str(mode or "").strip().upper()
        if mode_value in {"IDENTITY_LOCK", "STRICT_HEAD_SWAP"}:
            roles = ["IDENTITY"] + (["FACE_ANGLE"] if reference_count > 1 else [])
        elif mode_value == "HEAD_SWAP_SCENE_TRANSFER":
            roles = ["IDENTITY", "SCENE"] + (["FACE_ANGLE"] if reference_count > 2 else [])
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


def _intermediate_asset_urls(run: ImageStudioRun) -> list[dict[str, str]]:
    result = _result_for(run)
    assets = result.get("assets")
    if not isinstance(assets, dict):
        return []
    labels = {
        "base_face_crop": "Base Face Crop",
        "base_head_crop": "Base Head Crop",
        "identity_face_crop": "Identity Face Crop",
        "identity_head_crop": "Identity Head Crop",
        "identity_angle_crop": "Identity Angle Crop",
        "mask_binary": "Auto Mask",
        "mask_preview": "Mask Preview",
        "clean_frame_result": "Clean Frame Result",
        "scene_stage_result": "Scene Stage Result",
        "edited_head_roi": "Edited Head ROI",
    }
    items: list[dict[str, str]] = []
    for kind, label in labels.items():
        if assets.get(kind):
            items.append(
                {
                    "kind": kind,
                    "label": label,
                    "url": f"/api/image-studio/v1/runs/{run.run_id}/media/{kind}",
                    "download_url": f"/api/image-studio/v1/runs/{run.run_id}/download/{kind}",
                }
            )
    return items


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
        "pipeline_version": request.get("pipeline_version"),
        "strict_geometry": request.get("strict_geometry"),
        "identity_strength": request.get("identity_strength"),
        "head_edit_tightness": request.get("head_edit_tightness"),
        "keep_hair_color": request.get("keep_hair_color"),
        "keep_base_hair_shape": request.get("keep_base_hair_shape"),
        "clean_output": request.get("clean_output"),
        "output_long_edge": request.get("output_long_edge"),
        "output_size": result.get("output_size"),
        "seed": run.seed,
        "steps": run.steps,
        "compiled_prompt": request.get("compiled_prompt"),
        "negative_prompt": request.get("negative_prompt"),
        "scene_prompt": request.get("scene_prompt"),
        "scene_negative_prompt": request.get("scene_negative_prompt"),
        "stages": request.get("stages") or [],
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
        "intermediate_assets": _intermediate_asset_urls(run),
        "strict_validation": result.get("strict_validation"),
        "elapsed_ms": run.elapsed_ms,
        "worker_protocol": result.get("worker_protocol"),
        "mask_composited": result.get("mask_composited", False),
        "error": error,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
    }


def _queue_position(db: Session, run: ImageStudioRun) -> int | None:
    if run.status != "QUEUED":
        return None
    queued_ids = list(
        db.scalars(
            select(ImageStudioRun.run_id)
            .where(ImageStudioRun.status == "QUEUED")
            .order_by(ImageStudioRun.created_at.asc(), ImageStudioRun.run_id.asc())
        )
    )
    try:
        return queued_ids.index(run.run_id) + 1
    except ValueError:
        return None


def _queue_snapshot(db: Session) -> dict[str, Any]:
    queued = int(
        db.scalar(
            select(func.count())
            .select_from(ImageStudioRun)
            .where(ImageStudioRun.status == "QUEUED")
        )
        or 0
    )
    running = int(
        db.scalar(
            select(func.count())
            .select_from(ImageStudioRun)
            .where(ImageStudioRun.status == "RUNNING")
        )
        or 0
    )
    current_run_id = db.scalar(
        select(ImageStudioRun.run_id)
        .where(ImageStudioRun.status == "RUNNING")
        .order_by(ImageStudioRun.started_at.asc(), ImageStudioRun.created_at.asc())
        .limit(1)
    )
    return {
        "running": running,
        "queued": queued,
        "active": running + queued,
        "current_run_id": str(current_run_id) if current_run_id else None,
        "concurrency": 1,
        "policy": "FIFO_SINGLE_L4",
    }


def _response_with_queue(db: Session, run: ImageStudioRun) -> dict[str, Any]:
    payload = _response(run)
    payload["queue_position"] = _queue_position(db, run)
    payload["queue"] = _queue_snapshot(db)
    return payload


def _set_stage(stages: list[dict[str, Any]], name: str, status: str, **detail: Any) -> None:
    stage = next(
        (
            item
            for item in stages
            if isinstance(item, dict) and item.get("name") == name
        ),
        None,
    )
    if stage is None:
        stage = {"name": name, "status": status}
        stages.append(stage)
    else:
        stage["status"] = status
    for key, value in detail.items():
        if value is not None:
            stage[key] = value


def _persist_progress(
    db: Session,
    run: ImageStudioRun,
    request_state: dict[str, Any],
    result_state: dict[str, Any],
) -> None:
    run.request_json = json.dumps(request_state, ensure_ascii=False)
    run.result_json = json.dumps(result_state, ensure_ascii=False)
    db.commit()


def _strict_reference_indexes(roles: list[str]) -> tuple[int, int | None, int | None]:
    try:
        identity_index = roles.index("IDENTITY")
    except ValueError as exc:
        raise IdentityPreprocessError(
            "IDENTITY_REFERENCE_REQUIRED",
            "Strict identity transfer requires an IDENTITY reference.",
        ) from exc
    scene_index = roles.index("SCENE") if "SCENE" in roles else None
    angle_index = roles.index("FACE_ANGLE") if "FACE_ANGLE" in roles else None
    return identity_index, scene_index, angle_index


def _run_strict_head_stage(
    *,
    db: Session,
    run: ImageStudioRun,
    request_state: dict[str, Any],
    result_state: dict[str, Any],
    stages: list[dict[str, Any]],
    source_base_uri: str,
    reference_uris: list[str],
) -> tuple[bytes, dict[str, Any], int]:
    roles = [str(role) for role in request_state.get("reference_roles") or []]
    identity_index, _, angle_index = _strict_reference_indexes(roles)

    base_bytes, _ = _read_image_uri(source_base_uri, label="strict_head_base")
    identity_bytes, _ = _read_image_uri(
        reference_uris[identity_index],
        label="strict_identity_reference",
    )
    angle_bytes: bytes | None = None
    if angle_index is not None:
        angle_bytes, _ = _read_image_uri(
            reference_uris[angle_index],
            label="strict_face_angle_reference",
        )

    tightness = str(request_state.get("head_edit_tightness") or "MEDIUM").upper()
    _set_stage(stages, "AUTO_CROP", "RUNNING")
    _persist_progress(db, run, request_state, result_state)

    prepared = prepare_strict_identity_assets(
        base_bytes=base_bytes,
        identity_bytes=identity_bytes,
        angle_bytes=angle_bytes,
        tightness=tightness,
    )

    assets = result_state.setdefault("assets", {})
    assets["base_face_crop"] = _store_bytes(
        run.run_id,
        "base_face_crop",
        crop_png(base_bytes, prepared.base_face_box),
        "image/png",
        ".png",
    )
    assets["base_head_crop"] = _store_bytes(
        run.run_id,
        "base_head_crop",
        prepared.base_head_crop,
        "image/png",
        ".png",
    )
    assets["identity_face_crop"] = _store_bytes(
        run.run_id,
        "identity_face_crop",
        prepared.identity_face_crop,
        "image/png",
        ".png",
    )
    assets["identity_head_crop"] = _store_bytes(
        run.run_id,
        "identity_head_crop",
        prepared.identity_head_crop,
        "image/png",
        ".png",
    )
    if prepared.angle_head_crop:
        assets["identity_angle_crop"] = _store_bytes(
            run.run_id,
            "identity_angle_crop",
            prepared.angle_head_crop,
            "image/png",
            ".png",
        )

    request_state["strict_geometry"] = {
        "base_face_box": prepared.base_face_box.as_list(),
        "base_head_box": prepared.base_head_box.as_list(),
        "identity_face_box": prepared.identity_face_box.as_list(),
        "identity_head_box": prepared.identity_head_box.as_list(),
    }
    _set_stage(stages, "AUTO_CROP", "DONE")
    _set_stage(stages, "AUTO_MASK", "RUNNING")
    _persist_progress(db, run, request_state, result_state)

    assets["mask_binary"] = _store_bytes(
        run.run_id,
        "mask_binary",
        prepared.head_mask,
        "image/png",
        ".png",
    )
    assets["mask_preview"] = _store_bytes(
        run.run_id,
        "mask_preview",
        prepared.head_mask_preview,
        "image/png",
        ".png",
    )
    _set_stage(stages, "AUTO_MASK", "DONE")

    strict_prompt = compile_strict_head_swap_prompt(
        str(request_state.get("user_prompt") or request_state.get("compiled_prompt") or "Strict head swap."),
        has_angle_reference=angle_index is not None,
        keep_hair_color=bool(request_state.get("keep_hair_color", False)),
        keep_base_hair_shape=bool(request_state.get("keep_base_hair_shape", False)),
        identity_strength=str(request_state.get("identity_strength") or "HIGH"),
        negative_prompt=str(request_state.get("user_negative_prompt") or ""),
    )
    request_state["compiled_prompt"] = strict_prompt.prompt
    request_state["negative_prompt"] = strict_prompt.negative_prompt

    _set_stage(stages, "HEAD_SWAP", "RUNNING")
    _persist_progress(db, run, request_state, result_state)

    head_references = [assets["identity_head_crop"]]
    if angle_index is not None and assets.get("identity_angle_crop"):
        head_references.append(assets["identity_angle_crop"])

    worker = invoke_image_studio_worker(
        base_image_uri=assets["base_head_crop"],
        reference_image_uris=head_references,
        source_run_id=run.run_id,
        prompt=strict_prompt.prompt,
        negative_prompt=strict_prompt.negative_prompt,
        steps=int(run.steps or 25),
        seed=run.seed,
        resolution_mode="target_long_edge",
        target_long_edge=(1024 if int(request_state.get("output_long_edge") or 1024) >= 1024 else 768),
        worker_mode=IMAGE_STUDIO_IDENTITY_V2_MODE,
        pipeline_stage="STRICT_HEAD_SWAP",
    )
    edited_head_bytes, _ = _read_image_uri(
        worker["result_uri"],
        label="strict_edited_head",
    )
    assets["edited_head_roi"] = _store_bytes(
        run.run_id,
        "edited_head_roi",
        edited_head_bytes,
        "image/png",
        ".png",
    )
    _set_stage(stages, "HEAD_SWAP", "DONE")

    _set_stage(stages, "COMPOSITE", "RUNNING")
    _persist_progress(db, run, request_state, result_state)
    final_bytes = composite_head_roi(
        base_bytes=base_bytes,
        edited_head_bytes=edited_head_bytes,
        head_box=prepared.base_head_box,
        roi_mask_bytes=prepared.head_mask,
    )
    result_state["strict_validation"] = measure_strict_composite(
        base_bytes=base_bytes,
        result_bytes=final_bytes,
        head_box=prepared.base_head_box,
    )
    _set_stage(stages, "COMPOSITE", "DONE")
    _persist_progress(db, run, request_state, result_state)
    return final_bytes, worker, int(worker.get("elapsed_ms") or 0)


def _execute_queued_run(run_id: str) -> str:
    """Execute one already-claimed RUNNING job in a fresh DB session."""

    db = SessionLocal()
    try:
        run = db.get(ImageStudioRun, str(run_id))
        if run is None or run.status != "RUNNING":
            return "SKIPPED"

        request_state = _request_for(run)
        reference_uris = _reference_uris_for(run)
        result_state = _result_for(run)
        stages = request_state.get("stages")
        if not isinstance(stages, list):
            stages = []
            request_state["stages"] = stages

        try:
            total_elapsed_ms = 0
            worker_protocol: dict[str, Any] | list[Any] | None = None
            mask_composited = False

            if run.mode in {"STRICT_HEAD_SWAP", "HEAD_SWAP_SCENE_TRANSFER"}:
                source_base_uri = run.base_image_uri
                protocols: list[Any] = []

                if run.mode == "STRICT_HEAD_SWAP" and bool(request_state.get("clean_output", True)):
                    clean_prompt = compile_clean_frame_prompt(
                        str(request_state.get("user_prompt") or "")
                    )
                    request_state["clean_frame_prompt"] = clean_prompt.prompt
                    request_state["clean_frame_negative_prompt"] = clean_prompt.negative_prompt
                    _set_stage(stages, "CLEAN_FRAME", "RUNNING")
                    _persist_progress(db, run, request_state, result_state)

                    clean_worker = invoke_image_studio_worker(
                        base_image_uri=run.base_image_uri,
                        reference_image_uris=[],
                        source_run_id=run.run_id,
                        prompt=clean_prompt.prompt,
                        negative_prompt=clean_prompt.negative_prompt,
                        steps=int(run.steps or 25),
                        seed=run.seed,
                        resolution_mode="target_long_edge",
                        target_long_edge=int(request_state.get("output_long_edge") or 1024),
                        worker_mode=IMAGE_STUDIO_IDENTITY_V2_MODE,
                        pipeline_stage="CLEAN_FRAME",
                    )
                    clean_bytes, _ = _read_image_uri(
                        clean_worker["result_uri"],
                        label="clean_frame_result",
                    )
                    clean_uri = _store_bytes(
                        run.run_id,
                        "clean_frame_result",
                        clean_bytes,
                        "image/png",
                        ".png",
                    )
                    result_state.setdefault("assets", {})["clean_frame_result"] = clean_uri
                    source_base_uri = clean_uri
                    total_elapsed_ms += int(clean_worker.get("elapsed_ms") or 0)
                    protocols.append(clean_worker.get("worker_protocol"))
                    _set_stage(stages, "CLEAN_FRAME", "DONE")
                    _persist_progress(db, run, request_state, result_state)

                if run.mode == "HEAD_SWAP_SCENE_TRANSFER":
                    roles = [str(role) for role in request_state.get("reference_roles") or []]
                    _, scene_index, _ = _strict_reference_indexes(roles)
                    if scene_index is None:
                        raise IdentityPreprocessError(
                            "SCENE_REFERENCE_REQUIRED",
                            "HEAD_SWAP_SCENE_TRANSFER requires a SCENE reference.",
                        )
                    scene_prompt = compile_scene_transfer_stage_prompt(
                        str(request_state.get("user_prompt") or "Transfer the subject into the scene reference."),
                        negative_prompt=str(request_state.get("user_negative_prompt") or ""),
                        clean_output=bool(request_state.get("clean_output", True)),
                    )
                    request_state["scene_prompt"] = scene_prompt.prompt
                    request_state["scene_negative_prompt"] = scene_prompt.negative_prompt
                    _set_stage(stages, "SCENE_TRANSFER", "RUNNING")
                    _persist_progress(db, run, request_state, result_state)

                    scene_worker = invoke_image_studio_worker(
                        base_image_uri=run.base_image_uri,
                        reference_image_uris=[reference_uris[scene_index]],
                        source_run_id=run.run_id,
                        prompt=scene_prompt.prompt,
                        negative_prompt=scene_prompt.negative_prompt,
                        steps=int(run.steps or 25),
                        seed=run.seed,
                        resolution_mode="target_long_edge",
                        target_long_edge=int(request_state.get("output_long_edge") or 1024),
                        worker_mode=IMAGE_STUDIO_IDENTITY_V2_MODE,
                        pipeline_stage="SCENE_TRANSFER",
                    )
                    scene_bytes, _ = _read_image_uri(
                        scene_worker["result_uri"],
                        label="scene_stage_result",
                    )
                    scene_uri = _store_bytes(
                        run.run_id,
                        "scene_stage_result",
                        scene_bytes,
                        "image/png",
                        ".png",
                    )
                    result_state.setdefault("assets", {})["scene_stage_result"] = scene_uri
                    source_base_uri = scene_uri
                    total_elapsed_ms += int(scene_worker.get("elapsed_ms") or 0)
                    protocols.append(scene_worker.get("worker_protocol"))
                    _set_stage(stages, "SCENE_TRANSFER", "DONE")
                    _persist_progress(db, run, request_state, result_state)

                final_bytes, head_worker, head_elapsed = _run_strict_head_stage(
                    db=db,
                    run=run,
                    request_state=request_state,
                    result_state=result_state,
                    stages=stages,
                    source_base_uri=source_base_uri,
                    reference_uris=reference_uris,
                )
                total_elapsed_ms += head_elapsed
                protocols.append(head_worker.get("worker_protocol"))
                worker_protocol = {"pipeline": run.mode, "stages": protocols}
                run.seed = head_worker.get("seed", run.seed)
                mask_composited = True
            else:
                worker = invoke_image_studio_worker(
                    base_image_uri=run.base_image_uri,
                    reference_image_uris=reference_uris,
                    source_run_id=run.run_id,
                    prompt=str(request_state.get("compiled_prompt") or ""),
                    negative_prompt=str(request_state.get("negative_prompt") or ""),
                    steps=int(run.steps or 25),
                    seed=run.seed,
                    resolution_mode="target_long_edge",
                    target_long_edge=int(request_state.get("output_long_edge") or 1024),
                )
                generated_bytes, _ = _read_image_uri(
                    worker["result_uri"],
                    label="image_studio_output",
                )
                _set_stage(stages, "qwen_generation", "DONE")

                if run.mask_uri:
                    _set_stage(stages, "mask_composite", "RUNNING")
                    base_bytes, _ = _read_image_uri(run.base_image_uri, label="image_studio_base")
                    mask_bytes, _ = _read_image_uri(run.mask_uri, label="image_studio_mask")
                    final_bytes = _mask_composite(base_bytes, generated_bytes, mask_bytes)
                    _set_stage(stages, "mask_composite", "DONE")
                    mask_composited = True
                else:
                    with Image.open(io.BytesIO(generated_bytes)) as generated_source:
                        rgb = generated_source.convert("RGB")
                        output = io.BytesIO()
                        rgb.save(output, format="PNG")
                        final_bytes = output.getvalue()

                run.seed = worker.get("seed", run.seed)
                total_elapsed_ms = int(worker.get("elapsed_ms") or 0)
                worker_protocol = worker.get("worker_protocol")

            original_base_bytes, _ = _read_image_uri(
                run.base_image_uri,
                label="image_studio_output_aspect_base",
            )
            final_bytes, final_size = _resize_png_to_long_edge(
                final_bytes,
                long_edge=int(request_state.get("output_long_edge") or 1024),
                aspect_source_bytes=original_base_bytes,
            )
            result_state["output_size"] = {
                "width": final_size[0],
                "height": final_size[1],
                "long_edge": int(request_state.get("output_long_edge") or 1024),
                "aspect_source": "BASE",
            }
            _set_stage(stages, "OUTPUT_RESIZE", "DONE")

            output_uri = _store_bytes(
                run.run_id,
                "output",
                final_bytes,
                "image/png",
                ".png",
            )
            run.output_image_uri = output_uri
            run.elapsed_ms = total_elapsed_ms or None
            result_state.update(
                {
                    "worker_protocol": worker_protocol,
                    "reference_count": len(reference_uris),
                    "mask_composited": mask_composited,
                    "pipeline_version": "identity_transfer_v2"
                    if run.mode in {"STRICT_HEAD_SWAP", "HEAD_SWAP_SCENE_TRANSFER"}
                    else "image_studio_v1",
                }
            )
            run.request_json = json.dumps(request_state, ensure_ascii=False)
            run.result_json = json.dumps(result_state, ensure_ascii=False)
            run.status = "SUCCESS"
            run.finished_at = _utcnow()
            db.commit()
            return "SUCCESS"
        except PortraitWorkerError as exc:
            if exc.error_code == "QWEN_WORKER_NOT_READY":
                active_stage = next(
                    (
                        stage
                        for stage in reversed(stages)
                        if isinstance(stage, dict) and stage.get("status") == "RUNNING"
                    ),
                    None,
                )
                if active_stage is not None:
                    active_stage["status"] = "QUEUED"
                run.status = "QUEUED"
                run.started_at = None
                run.finished_at = None
                run.error_code = None
                run.error_message = None
                _persist_progress(db, run, request_state, result_state)
                return "REQUEUED"

            active_stage = next(
                (
                    stage
                    for stage in reversed(stages)
                    if isinstance(stage, dict) and stage.get("status") == "RUNNING"
                ),
                None,
            )
            if active_stage is not None:
                active_stage["status"] = "FAILED"
            run.status = "FAILED"
            run.error_code = exc.error_code
            run.error_message = str(exc)[:3000]
            run.finished_at = _utcnow()
            _persist_progress(db, run, request_state, result_state)
            return "FAILED"
        except IdentityPreprocessError as exc:
            active_stage = next(
                (
                    stage
                    for stage in reversed(stages)
                    if isinstance(stage, dict) and stage.get("status") == "RUNNING"
                ),
                None,
            )
            if active_stage is not None:
                active_stage["status"] = "FAILED"
            run.status = "FAILED"
            run.error_code = exc.code
            run.error_message = str(exc)[:3000]
            run.finished_at = _utcnow()
            _persist_progress(db, run, request_state, result_state)
            return "FAILED"
        except Exception as exc:
            active_stage = next(
                (
                    stage
                    for stage in reversed(stages)
                    if isinstance(stage, dict) and stage.get("status") == "RUNNING"
                ),
                None,
            )
            if active_stage is not None:
                active_stage["status"] = "FAILED"
            safe_message = f"{exc.__class__.__name__}: {exc}"[:2000]
            run.status = "FAILED"
            run.error_code = "IMAGE_STUDIO_FAILED"
            run.error_message = safe_message
            run.finished_at = _utcnow()
            _persist_progress(db, run, request_state, result_state)
            return "FAILED"
    finally:
        db.close()


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
    resolution_mode: str = Form(default="target_long_edge"),
    output_long_edge: int = Form(default=1024),
    clean_output: bool = Form(default=True),
    identity_strength: str = Form(default="HIGH"),
    head_edit_tightness: str = Form(default="MEDIUM"),
    keep_hair_color: bool = Form(default=False),
    keep_base_hair_shape: bool = Form(default=False),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    mode_value = str(mode or "").strip().upper()
    reference_uploads = list(references or [])
    max_references = 3 if mode_value == "HEAD_SWAP_SCENE_TRANSFER" else 2
    if len(reference_uploads) > max_references:
        raise HTTPException(
            status_code=422,
            detail=f"{mode_value or 'Image Studio'} 最多支持 {max_references} 张参考图",
        )
    if not 1 <= int(steps) <= 100:
        raise HTTPException(status_code=422, detail="steps 必须在 1 到 100 之间")
    if resolution_mode not in {"current", "real_768", "target_long_edge"}:
        raise HTTPException(status_code=422, detail="resolution_mode 必须是 current / real_768 / target_long_edge")
    if int(output_long_edge) not in {768, 1024, 1536, 2048}:
        raise HTTPException(status_code=422, detail="output_long_edge 必须是 768 / 1024 / 1536 / 2048")

    identity_strength_value = str(identity_strength or "HIGH").strip().upper()
    if identity_strength_value not in {"LOW", "MEDIUM", "HIGH"}:
        raise HTTPException(status_code=422, detail="identity_strength 必须是 LOW / MEDIUM / HIGH")
    tightness_value = str(head_edit_tightness or "MEDIUM").strip().upper()
    if tightness_value not in {"TIGHT", "MEDIUM", "LOOSE"}:
        raise HTTPException(status_code=422, detail="head_edit_tightness 必须是 TIGHT / MEDIUM / LOOSE")

    try:
        seed_value = int(seed) if str(seed or "").strip() else None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="seed 必须是整数") from exc

    roles = _parse_roles(reference_roles, mode_value, len(reference_uploads))

    if mode_value == "STRICT_HEAD_SWAP":
        if not roles or roles[0] != "IDENTITY":
            raise HTTPException(status_code=422, detail="STRICT_HEAD_SWAP 第一张参考图必须是 IDENTITY")
        if len(roles) > 1 and roles[1] != "FACE_ANGLE":
            raise HTTPException(status_code=422, detail="STRICT_HEAD_SWAP 第二张参考图只能是 FACE_ANGLE")
    elif mode_value == "HEAD_SWAP_SCENE_TRANSFER":
        expected = ["IDENTITY", "SCENE"] + (["FACE_ANGLE"] if len(roles) == 3 else [])
        if roles != expected:
            raise HTTPException(
                status_code=422,
                detail="HEAD_SWAP_SCENE_TRANSFER 参考顺序必须是 IDENTITY, SCENE, 可选 FACE_ANGLE",
            )

    # BUSY with a loaded model is healthy for the durable FIFO queue.
    _require_worker_ready()

    try:
        if mode_value == "STRICT_HEAD_SWAP":
            compiled = compile_strict_head_swap_prompt(
                prompt,
                has_angle_reference="FACE_ANGLE" in roles,
                keep_hair_color=bool(keep_hair_color),
                keep_base_hair_shape=bool(keep_base_hair_shape),
                identity_strength=identity_strength_value,
                negative_prompt=negative_prompt,
                clean_output=bool(clean_output),
            )
            scene_compiled = None
        elif mode_value == "HEAD_SWAP_SCENE_TRANSFER":
            compiled = compile_strict_head_swap_prompt(
                prompt,
                has_angle_reference="FACE_ANGLE" in roles,
                keep_hair_color=bool(keep_hair_color),
                keep_base_hair_shape=bool(keep_base_hair_shape),
                identity_strength=identity_strength_value,
                negative_prompt=negative_prompt,
                clean_output=bool(clean_output),
            )
            scene_compiled = compile_scene_transfer_stage_prompt(
                prompt,
                negative_prompt=negative_prompt,
                clean_output=bool(clean_output),
            )
        else:
            compiled = compile_image_studio_prompt(
                prompt,
                mode=mode_value,
                preservation=preservation,
                reference_roles=roles,
                negative_prompt=negative_prompt,
                clean_output=bool(clean_output),
            )
            scene_compiled = None
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

    if mode_value == "STRICT_HEAD_SWAP":
        stages = [
            {"name": "INPUT", "status": "DONE"},
            {"name": "PROMPT_COMPILE", "status": "DONE"},
            {"name": "CLEAN_FRAME", "status": "QUEUED" if clean_output else "SKIPPED"},
            {"name": "AUTO_CROP", "status": "QUEUED"},
            {"name": "AUTO_MASK", "status": "QUEUED"},
            {"name": "HEAD_SWAP", "status": "QUEUED"},
            {"name": "COMPOSITE", "status": "QUEUED"},
            {"name": "OUTPUT_RESIZE", "status": "QUEUED"},
        ]
    elif mode_value == "HEAD_SWAP_SCENE_TRANSFER":
        stages = [
            {"name": "INPUT", "status": "DONE"},
            {"name": "PROMPT_COMPILE", "status": "DONE"},
            {"name": "SCENE_TRANSFER", "status": "QUEUED"},
            {"name": "AUTO_CROP", "status": "QUEUED"},
            {"name": "AUTO_MASK", "status": "QUEUED"},
            {"name": "HEAD_SWAP", "status": "QUEUED"},
            {"name": "COMPOSITE", "status": "QUEUED"},
            {"name": "OUTPUT_RESIZE", "status": "QUEUED"},
        ]
    else:
        stages = [
            {"name": "input", "status": "DONE"},
            {"name": "prompt_compile", "status": "DONE"},
            {"name": "qwen_generation", "status": "QUEUED"},
            {"name": "OUTPUT_RESIZE", "status": "QUEUED"},
        ]

    request_state = {
        "pipeline_version": (
            "identity_transfer_v2"
            if mode_value in {"STRICT_HEAD_SWAP", "HEAD_SWAP_SCENE_TRANSFER"}
            else "image_studio_v1"
        ),
        "reference_roles": roles,
        "user_prompt": str(prompt or "").strip(),
        "user_negative_prompt": str(negative_prompt or "").strip(),
        "compiled_prompt": compiled.prompt,
        "negative_prompt": compiled.negative_prompt,
        "scene_prompt": scene_compiled.prompt if scene_compiled else None,
        "scene_negative_prompt": scene_compiled.negative_prompt if scene_compiled else None,
        "resolution_mode": "target_long_edge",
        "output_long_edge": int(output_long_edge),
        "clean_output": bool(clean_output),
        "identity_strength": identity_strength_value,
        "head_edit_tightness": tightness_value,
        "keep_hair_color": bool(keep_hair_color),
        "keep_base_hair_shape": bool(keep_base_hair_shape),
        "preserve_clothing": True,
        "preserve_body_shape": True,
        "preserve_background": mode_value != "HEAD_SWAP_SCENE_TRANSFER",
        "queue_policy": "FIFO_SINGLE_L4",
        "stages": stages,
    }
    run = ImageStudioRun(
        run_id=run_id,
        status="QUEUED",
        mode=mode_value,
        preservation=(
            "MAX" if mode_value == "STRICT_HEAD_SWAP"
            else "STRONG" if mode_value == "HEAD_SWAP_SCENE_TRANSFER"
            else compiled.preservation
        ),
        model_version=MODEL_ID,
        request_json=json.dumps(request_state, ensure_ascii=False),
        result_json="{}",
        base_image_uri=base_uri,
        reference_uris_json=json.dumps(reference_uris, ensure_ascii=False),
        mask_uri=mask_uri,
        seed=seed_value,
        steps=int(steps),
        started_at=None,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    payload = _response_with_queue(db, run)
    enqueue_image_studio_queue()
    return payload


@router.get("/runs")
def list_runs(db: Session = Depends(get_db)) -> dict[str, Any]:
    rows = list(
        db.scalars(
            select(ImageStudioRun)
            .order_by(ImageStudioRun.created_at.desc())
            .limit(50)
        )
    )
    return {"items": [_response_with_queue(db, run) for run in rows], "count": len(rows), "queue": _queue_snapshot(db)}


@router.get("/runs/{run_id}")
def get_run(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    run = db.get(ImageStudioRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Image Studio run 不存在")
    return _response_with_queue(db, run)


@router.get("/queue")
def get_queue(db: Session = Depends(get_db)) -> dict[str, Any]:
    return _queue_snapshot(db)


@router.delete("/runs/failed")
def delete_all_failed_runs(db: Session = Depends(get_db)) -> dict[str, Any]:
    failed_runs = list(
        db.scalars(
            select(ImageStudioRun)
            .where(ImageStudioRun.status == "FAILED")
            .order_by(ImageStudioRun.created_at.asc(), ImageStudioRun.run_id.asc())
        )
    )
    deleted_runs = 0
    deleted_assets = 0
    for run in failed_runs:
        result = _delete_failed_run(db, run)
        deleted_runs += 1
        deleted_assets += int(result.get("deleted_assets") or 0)
    return {
        "deleted": True,
        "deleted_runs": deleted_runs,
        "deleted_assets": deleted_assets,
    }


@router.delete("/runs/{run_id}")
def delete_failed_run(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    run = db.get(ImageStudioRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Image Studio run 不存在")
    return _delete_failed_run(db, run)


def _media_uri_for(run: ImageStudioRun, kind: str) -> str | None:
    reference_uris = _reference_uris_for(run)
    if kind == "base":
        return run.base_image_uri
    if kind == "mask":
        return run.mask_uri
    if kind.startswith("reference_"):
        try:
            index = int(kind.split("_", 1)[1]) - 1
            return reference_uris[index]
        except (ValueError, IndexError):
            return None
    if kind == "output":
        return run.output_image_uri

    result = _result_for(run)
    assets = result.get("assets")
    if isinstance(assets, dict):
        value = assets.get(kind)
        return str(value) if value else None
    return None


def _download_filename(run: ImageStudioRun, kind: str, media_type: str | None) -> str:
    extension = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }.get(str(media_type or "").lower(), ".bin")
    safe_kind = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in kind)
    return f"{run.run_id}_{safe_kind}{extension}"


@router.get("/runs/{run_id}/media/{kind}")
def get_media(run_id: str, kind: str, db: Session = Depends(get_db)) -> Response:
    run = db.get(ImageStudioRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Image Studio run 不存在")
    uri = _media_uri_for(run, kind)
    if not uri:
        raise HTTPException(status_code=404, detail="媒体资源不存在")
    try:
        data, media_type = _read_image_uri(uri, label=f"image_studio_{kind}")
    except Exception as exc:
        raise HTTPException(status_code=404, detail="媒体资源不可读取") from exc
    return Response(content=data, media_type=media_type or "application/octet-stream")


@router.get("/runs/{run_id}/download/{kind}")
def download_media(run_id: str, kind: str, db: Session = Depends(get_db)) -> Response:
    run = db.get(ImageStudioRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Image Studio run 不存在")
    uri = _media_uri_for(run, kind)
    if not uri:
        raise HTTPException(status_code=404, detail="媒体资源不存在")
    try:
        data, media_type = _read_image_uri(uri, label=f"image_studio_{kind}")
    except Exception as exc:
        raise HTTPException(status_code=404, detail="媒体资源不可读取") from exc
    resolved_type = media_type or "application/octet-stream"
    filename = _download_filename(run, kind, resolved_type)
    return Response(
        content=data,
        media_type=resolved_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


__all__ = ["STORAGE_TYPE", "router", "_mask_composite"]
