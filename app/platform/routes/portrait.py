"""Fish Portrait POC orchestration API.

This module owns only the experimental Fish Portrait path. It reuses the
existing DatasetItem, FishAsset, PipelineRun and PlatformOperationLog tables;
the actual SDXL + IP-Adapter inference runs on a dedicated GPU worker.
"""
from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import secrets
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from google.cloud import storage
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.dataset_models import DatasetItem
from app.db import SessionLocal, get_db
from app.models import DatasetVersion, ImageAsset
from app.platform.models import FishAsset, PipelineRun
from app.fish_knowledge.asset_types import COVER_ASSET_TYPES
from app.platform.services import adapters
from app.portrait_worker_client import (
    PortraitWorkerError,
    check_portrait_worker,
    invoke_portrait_worker,
)
from app.species_policy import TARGET_SPECIES_PRESETS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/platform", tags=["fish-portrait-poc"])

PIPELINE_TYPE = "FISH_PORTRAIT_POC"
MODEL_ID = "sdxl_ip_adapter"
MODEL_LABEL = "SDXL + IP-Adapter"
DEFAULT_PARAMS = {
    "ip_scale": 0.8,
    "steps": 25,
    "width": 768,
    "height": 768,
}
STAGES = ("load_source", "load_reference", "sdxl_generate", "persist_result")
ACTIVE_JOB_STATUSES = {"PENDING", "RUNNING"}
REFERENCE_STATUSES = {"ACTIVE", "READY", "PUBLISHED", "DRAFT"}
PORTRAIT_PREFIX = "PORTRAIT_"


class PortraitParams(BaseModel):
    ip_scale: float = Field(default=0.8, ge=0.0, le=1.5)
    steps: int = Field(default=25, ge=1, le=100)
    width: int = Field(default=768, ge=256, le=1536)
    height: int = Field(default=768, ge=256, le=1536)


class PortraitJobCreate(BaseModel):
    dataset_id: str | None = Field(default=None, max_length=128)
    dataset_version: str | None = Field(default=None, max_length=128)
    source_item_id: int | str
    reference_asset_id: str | None = Field(default=None, max_length=128)
    model: str = Field(default=MODEL_ID, max_length=64)
    params: PortraitParams = Field(default_factory=PortraitParams)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Any) -> str | None:
    return value.isoformat() if value else None


def _json(value: Any, fallback: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return fallback
    return parsed


def _params_dict(params: PortraitParams) -> dict[str, Any]:
    if hasattr(params, "model_dump"):
        return params.model_dump()
    return params.dict()


def _normalize(value: Any) -> str:
    return str(value or "").strip().casefold()


def _species_context(value: Any) -> dict[str, Any]:
    raw = str(value or "").strip()
    normalized = _normalize(raw)
    for preset in TARGET_SPECIES_PRESETS:
        candidates = {
            str(preset.get("species_key") or ""),
            str(preset.get("common_name_zh") or ""),
            str(preset.get("common_name_en") or ""),
            *(str(alias) for alias in (preset.get("aliases") or [])),
        }
        if normalized in {_normalize(candidate) for candidate in candidates if candidate}:
            return {
                "species_id": str(preset["species_key"]),
                "species_name": str(preset["common_name_zh"]),
                "species_name_en": str(preset.get("common_name_en") or ""),
            }
    return {
        "species_id": raw,
        "species_name": raw,
        "species_name_en": "",
    }


def _species_matches(value: Any, wanted: dict[str, Any] | str) -> bool:
    expected = wanted if isinstance(wanted, dict) else _species_context(wanted)
    actual = _species_context(value)
    if _normalize(value) == _normalize(expected.get("species_id")):
        return True
    return bool(
        expected.get("species_id")
        and actual.get("species_id")
        and _normalize(actual["species_id"]) == _normalize(expected["species_id"])
    )


def _status(value: Any) -> str:
    return str(value or "UNKNOWN").strip().upper()


def _new_run_id() -> str:
    return "PORTRAIT_" + _utcnow().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)


def _source_image_url(item: DatasetItem) -> str:
    return f"/media/{item.batch_id}/{item.image_id}"


def _asset_media_url(asset_id: str, kind: str = "transparent") -> str:
    return f"/api/platform/assets/{asset_id}/media/{kind}"


def _reference_uri(row: FishAsset) -> str | None:
    return (row.asset_uri or row.transparent_uri or "").strip() or None


def _reference_rows(db: Session, species_id: str, asset_type: str = "transparent") -> list[FishAsset]:
    requested_type = str(asset_type or "transparent").strip().lower()
    aliases = {
        "transparent": None,
        "transparent_main": "COVER_CARD_TRANSPARENT_LEFT",
        "transparent_left": "COVER_CARD_TRANSPARENT_LEFT",
        "left": "COVER_CARD_TRANSPARENT_LEFT",
        "transparent_alt": "COVER_CARD_TRANSPARENT_RIGHT",
        "transparent_right": "COVER_CARD_TRANSPARENT_RIGHT",
        "right": "COVER_CARD_TRANSPARENT_RIGHT",
    }
    if requested_type not in aliases:
        raise HTTPException(status_code=400, detail="仅支持 transparent 鱼体参考资产")
    wanted = _species_context(species_id)
    rows = db.scalars(
        select(FishAsset)
        .where(
            FishAsset.asset_type.in_(COVER_ASSET_TYPES)
            | FishAsset.transparent_uri.is_not(None)
        )
        .order_by(FishAsset.created_at.desc(), FishAsset.asset_id.desc())
    ).all()
    matched: list[FishAsset] = []
    for row in rows:
        if not _reference_uri(row) or _status(row.status) not in REFERENCE_STATUSES:
            continue
        if str(row.asset_id or "").upper().startswith(PORTRAIT_PREFIX):
            continue
        if not _species_matches(row.species, wanted):
            continue
        canonical = str(row.asset_type or "").upper()
        if aliases[requested_type] and canonical != aliases[requested_type]:
            continue
        matched.append(row)
    canonical_rows = [row for row in matched if str(row.asset_type or "").upper() in COVER_ASSET_TYPES]
    legacy_rows = [row for row in matched if row not in canonical_rows]
    matched = sorted(
        canonical_rows + legacy_rows,
        key=lambda row: (
            0 if str(row.asset_type or "").upper().endswith("LEFT") else
            1 if str(row.asset_type or "").upper().endswith("RIGHT") else 2,
            -(row.created_at.timestamp() if row.created_at else 0),
            str(row.asset_id),
        ),
    )
    if requested_type in {"transparent_main", "transparent_left", "left"}:
        return matched[:1]
    if requested_type in {"transparent_alt", "transparent_right", "right"}:
        canonical_right = [row for row in matched if str(row.asset_type or "").upper().endswith("RIGHT")]
        return canonical_right[:1] if canonical_right else matched[1:]
    return matched


def _reference_dto(row: FishAsset, species_id: str, index: int) -> dict[str, Any]:
    context = _species_context(species_id)
    canonical = str(row.asset_type or "").upper()
    if canonical.endswith("LEFT"):
        reference_type = "LEFT"
    elif canonical.endswith("RIGHT"):
        reference_type = "RIGHT"
    else:
        reference_type = "transparent_main" if index == 0 else "transparent_alt"
    return {
        "asset_id": row.asset_id,
        "type": reference_type,
        "url": _reference_uri(row) if row.asset_uri else _asset_media_url(row.asset_id),
        "species_id": context["species_id"],
        "species_name": context["species_name"],
        "version": row.version,
        "status": _status(row.status),
        "source": {
            "batch_id": row.source_batch_id,
            "image_id": row.source_image_id,
        },
    }



def _resolve_source_item(db: Session, dataset_id: str, source_item_id: int | str) -> DatasetItem:
    candidate = str(source_item_id).strip()
    item = None
    if candidate.isdigit():
        item = db.scalar(
            select(DatasetItem).where(
                DatasetItem.dataset_version == dataset_id,
                DatasetItem.id == int(candidate),
            )
        )
    if item is None:
        item = db.scalar(
            select(DatasetItem).where(
                DatasetItem.dataset_version == dataset_id,
                DatasetItem.image_id == candidate,
            )
        )
    if item is None:
        raise HTTPException(status_code=404, detail="数据集图片不存在")
    if not str(item.gcs_uri or "").strip():
        raise HTTPException(status_code=422, detail="数据集图片没有可读取的存储地址")
    return item


def _source_state(db: Session, item: DatasetItem) -> dict[str, Any]:
    image = db.get(ImageAsset, item.image_asset_id)
    return {
        "item_id": item.id,
        "image_id": item.image_id,
        "batch_id": item.batch_id,
        "species_id": item.species_key,
        "species_name": item.species_name,
        "split": item.split,
        "uri": item.gcs_uri,
        "review_status": getattr(image, "review_status", None),
        "source": getattr(image, "source_platform", None) or "DATASET_FREEZE",
    }


def _public_source(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_id": source.get("item_id"),
        "image_id": source.get("image_id"),
        "batch_id": source.get("batch_id"),
        "species_id": source.get("species_id"),
        "species_name": source.get("species_name"),
        "split": source.get("split"),
        "review_status": source.get("review_status"),
        "source": source.get("source"),
        "image_url": (
            f"/media/{source['batch_id']}/{source['image_id']}"
            if source.get("batch_id") and source.get("image_id")
            else None
        ),
    }


def _public_reference(reference: dict[str, Any] | None) -> dict[str, Any] | None:
    if not reference:
        return None
    return {
        "asset_id": reference.get("asset_id"),
        "species_id": reference.get("species_id"),
        "species_name": reference.get("species_name"),
        "type": reference.get("type", "transparent_main"),
        "url": reference.get("url") or (_asset_media_url(str(reference["asset_id"])) if reference.get("asset_id") else None),
        "version": reference.get("version"),
    }


def _initial_state(
    *,
    dataset_id: str,
    source: dict[str, Any],
    reference: FishAsset,
    params: dict[str, Any],
) -> dict[str, Any]:
    reference_context = _species_context(reference.species)
    return {
        "request": {
            "dataset_id": dataset_id,
            "source_item_id": str(source["item_id"]),
            "reference_asset_id": reference.asset_id,
            "model": MODEL_ID,
            "params": params,
        },
        "source": source,
        "reference": {
            "asset_id": reference.asset_id,
            "species_id": reference_context["species_id"],
            "species_name": reference_context["species_name"],
            "uri": _reference_uri(reference),
            "url": _reference_uri(reference),
            "version": reference.version,
        },
        "stages": [{"name": stage, "status": "PENDING"} for stage in STAGES],
        "worker": None,
        "result": None,
        "error": None,
    }


def _find_active_job(
    db: Session,
    *,
    dataset_id: str,
    source_item_id: str,
    reference_asset_id: str,
    params: dict[str, Any],
) -> PipelineRun | None:
    rows = db.scalars(
        select(PipelineRun)
        .where(
            PipelineRun.pipeline_type == PIPELINE_TYPE,
            PipelineRun.status.in_(ACTIVE_JOB_STATUSES),
        )
        .order_by(PipelineRun.created_at.desc(), PipelineRun.run_id.desc())
    ).all()
    for row in rows:
        state = _json(row.stage_json, {})
        request = state.get("request", {}) if isinstance(state, dict) else {}
        if (
            str(request.get("dataset_id") or "") == dataset_id
            and str(request.get("source_item_id") or "") == source_item_id
            and str(request.get("reference_asset_id") or "") == reference_asset_id
            and request.get("params") == params
        ):
            return row
    return None


def _stage(state: dict[str, Any], name: str, status: str, *, error: str | None = None) -> None:
    for item in state.get("stages", []):
        if item.get("name") != name:
            continue
        now = _utcnow()
        item["status"] = status
        if status == "RUNNING":
            item["started_at"] = now.isoformat()
        if status in {"DONE", "FAILED"}:
            item["finished_at"] = now.isoformat()
            started = item.get("started_at")
            if started:
                try:
                    item["duration_ms"] = max(
                        0,
                        int((now - datetime.fromisoformat(started)).total_seconds() * 1000),
                    )
                except (TypeError, ValueError):
                    item["duration_ms"] = None
        if error:
            item["error"] = error
        return


def _set_state(run: PipelineRun, state: dict[str, Any]) -> None:
    run.stage_json = json.dumps(state, ensure_ascii=False)


def _public_run(run: PipelineRun, state: dict[str, Any]) -> dict[str, Any]:
    result = state.get("result") if isinstance(state, dict) else None
    return {
        "run_id": run.run_id,
        "id": run.run_id,
        "task_id": run.run_id,
        "type": run.pipeline_type,
        "status": _status(run.status),
        "stage": run.current_stage,
        "current_stage": run.current_stage,
        "steps": state.get("stages", []) if isinstance(state, dict) else [],
        "stages": state.get("stages", []) if isinstance(state, dict) else [],
        "source": _public_source(state.get("source", {})) if isinstance(state, dict) else None,
        "reference": _public_reference(state.get("reference")) if isinstance(state, dict) else None,
        "result": {
            "asset_id": result.get("asset_id"),
            "generated_image": _asset_media_url(str(result["asset_id"]))
            if result and result.get("asset_id")
            else None,
        }
        if isinstance(result, dict)
        else None,
        "worker": state.get("worker") if isinstance(state, dict) else None,
        "error_code": (state.get("error") or {}).get("code") if isinstance(state, dict) else None,
        "error_message": run.error_message,
        "created_at": _iso(run.created_at),
        "started_at": _iso(run.started_at),
        "finished_at": _iso(run.finished_at),
        "duration_ms": run.duration_ms,
    }


def _fail_job(
    db: Session,
    run: PipelineRun,
    state: dict[str, Any],
    *,
    stage: str,
    error_code: str,
    message: str,
) -> None:
    safe_message = str(message or error_code)[:3000]
    _stage(state, stage, "FAILED", error=f"{error_code}: {safe_message}")
    state["error"] = {"code": error_code, "message": safe_message}
    run.status = "FAILED"
    run.current_stage = stage
    run.error_stage = stage
    run.error_message = f"{error_code}: {safe_message}"
    run.finished_at = _utcnow()
    if run.started_at:
        run.duration_ms = max(0, int((run.finished_at - run.started_at).total_seconds() * 1000))
    _set_state(run, state)
    adapters.record_operation(
        db,
        "CREATE_PORTRAIT_RUN",
        "PIPELINE_RUN",
        run.run_id,
        status="FAILED",
        message=safe_message,
        detail={"error_code": error_code, "stage": stage},
    )
    db.commit()


def _output_bucket(source_uri: str) -> str:
    configured = os.getenv("GCS_BUCKET", "").strip()
    if configured:
        return configured
    if source_uri.startswith("gs://") and "/" in source_uri[5:]:
        return source_uri[5:].split("/", 1)[0]
    return ""


def _persist_output_bytes(run_id: str, data: bytes, media_type: str, source_uri: str) -> str:
    media_type = media_type.split(";", 1)[0].strip().lower()
    extension = mimetypes.guess_extension(media_type) or ".png"
    bucket_name = _output_bucket(source_uri)
    object_name = f"experiments/fish_portrait_poc/{run_id}/generated{extension}"
    if bucket_name:
        blob = storage.Client().bucket(bucket_name).blob(object_name)
        blob.upload_from_string(data, content_type=media_type or "image/png")
        return f"gs://{bucket_name}/{object_name}"
    path = Path("/tmp") / "yujian" / "fish_portrait_poc" / run_id / f"generated{extension}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def _decode_data_url(value: Any) -> tuple[bytes, str] | None:
    if not isinstance(value, str) or not value.startswith("data:") or "," not in value:
        return None
    header, encoded = value.split(",", 1)
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError):
        return None
    media_type = header[5:].split(";", 1)[0] or "image/png"
    return data, media_type


def _materialize_output(result: dict[str, Any], run_id: str, source_uri: str) -> str:
    data_url = _decode_data_url(result.get("generated_image")) or _decode_data_url(result.get("result_uri"))
    if data_url:
        return _persist_output_bytes(run_id, data_url[0], data_url[1], source_uri)
    uri = str(result.get("result_uri") or "").strip()
    if not uri:
        raise PortraitWorkerError("PORTRAIT_OUTPUT_MISSING", "Portrait worker did not return an output URI")
    if uri.startswith(("gs://", "/")):
        if uri.startswith("/") and not Path(uri).is_file():
            raise PortraitWorkerError("PORTRAIT_OUTPUT_NOT_FOUND", "Portrait worker output file does not exist")
        return uri
    if uri.startswith(("http://", "https://")):
        try:
            with urllib.request.urlopen(uri, timeout=120) as response:
                data = response.read()
                media_type = response.headers.get_content_type() or "image/png"
        except Exception as exc:
            raise PortraitWorkerError("PORTRAIT_OUTPUT_DOWNLOAD_FAILED", str(exc)) from exc
        return _persist_output_bytes(run_id, data, media_type, source_uri)
    raise PortraitWorkerError("PORTRAIT_OUTPUT_URI_INVALID", "Portrait worker returned an unsupported output URI")


def _execute_portrait_job(run_id: str) -> None:
    db = SessionLocal()
    active_stage = "load_source"
    try:
        run = db.get(PipelineRun, run_id)
        if run is None or run.pipeline_type != PIPELINE_TYPE:
            return
        state = _json(run.stage_json, {})
        if not isinstance(state, dict):
            state = {}
        run.status = "RUNNING"
        run.started_at = run.started_at or _utcnow()
        run.current_stage = active_stage
        _stage(state, active_stage, "RUNNING")
        _set_state(run, state)
        db.commit()

        source_uri = str((state.get("source") or {}).get("uri") or "").strip()
        reference_uri = str((state.get("reference") or {}).get("uri") or "").strip()
        if not source_uri:
            raise PortraitWorkerError("PORTRAIT_SOURCE_URI_INVALID", "source image URI is empty")
        _stage(state, active_stage, "DONE")
        _set_state(run, state)
        db.commit()

        active_stage = "load_reference"
        run.current_stage = active_stage
        _stage(state, active_stage, "RUNNING")
        _set_state(run, state)
        db.commit()
        if not reference_uri:
            raise PortraitWorkerError("PORTRAIT_REFERENCE_URI_INVALID", "reference asset URI is empty")
        _stage(state, active_stage, "DONE")
        _set_state(run, state)
        db.commit()

        active_stage = "sdxl_generate"
        run.current_stage = active_stage
        _stage(state, active_stage, "RUNNING")
        _set_state(run, state)
        db.commit()
        health = check_portrait_worker()
        state["worker"] = {
            "health_status": health.get("status"),
            "endpoint_configured": bool(health.get("endpoint_configured")),
        }
        if health.get("status") != "READY":
            code = (
                "PORTRAIT_WORKER_NOT_CONFIGURED"
                if not health.get("endpoint_configured")
                else "PORTRAIT_WORKER_NOT_READY"
            )
            raise PortraitWorkerError(code, str(health.get("reason") or "Portrait worker is not ready"))
        request = state.get("request") or {}
        worker_result = invoke_portrait_worker(
            source_image_uri=source_uri,
            reference_image_uri=reference_uri,
            dataset_id=str(request.get("dataset_id") or ""),
            source_item_id=str(request.get("source_item_id") or ""),
            reference_asset_id=str(request.get("reference_asset_id") or ""),
            model=MODEL_ID,
            params=dict(request.get("params") or DEFAULT_PARAMS),
        )
        state["worker"].update(
            {
                "status": worker_result.get("worker_status", "WORKER_EXECUTED"),
                "model_version": worker_result.get("model_version"),
                "inference_time_ms": worker_result.get("inference_time_ms"),
                "request_id": worker_result.get("request_id"),
            }
        )
        _stage(state, active_stage, "DONE")
        _set_state(run, state)
        db.commit()

        active_stage = "persist_result"
        run.current_stage = active_stage
        _stage(state, active_stage, "RUNNING")
        _set_state(run, state)
        db.commit()
        generated_uri = _materialize_output(worker_result, run_id, source_uri)
        asset_id = PORTRAIT_PREFIX + run_id
        asset = db.get(FishAsset, asset_id)
        if asset is None:
            asset = FishAsset(asset_id=asset_id)
            db.add(asset)
        asset.pipeline_run_id = run.run_id
        asset.source_batch_id = (state.get("source") or {}).get("batch_id")
        asset.source_image_id = (state.get("source") or {}).get("image_id")
        asset.species = (state.get("source") or {}).get("species_id") or (state.get("source") or {}).get("species_name")
        asset.status = "ACTIVE"
        asset.original_uri = source_uri
        asset.transparent_uri = generated_uri
        asset.version = "portrait-poc-v1"
        state["result"] = {
            "asset_id": asset_id,
            "generated_uri": generated_uri,
            "model": MODEL_LABEL,
            "params": request.get("params") or DEFAULT_PARAMS,
            "reference_asset_id": request.get("reference_asset_id"),
        }
        _stage(state, active_stage, "DONE")
        run.status = "SUCCESS"
        run.current_stage = "complete"
        run.finished_at = _utcnow()
        if run.started_at:
            run.duration_ms = max(0, int((run.finished_at - run.started_at).total_seconds() * 1000))
        _set_state(run, state)
        adapters.record_operation(
            db,
            "CREATE_PORTRAIT_RUN",
            "PIPELINE_RUN",
            run.run_id,
            status="SUCCESS",
            message="Fish Portrait 生成完成",
            detail={"asset_id": asset_id, "model": MODEL_ID},
        )
        db.commit()
    except PortraitWorkerError as exc:
        db.rollback()
        run = db.get(PipelineRun, run_id)
        if run is not None:
            state = _json(run.stage_json, {})
            if not isinstance(state, dict):
                state = {}
            _fail_job(
                db,
                run,
                state,
                stage=active_stage,
                error_code=exc.error_code,
                message=str(exc),
            )
        logger.exception("fish_portrait_worker_failed run_id=%s stage=%s code=%s", run_id, active_stage, exc.error_code)
    except Exception as exc:
        db.rollback()
        run = db.get(PipelineRun, run_id)
        if run is not None:
            state = _json(run.stage_json, {})
            if not isinstance(state, dict):
                state = {}
            _fail_job(
                db,
                run,
                state,
                stage=active_stage,
                error_code="PORTRAIT_INTERNAL_ERROR",
                message=str(exc),
            )
        logger.exception("fish_portrait_failed run_id=%s stage=%s", run_id, active_stage)
    finally:
        db.close()


@router.get("/datasets/{dataset_id}/items")
def portrait_dataset_items(
    dataset_id: str,
    species: str | None = None,
    status: str | None = None,
    keyword: str | None = None,
    page: int = Query(default=1, ge=1, le=10000),
    size: int = Query(default=60, ge=1, le=200),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    dataset = db.get(DatasetVersion, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="数据集不存在")
    if _status(dataset.status) != "FROZEN":
        raise HTTPException(status_code=409, detail="只能从已冻结 Dataset 选择图片")
    wanted_species = _normalize(species)
    wanted_status = _normalize(status)
    wanted_keyword = _normalize(keyword)
    rows = db.scalars(
        select(DatasetItem)
        .where(DatasetItem.dataset_version == dataset_id)
        .order_by(DatasetItem.id)
    ).all()
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if wanted_species and wanted_species not in {
            _normalize(row.species_key),
            _normalize(row.species_name),
        }:
            continue
        image = db.get(ImageAsset, row.image_asset_id)
        review_status = str(getattr(image, "review_status", "") or "UNKNOWN")
        presence_status = str(row.presence_status or "")
        if wanted_status and wanted_status not in {_normalize(review_status), _normalize(presence_status)}:
            continue
        haystack = " ".join(
            [
                str(row.image_id or ""),
                str(row.species_key or ""),
                str(row.species_name or ""),
            ]
        ).casefold()
        if wanted_keyword and wanted_keyword not in haystack:
            continue
        filtered.append(
            {
                "item_id": row.id,
                "dataset_item_id": row.id,
                "image_id": row.image_id,
                "species_id": row.species_key,
                "species_name": row.species_name,
                "image_url": _source_image_url(row),
                "preview_url": _source_image_url(row) + "?variant=thumbnail",
                "review_status": review_status,
                "source": getattr(image, "source_platform", None) or "DATASET_FREEZE",
                "batch_id": row.batch_id,
                "split": row.split,
            }
        )
    start = (page - 1) * size
    return {
        "items": filtered[start : start + size],
        "total": len(filtered),
        "page": page,
        "size": size,
    }


@router.get("/assets/fish-reference")
def fish_reference_assets(
    species_id: str = Query(..., min_length=1, max_length=128),
    asset_type: str = Query(default="transparent", max_length=32),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    rows = _reference_rows(db, species_id, asset_type)
    context = _species_context(species_id)
    return {
        "species_id": context["species_id"],
        "species_name": context["species_name"],
        "assets": [_reference_dto(row, context["species_id"], index) for index, row in enumerate(rows)],
        "references": [_reference_dto(row, context["species_id"], index) for index, row in enumerate(rows)],
    }


@router.get("/portrait/reference/{species_id}")
def portrait_reference(species_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    rows = _reference_rows(db, species_id, "transparent")
    if not rows:
        raise HTTPException(
            status_code=404,
            detail={
                "error_code": "REFERENCE_ASSET_NOT_FOUND",
                "message": "没有找到该鱼种的 transparent 标准鱼体资产",
                "species_id": species_id,
            },
        )
    context = _species_context(species_id)
    return {
        "species_id": context["species_id"],
        "species": context["species_name"],
        "reference_asset": _reference_dto(rows[0], context["species_id"], 0),
    }


@router.post("/portrait/jobs")
def create_portrait_job(
    payload: PortraitJobCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    dataset_id = str(payload.dataset_id or payload.dataset_version or "").strip()
    if not dataset_id:
        raise HTTPException(status_code=422, detail="dataset_id 不能为空")
    dataset = db.get(DatasetVersion, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="数据集不存在")
    if _status(dataset.status) != "FROZEN":
        raise HTTPException(status_code=409, detail="只能发布已冻结 Dataset 的实验")
    normalized_model = _normalize(payload.model).replace("+", "_").replace("-", "_").replace(" ", "_")
    if normalized_model not in {"sdxl_ip_adapter", "sdxl_ipadapter"}:
        raise HTTPException(
            status_code=422,
            detail={"error_code": "MODEL_NOT_SUPPORTED", "message": "当前仅支持 SDXL + IP-Adapter"},
        )
    item = _resolve_source_item(db, dataset_id, payload.source_item_id)
    source = _source_state(db, item)
    wanted_species = _species_context(item.species_key or item.species_name)
    reference = None
    if payload.reference_asset_id:
        reference = db.get(FishAsset, payload.reference_asset_id)
        if reference is None:
            raise HTTPException(status_code=404, detail="标准鱼体参考资产不存在")
        if not _reference_uri(reference) or _status(reference.status) not in REFERENCE_STATUSES:
            raise HTTPException(status_code=409, detail="参考资产没有可用的 transparent 图")
        if str(reference.asset_id or "").upper().startswith(PORTRAIT_PREFIX):
            raise HTTPException(status_code=409, detail="生成结果不能作为标准参考资产")
        if not _species_matches(reference.species, wanted_species):
            raise HTTPException(status_code=409, detail="参考鱼体与 A 图鱼种不匹配")
    else:
        rows = _reference_rows(db, wanted_species["species_id"], "transparent")
        if not rows:
            raise HTTPException(
                status_code=409,
                detail={
                    "error_code": "REFERENCE_ASSET_NOT_FOUND",
                    "message": "没有找到该鱼种的 transparent 标准鱼体资产",
                },
            )
        reference = rows[0]
    params = _params_dict(payload.params)
    existing = _find_active_job(
        db,
        dataset_id=dataset_id,
        source_item_id=str(item.id),
        reference_asset_id=reference.asset_id,
        params=params,
    )
    if existing is not None:
        state = _json(existing.stage_json, {})
        return {
            **_public_run(existing, state if isinstance(state, dict) else {}),
            "already_running": True,
        }
    run_id = _new_run_id()
    state = _initial_state(
        dataset_id=dataset_id,
        source=source,
        reference=reference,
        params=params,
    )
    run = PipelineRun(
        run_id=run_id,
        source_batch_id=item.batch_id,
        source_image_id=item.image_id,
        pipeline_type=PIPELINE_TYPE,
        status="PENDING",
        current_stage="queued",
        stage_json=json.dumps(state, ensure_ascii=False),
        model_version=MODEL_ID,
    )
    db.add(run)
    adapters.record_operation(
        db,
        "CREATE_PORTRAIT_RUN",
        "PIPELINE_RUN",
        run_id,
        detail={
            "dataset_id": dataset_id,
            "source_item_id": item.id,
            "reference_asset_id": reference.asset_id,
            "model": MODEL_ID,
        },
    )
    db.commit()
    background_tasks.add_task(_execute_portrait_job, run_id)
    return {
        **_public_run(run, state),
        "already_running": False,
    }


@router.get("/pipeline/{run_id}")
def portrait_pipeline_status(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    run = db.get(PipelineRun, run_id)
    if run is None or run.pipeline_type != PIPELINE_TYPE:
        raise HTTPException(status_code=404, detail="Fish Portrait 流水线不存在")
    state = _json(run.stage_json, {})
    return _public_run(run, state if isinstance(state, dict) else {})


@router.get("/portrait/results/{run_id}")
def portrait_result(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    run = db.get(PipelineRun, run_id)
    if run is None or run.pipeline_type != PIPELINE_TYPE:
        raise HTTPException(status_code=404, detail="Fish Portrait 结果不存在")
    state = _json(run.stage_json, {})
    state = state if isinstance(state, dict) else {}
    result = state.get("result") if isinstance(state.get("result"), dict) else {}
    source = state.get("source") if isinstance(state.get("source"), dict) else {}
    reference = state.get("reference") if isinstance(state.get("reference"), dict) else {}
    metadata = {
        "model": MODEL_LABEL,
        "model_id": MODEL_ID,
        "params": state.get("request", {}).get("params") or DEFAULT_PARAMS,
        "run_id": run.run_id,
        "species_id": source.get("species_id"),
        "species_name": source.get("species_name"),
        "reference_asset_id": reference.get("asset_id"),
        "worker": state.get("worker"),
    }
    return {
        "run_id": run.run_id,
        "status": _status(run.status),
        "source_image": _public_source(source).get("image_url"),
        "reference_image": _public_reference(reference).get("url") if reference else None,
        "generated_image": _asset_media_url(str(result["asset_id"])) if result.get("asset_id") else None,
        "metadata": metadata,
    }


__all__ = [
    "PortraitJobCreate",
    "PortraitParams",
    "PIPELINE_TYPE",
    "_execute_portrait_job",
    "router",
]
