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

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Response
from google.cloud import storage
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.dataset_models import DatasetItem
from app.db import SessionLocal, get_db
from app.factory import get_bucket_name
from app.fish_knowledge.cover import FishSpeciesCover
from app.fish_knowledge.gallery import managed_knowledge_asset_url
from app.fish_knowledge.import_batch import FishKnowledgeAssetVersion
from app.models import DatasetVersion, ImageAsset
from app.platform.models import FishAsset, PipelineRun
from app.platform.services import adapters
from app.portrait_worker_client import (
    INPAINT_DEFAULT_NEGATIVE_PROMPT,
    INPAINT_DEFAULT_PROMPT,
    PortraitWorkerError,
    check_portrait_worker,
    invoke_portrait_inpaint_worker,
    invoke_portrait_worker,
)
from app.species_policy import TARGET_SPECIES_PRESETS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/platform", tags=["fish-portrait-poc"])

PIPELINE_TYPE = "FISH_PORTRAIT_POC"
MODEL_ID = "sdxl_ip_adapter"
MODEL_LABEL = "SDXL + IP-Adapter"
INPAINT_MODE = "fish_preserve_inpaint_v2"
DUAL_IP_MODE = "dual_ip_adapter_v1"
INPAINT_MODEL_ID = "sdxl_inpaint"
INPAINT_MODEL_LABEL = "SDXL Inpaint"
DEFAULT_PARAMS = {
    "source_scale": 0.8,
    "reference_scale": 0.35,
    "steps": 25,
    "width": 768,
    "height": 768,
}
STAGES = ("load_source", "load_reference", "sdxl_generate", "persist_result")
INPAINT_STAGES = ("load_source", "load_masks", "sdxl_inpaint", "persist_result")
ACTIVE_JOB_STATUSES = {"PENDING", "RUNNING"}
REFERENCE_STATUSES = {"ACTIVE", "READY", "PUBLISHED"}
PORTRAIT_PREFIX = "PORTRAIT_"
KNOWLEDGE_REFERENCE_PREFIX = "KNOWLEDGE_COVER_"
REFERENCE_VARIANT_ORDER = (
    "COVER_CARD_TRANSPARENT_LEFT",
    "COVER_CARD_TRANSPARENT_RIGHT",
    "COVER_CARD",
)


class PortraitParams(BaseModel):
    source_scale: float = Field(default=0.8, ge=0.0, le=1.5)
    reference_scale: float = Field(default=0.35, ge=0.0, le=1.5)
    steps: int = Field(default=25, ge=1, le=100)
    width: int = Field(default=768, ge=256, le=1536)
    height: int = Field(default=768, ge=256, le=1536)


class PortraitInpaintParams(BaseModel):
    strength: float = Field(default=0.25, ge=0.1, le=0.5)
    steps: int = Field(default=25, ge=1, le=100)
    width: int = Field(default=768, ge=256, le=1536)
    height: int = Field(default=768, ge=256, le=1536)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)


class PortraitJobCreate(BaseModel):
    # New UI requests default to the preserve-inpaint pipeline.  The route
    # keeps a small compatibility fallback for older V1 clients that still
    # send reference_asset_id without an explicit mode.
    mode: str = Field(default=INPAINT_MODE, max_length=64)
    dataset_id: str | None = Field(default=None, max_length=128)
    dataset_version: str | None = Field(default=None, max_length=128)
    source_item_id: int | str | None = None
    reference_asset_id: str | None = Field(default=None, max_length=128)
    model: str = Field(default=MODEL_ID, max_length=64)
    params: PortraitParams = Field(default_factory=PortraitParams)
    original_image_uri: str | None = Field(default=None, max_length=4096)
    fish_mask_uri: str | None = Field(default=None, max_length=4096)
    completion_mask_uri: str | None = Field(default=None, max_length=4096)
    species: str | None = Field(default=None, max_length=128)
    prompt: str | None = Field(default=None, max_length=2000)
    negative_prompt: str | None = Field(default=None, max_length=2000)
    inpaint: PortraitInpaintParams = Field(default_factory=PortraitInpaintParams)
    # Accept the worker contract's flat V2 fields as well as the nested UI
    # object.  Flat values win when both forms are supplied.
    strength: float | None = Field(default=None, ge=0.1, le=0.5)
    steps: int | None = Field(default=None, ge=1, le=100)
    width: int | None = Field(default=None, ge=256, le=1536)
    height: int | None = Field(default=None, ge=256, le=1536)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)


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


def _inpaint_params_dict(params: PortraitInpaintParams) -> dict[str, Any]:
    if hasattr(params, "model_dump"):
        return params.model_dump()
    return params.dict()


def _request_inpaint_params(payload: PortraitJobCreate) -> dict[str, Any]:
    values = _inpaint_params_dict(payload.inpaint)
    for name in ("strength", "steps", "width", "height", "seed"):
        value = getattr(payload, name, None)
        if value is not None:
            values[name] = value
    return values


def _normalize_mode(value: Any, payload: PortraitJobCreate | None = None) -> str:
    mode = _normalize(value).replace("-", "_").replace("+", "_").replace(" ", "_")
    if mode in {"dual_ip_adapter_v1", "sdxl_ip_adapter", "sdxl_ipadapter", "ip_adapter"}:
        return DUAL_IP_MODE
    if mode in {"fish_preserve_inpaint_v2", "fish_preserve_inpaint", "sdxl_inpaint", "inpaint"}:
        # Older callers did not send mode but did send a reference asset.  Do
        # not break those historical V1 experiments after the V2 default.
        if (
            mode == "fish_preserve_inpaint_v2"
            and payload is not None
            and payload.reference_asset_id
            and not payload.original_image_uri
            and not payload.fish_mask_uri
            and not payload.completion_mask_uri
        ):
            return DUAL_IP_MODE
        return INPAINT_MODE
    raise HTTPException(
        status_code=422,
        detail={
            "error_code": "PORTRAIT_MODE_UNSUPPORTED",
            "message": "mode 必须是 dual_ip_adapter_v1 或 fish_preserve_inpaint_v2",
        },
    )


def _stages_for_mode(mode: str) -> tuple[str, ...]:
    return INPAINT_STAGES if mode == INPAINT_MODE else STAGES


def _model_for_mode(mode: str) -> tuple[str, str]:
    return (INPAINT_MODEL_ID, INPAINT_MODEL_LABEL) if mode == INPAINT_MODE else (MODEL_ID, MODEL_LABEL)


def _experiment_metadata(
    params: dict[str, Any] | None,
    *,
    mode: str = DUAL_IP_MODE,
    species: str | None = None,
) -> dict[str, Any]:
    """Return the stable parameter shape stored with every PipelineRun.

    The request params remain in the state for replay/debugging. This separate
    shape is intentionally small and presentation-friendly so result pages and
    later A/B analysis do not need to infer which values belong to which
    adapter.
    """

    values = params if isinstance(params, dict) else {}
    if mode == INPAINT_MODE:
        result = {
            "mode": INPAINT_MODE,
            "strength": float(values.get("strength", 0.25)),
            "steps": int(values.get("steps", 25)),
            "width": int(values.get("width", 768)),
            "height": int(values.get("height", 768)),
            "seed": values.get("seed"),
            "mask_type": "completion_mask",
            "model": INPAINT_MODEL_LABEL,
        }
        if species:
            result["species"] = species
        return result
    return {
        "mode": DUAL_IP_MODE,
        "adapter_config": {
            "source_scale": float(values.get("source_scale", DEFAULT_PARAMS["source_scale"])),
            "reference_scale": float(values.get("reference_scale", DEFAULT_PARAMS["reference_scale"])),
        },
        "generation": {
            "steps": int(values.get("steps", DEFAULT_PARAMS["steps"])),
            "width": int(values.get("width", DEFAULT_PARAMS["width"])),
            "height": int(values.get("height", DEFAULT_PARAMS["height"])),
        },
    }


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


def _knowledge_cover_variant(row: FishKnowledgeAssetVersion) -> str:
    metadata = _json(getattr(row, "metadata_json", None), {})
    metadata = metadata if isinstance(metadata, dict) else {}
    value = str(
        metadata.get("cover_variant")
        or metadata.get("asset_role")
        or metadata.get("reference_variant")
        or ""
    ).strip().upper()
    aliases = {
        "COVER_CARD_TRANSPARENT_MAIN": "COVER_CARD_TRANSPARENT_LEFT",
        "TRANSPARENT_MAIN": "COVER_CARD_TRANSPARENT_LEFT",
        "TRANSPARENT_LEFT": "COVER_CARD_TRANSPARENT_LEFT",
        "COVER_CARD_TRANSPARENT_ALT": "COVER_CARD_TRANSPARENT_RIGHT",
        "TRANSPARENT_ALT": "COVER_CARD_TRANSPARENT_RIGHT",
        "TRANSPARENT_RIGHT": "COVER_CARD_TRANSPARENT_RIGHT",
        "COVER": "COVER_CARD",
    }
    value = aliases.get(value, value)
    if value in REFERENCE_VARIANT_ORDER:
        return value
    source_filename = str(metadata.get("source_filename") or "").strip().lower()
    stem = source_filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    if stem.startswith("01_transparent_main"):
        return "COVER_CARD_TRANSPARENT_LEFT"
    if stem.startswith("02_transparent_alt"):
        return "COVER_CARD_TRANSPARENT_RIGHT"
    return "COVER_CARD"


def _knowledge_reference_id(row: FishKnowledgeAssetVersion, variant: str) -> str:
    return f"{KNOWLEDGE_REFERENCE_PREFIX}{row.species_id}_{variant}_V{row.version}"


def _knowledge_reference_url(row: FishKnowledgeAssetVersion) -> str:
    value = str(row.image_url or "").strip()
    if value:
        return managed_knowledge_asset_url(row.species_id, "COVER", value)
    return f"/api/v1/fish/knowledge-media/{row.species_id}/cover/v{row.version}.webp"


def _knowledge_reference_uri(row: FishKnowledgeAssetVersion) -> str:
    try:
        bucket = get_bucket_name()
    except Exception:
        bucket = ""
    if bucket and row.object_name:
        return f"gs://{bucket}/{row.object_name}"
    return _knowledge_reference_url(row)


def _knowledge_reference_record(
    row: FishKnowledgeAssetVersion,
    *,
    species_id: str,
    variant: str | None = None,
) -> dict[str, Any]:
    context = _species_context(species_id)
    selected_variant = variant or _knowledge_cover_variant(row)
    return {
        "asset_id": _knowledge_reference_id(row, selected_variant),
        "type": selected_variant,
        "url": _knowledge_reference_url(row),
        "uri": _knowledge_reference_uri(row),
        "species_id": context["species_id"],
        "species_name": context["species_name"],
        "version": f"v{row.version}",
        "status": _status(row.status),
        "kind": "knowledge_cover",
        "cover_variant": selected_variant,
        "source": {
            "asset_type": "COVER",
            "object_name": row.object_name,
            "version_id": row.id,
        },
    }


def _knowledge_cover_reference_rows(db: Session, species_id: str) -> list[dict[str, Any]]:
    wanted = _species_context(species_id)
    rows = db.scalars(
        select(FishKnowledgeAssetVersion)
        .where(
            FishKnowledgeAssetVersion.asset_type == "COVER",
            FishKnowledgeAssetVersion.status == "ACTIVE",
        )
        .order_by(
            FishKnowledgeAssetVersion.created_at.desc(),
            FishKnowledgeAssetVersion.version.desc(),
            FishKnowledgeAssetVersion.id.desc(),
        )
    ).all()
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not _species_matches(row.species_id, wanted):
            continue
        variant = _knowledge_cover_variant(row)
        if variant not in selected:
            selected[variant] = _knowledge_reference_record(
                row,
                species_id=wanted["species_id"],
                variant=variant,
            )
    # A manually managed FishSpeciesCover remains a valid COVER_CARD fallback
    # when the imported version table does not contain the list-page cover.
    if "COVER_CARD" not in selected:
        cover = db.scalar(
            select(FishSpeciesCover).where(
                FishSpeciesCover.species_id == wanted["species_id"],
                FishSpeciesCover.status == "ACTIVE",
            )
        )
        if cover is not None and str(cover.image_url or "").strip():
            context = _species_context(wanted["species_id"])
            selected["COVER_CARD"] = {
                "asset_id": f"{KNOWLEDGE_REFERENCE_PREFIX}{wanted['species_id']}_COVER_CARD",
                "type": "COVER_CARD",
                "url": managed_knowledge_asset_url(wanted["species_id"], "COVER", cover.image_url),
                "uri": str(cover.image_url).strip(),
                "species_id": context["species_id"],
                "species_name": context["species_name"],
                "version": None,
                "status": _status(cover.status),
                "kind": "knowledge_cover",
                "cover_variant": "COVER_CARD",
                "source": {"asset_type": "COVER", "cover_id": cover.id},
            }
    return [
        selected[variant]
        for variant in REFERENCE_VARIANT_ORDER
        if variant in selected
    ]


def _fish_reference_record(row: FishAsset, species_id: str, index: int) -> dict[str, Any]:
    context = _species_context(row.species or species_id)
    return {
        "asset_id": row.asset_id,
        "type": "transparent_main" if index == 0 else "transparent_alt",
        "url": _asset_media_url(row.asset_id),
        "uri": row.transparent_uri,
        "species_id": context["species_id"],
        "species_name": context["species_name"],
        "version": row.version,
        "status": _status(row.status),
        "kind": "fish_asset",
        "source": {
            "batch_id": row.source_batch_id,
            "image_id": row.source_image_id,
        },
    }


def _reference_rows(db: Session, species_id: str, asset_type: str = "transparent") -> list[dict[str, Any]]:
    requested_type = str(asset_type or "transparent").strip().lower()
    if requested_type in {"cover", "cover_card"}:
        requested_type = "transparent"
    if requested_type not in {"transparent", "transparent_main", "transparent_alt"}:
        raise HTTPException(status_code=400, detail="仅支持 transparent 或 Cover 资产包参考图")
    wanted = _species_context(species_id)
    rows = db.scalars(
        select(FishAsset)
        .where(FishAsset.transparent_uri.is_not(None))
        .order_by(FishAsset.created_at.desc(), FishAsset.asset_id.desc())
    ).all()
    matched = [
        _fish_reference_record(row, wanted["species_id"], index)
        for index, row in enumerate(
            row
            for row in rows
            if _status(row.status) in REFERENCE_STATUSES
            and not str(row.asset_id or "").upper().startswith(PORTRAIT_PREFIX)
            and _species_matches(row.species, wanted)
        )
    ]
    # The Cover/Card package is the supported fallback when no FishAsset
    # transparent output exists for this species.
    if not matched:
        matched = _knowledge_cover_reference_rows(db, wanted["species_id"])
    if requested_type == "transparent_main":
        return matched[:1]
    if requested_type == "transparent_alt":
        return matched[1:]
    return matched


def _reference_dto(reference: dict[str, Any], species_id: str, index: int) -> dict[str, Any]:
    context = _species_context(species_id)
    return {
        "asset_id": reference.get("asset_id"),
        "type": reference.get("type") or ("transparent_main" if index == 0 else "transparent_alt"),
        "url": reference.get("url"),
        "species_id": reference.get("species_id") or context["species_id"],
        "species_name": reference.get("species_name") or context["species_name"],
        "version": reference.get("version"),
        "status": reference.get("status"),
        "source": reference.get("source"),
        "source_kind": reference.get("kind"),
        "cover_variant": reference.get("cover_variant"),
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
            else source.get("uri") if str(source.get("uri") or "").startswith(("data:", "http://", "https://")) else None
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
        "url": reference.get("url"),
        "version": reference.get("version"),
        "source_kind": reference.get("kind"),
        "cover_variant": reference.get("cover_variant"),
    }


def _initial_state(
    *,
    dataset_id: str,
    source: dict[str, Any],
    reference: dict[str, Any] | None,
    params: dict[str, Any],
    mode: str = DUAL_IP_MODE,
    input_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_id, _model_label = _model_for_mode(mode)
    request_species = str((input_state or {}).get("species") or "").strip()
    experiment = _experiment_metadata(
        params,
        mode=mode,
        species=request_species or str(source.get("species_name") or source.get("species_id") or "").strip() or None,
    )
    return {
        "request": {
            "dataset_id": dataset_id,
            "source_item_id": str(source["item_id"]) if source.get("item_id") is not None else None,
            "reference_asset_id": reference.get("asset_id") if reference else None,
            "mode": mode,
            "model": model_id,
            "params": params,
            **(input_state or {}),
        },
        "source": source,
        "reference": (
            {
                "asset_id": reference.get("asset_id"),
                "species_id": reference.get("species_id"),
                "species_name": reference.get("species_name"),
                "uri": reference.get("uri"),
                "version": reference.get("version"),
                "type": reference.get("type"),
                "kind": reference.get("kind"),
                "cover_variant": reference.get("cover_variant"),
                "url": reference.get("url"),
            }
            if reference
            else None
        ),
        "experiment": experiment,
        "stages": [{"name": stage, "status": "PENDING"} for stage in _stages_for_mode(mode)],
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
    mode: str = DUAL_IP_MODE,
    input_state: dict[str, Any] | None = None,
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
            and str(request.get("mode") or DUAL_IP_MODE) == mode
            and request.get("params") == params
            and all(request.get(key) == value for key, value in (input_state or {}).items())
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
    request = state.get("request", {}) if isinstance(state, dict) and isinstance(state.get("request"), dict) else {}
    return {
        "run_id": run.run_id,
        "id": run.run_id,
        "task_id": run.run_id,
        "type": run.pipeline_type,
        "status": _status(run.status),
        "stage": run.current_stage,
        "current_stage": run.current_stage,
        "mode": request.get("mode") or DUAL_IP_MODE,
        "steps": state.get("stages", []) if isinstance(state, dict) else [],
        "stages": state.get("stages", []) if isinstance(state, dict) else [],
        "source": _public_source(state.get("source", {})) if isinstance(state, dict) else None,
        "reference": _public_reference(state.get("reference")) if isinstance(state, dict) else None,
        "experiment": state.get("experiment") if isinstance(state, dict) else None,
        "result": {
            "asset_id": result.get("asset_id"),
            "generated_image": _asset_media_url(str(result["asset_id"]))
            if result and result.get("asset_id")
            else None,
            "metadata": result.get("metadata") if isinstance(result, dict) else None,
        }
        if isinstance(result, dict)
        else None,
        "worker": state.get("worker") if isinstance(state, dict) else None,
        "error_code": (state.get("error") or {}).get("code") if isinstance(state, dict) else None,
        "error_stage": run.error_stage,
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
    stages = tuple(item.get("name") for item in state.get("stages", []) if item.get("name")) or STAGES
    safe_message = str(message or error_code)[:3000]
    _stage(state, stage, "FAILED", error=f"{error_code}: {safe_message}")
    try:
        failed_index = stages.index(stage)
    except ValueError:
        failed_index = len(stages) - 1
    for skipped_stage in stages[failed_index + 1:]:
        _stage(state, skipped_stage, "SKIPPED", error=f"未执行：前置阶段 {stage} 失败")
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


def _read_managed_uri(uri: str) -> tuple[bytes, str]:
    value = str(uri or "").strip()
    if value.startswith("gs://"):
        bucket_name, object_name = value[5:].split("/", 1)
        blob = storage.Client().bucket(bucket_name).blob(object_name)
        data = blob.download_as_bytes(timeout=120)
        return data, mimetypes.guess_type(object_name)[0] or "application/octet-stream"
    if value.startswith("local://"):
        relative = value[len("local://") :].lstrip("/")
        root = Path.cwd().resolve()
        path = (root / relative).resolve()
        if path != root and root not in path.parents:
            raise ValueError("local URI escapes the application workspace")
        if not path.is_file():
            raise FileNotFoundError(value)
        return path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if value.startswith("/"):
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(value)
        return path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if value.startswith(("http://", "https://")):
        with urllib.request.urlopen(value, timeout=120) as response:
            return response.read(), response.headers.get_content_type() or "application/octet-stream"
    raise ValueError("unsupported managed URI")


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
        request = state.get("request") if isinstance(state.get("request"), dict) else {}
        mode = str(request.get("mode") or DUAL_IP_MODE)
        if mode not in {DUAL_IP_MODE, INPAINT_MODE}:
            raise PortraitWorkerError("PORTRAIT_MODE_UNSUPPORTED", "unsupported portrait mode")
        model_id, model_label = _model_for_mode(mode)
        run.status = "RUNNING"
        run.started_at = run.started_at or _utcnow()
        run.current_stage = active_stage
        _stage(state, active_stage, "RUNNING")
        _set_state(run, state)
        db.commit()

        source_value = (
            request.get("original_image_uri") or (state.get("source") or {}).get("uri")
            if mode == INPAINT_MODE
            else (state.get("source") or {}).get("uri")
        )
        source_uri = str(source_value or "").strip()
        reference_uri = str((state.get("reference") or {}).get("uri") or "").strip()
        if not source_uri:
            raise PortraitWorkerError("PORTRAIT_SOURCE_URI_INVALID", "source image URI is empty")
        _stage(state, active_stage, "DONE")
        _set_state(run, state)
        db.commit()

        active_stage = "load_masks" if mode == INPAINT_MODE else "load_reference"
        run.current_stage = active_stage
        _stage(state, active_stage, "RUNNING")
        _set_state(run, state)
        db.commit()
        if mode == INPAINT_MODE:
            fish_mask_uri = str(request.get("fish_mask_uri") or "").strip()
            completion_mask_uri = str(request.get("completion_mask_uri") or "").strip()
            if not fish_mask_uri:
                raise PortraitWorkerError("PORTRAIT_FISH_MASK_URI_INVALID", "fish_mask_uri is empty")
            if not completion_mask_uri:
                raise PortraitWorkerError("PORTRAIT_COMPLETION_MASK_URI_INVALID", "completion_mask_uri is empty")
        elif not reference_uri:
            raise PortraitWorkerError("PORTRAIT_REFERENCE_URI_INVALID", "reference asset URI is empty")
        _stage(state, active_stage, "DONE")
        _set_state(run, state)
        db.commit()

        active_stage = "sdxl_inpaint" if mode == INPAINT_MODE else "sdxl_generate"
        run.current_stage = active_stage
        _stage(state, active_stage, "RUNNING")
        _set_state(run, state)
        db.commit()
        health = check_portrait_worker()
        state["worker"] = {
            "health_status": health.get("status"),
            "endpoint_configured": bool(health.get("endpoint_configured")),
            "worker_url": health.get("worker_url"),
        }
        if health.get("status") != "READY":
            code = (
                "PORTRAIT_WORKER_NOT_CONFIGURED"
                if not health.get("endpoint_configured")
                else "PORTRAIT_WORKER_NOT_READY"
            )
            raise PortraitWorkerError(code, str(health.get("reason") or "Portrait worker is not ready"))
        worker_url = str(health.get("worker_url") or "").strip()
        logger.info(
            "fish_portrait_generate mode=%s run_id=%s source_image_id=%s reference_asset_id=%s worker_url=%s",
            mode,
            run_id,
            (state.get("source") or {}).get("image_id"),
            request.get("reference_asset_id"),
            worker_url or "<not-configured>",
        )
        if mode == INPAINT_MODE:
            inpaint = dict(request.get("inpaint") or {})
            worker_result = invoke_portrait_inpaint_worker(
                original_image_uri=source_uri,
                fish_mask_uri=fish_mask_uri,
                completion_mask_uri=completion_mask_uri,
                species=str(request.get("species") or (state.get("source") or {}).get("species_name") or ""),
                prompt=request.get("prompt"),
                negative_prompt=request.get("negative_prompt"),
                strength=float(inpaint.get("strength", 0.25)),
                steps=int(inpaint.get("steps", 25)),
                width=int(inpaint.get("width", 768)),
                height=int(inpaint.get("height", 768)),
                seed=inpaint.get("seed"),
            )
        else:
            logger.info(
                "fish_portrait_worker_request run_id=%s method=POST path=%s content_type=multipart/form-data source_field=image reference_field=reference_image params=%s",
                run_id,
                os.getenv("FISH_PORTRAIT_WORKER_PATH", "/portrait"),
                json.dumps(request.get("params") or DEFAULT_PARAMS, separators=(",", ":")),
            )
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
                "worker_http_status": worker_result.get("worker_http_status"),
                "worker_protocol": worker_result.get("worker_protocol"),
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
        asset.species = request.get("species") or (state.get("source") or {}).get("species_id") or (state.get("source") or {}).get("species_name")
        asset.status = "ACTIVE"
        asset.original_uri = source_uri
        asset.mask_uri = request.get("completion_mask_uri") if mode == INPAINT_MODE else None
        asset.transparent_uri = generated_uri
        asset.version = "fish-preserve-inpaint-v2" if mode == INPAINT_MODE else "portrait-poc-v1"
        state["result"] = {
            "asset_id": asset_id,
            "generated_uri": generated_uri,
            "mode": mode,
            "model": model_label,
            "params": request.get("inpaint") if mode == INPAINT_MODE else request.get("params") or DEFAULT_PARAMS,
            "metadata": state.get("experiment") or _experiment_metadata(
                request.get("inpaint") if mode == INPAINT_MODE else request.get("params"),
                mode=mode,
            ),
            "reference_asset_id": request.get("reference_asset_id"),
            "fish_mask_uri": request.get("fish_mask_uri") if mode == INPAINT_MODE else None,
            "completion_mask_uri": request.get("completion_mask_uri") if mode == INPAINT_MODE else None,
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
            detail={"asset_id": asset_id, "model": model_id, "mode": mode},
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
    }


@router.get("/portrait/worker-health")
def portrait_worker_health() -> dict[str, Any]:
    """Expose non-secret Worker connectivity for the POC page and smoke tests."""

    configured_url = str(os.getenv("FISH_PORTRAIT_WORKER_URL", "") or "").strip().rstrip("/")
    if not configured_url:
        return {
            "status": "NOT_CONFIGURED",
            "configured": False,
            "worker_url": None,
            "error_code": "PORTRAIT_WORKER_NOT_CONFIGURED",
            "message": "FISH_PORTRAIT_WORKER_URL is not configured",
        }
    try:
        health = check_portrait_worker()
    except PortraitWorkerError as exc:
        return {
            "status": "UNAVAILABLE",
            "configured": True,
            "worker_url": configured_url,
            "error_code": exc.error_code,
            "message": "Fish Portrait Worker unavailable",
            "detail": str(exc),
        }
    if health.get("status") != "READY":
        return {
            "status": "UNAVAILABLE",
            "configured": True,
            "worker_url": configured_url,
            "error_code": "PORTRAIT_WORKER_NOT_READY",
            "message": "Fish Portrait Worker unavailable",
        }
    return {
        "status": "CONNECTED",
        "configured": True,
        "worker_url": configured_url,
        "health": health.get("health") or {},
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
    mode = _normalize_mode(payload.mode, payload)
    dataset_id = str(payload.dataset_id or payload.dataset_version or "").strip()
    item = None
    source: dict[str, Any]
    if dataset_id:
        dataset = db.get(DatasetVersion, dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="数据集不存在")
        if _status(dataset.status) != "FROZEN":
            raise HTTPException(status_code=409, detail="只能发布已冻结 Dataset 的实验")
        if payload.source_item_id is not None:
            item = _resolve_source_item(db, dataset_id, payload.source_item_id)
            source = _source_state(db, item)
        elif mode == INPAINT_MODE and payload.original_image_uri:
            source = {
                "item_id": None,
                "image_id": None,
                "batch_id": None,
                "species_id": None,
                "species_name": payload.species,
                "split": None,
                "uri": payload.original_image_uri,
                "review_status": None,
                "source": "LOCAL_UPLOAD",
            }
        else:
            raise HTTPException(status_code=422, detail="source_item_id 不能为空")
    elif mode == INPAINT_MODE and payload.original_image_uri:
        source = {
            "item_id": None,
            "image_id": None,
            "batch_id": None,
            "species_id": None,
            "species_name": payload.species,
            "split": None,
            "uri": payload.original_image_uri,
            "review_status": None,
            "source": "LOCAL_UPLOAD",
        }
    else:
        raise HTTPException(status_code=422, detail="dataset_id 或 original_image_uri 不能为空")

    model_id, _model_label = _model_for_mode(mode)
    if mode == DUAL_IP_MODE:
        normalized_model = _normalize(payload.model).replace("+", "_").replace("-", "_").replace(" ", "_")
        if normalized_model not in {"sdxl_ip_adapter", "sdxl_ipadapter"}:
            raise HTTPException(
                status_code=422,
                detail={"error_code": "MODEL_NOT_SUPPORTED", "message": "当前仅支持 SDXL + IP-Adapter"},
            )
    else:
        if not payload.original_image_uri and not source.get("uri"):
            raise HTTPException(status_code=422, detail="original_image_uri 不能为空")
        if not payload.fish_mask_uri:
            raise HTTPException(status_code=422, detail="fish_mask_uri 不能为空")
        if not payload.completion_mask_uri:
            raise HTTPException(status_code=422, detail="completion_mask_uri 不能为空")

    wanted_species = _species_context(
        payload.species or source.get("species_id") or source.get("species_name")
    )
    reference: dict[str, Any] | None = None
    if mode == DUAL_IP_MODE and payload.reference_asset_id:
        fish_asset = db.get(FishAsset, payload.reference_asset_id)
        if fish_asset is not None:
            reference = _fish_reference_record(fish_asset, wanted_species["species_id"], 0)
        else:
            reference = next(
                (
                    row
                    for row in _reference_rows(db, wanted_species["species_id"], "transparent")
                    if row.get("asset_id") == payload.reference_asset_id
                ),
                None,
            )
        if reference is None:
            raise HTTPException(status_code=404, detail="标准鱼体参考资产不存在")
        if not reference.get("uri") or _status(reference.get("status")) not in REFERENCE_STATUSES:
            raise HTTPException(status_code=409, detail="参考资产没有可用的 transparent 或 Cover 图")
        if str(reference.get("asset_id") or "").upper().startswith(PORTRAIT_PREFIX):
            raise HTTPException(status_code=409, detail="生成结果不能作为标准参考资产")
        if not _species_matches(reference.get("species_id"), wanted_species):
            raise HTTPException(status_code=409, detail="参考鱼体与 A 图鱼种不匹配")
    elif mode == DUAL_IP_MODE:
        rows = _reference_rows(db, wanted_species["species_id"], "transparent")
        if not rows:
            raise HTTPException(
                status_code=409,
                detail={
                    "error_code": "REFERENCE_ASSET_NOT_FOUND",
                    "message": "没有找到该鱼种的 transparent 或 Cover 资产包参考图",
                },
            )
        reference = rows[0]
    if mode == INPAINT_MODE:
        params = _request_inpaint_params(payload)
        input_state = {
            "original_image_uri": str(payload.original_image_uri or source.get("uri") or "").strip(),
            "fish_mask_uri": str(payload.fish_mask_uri or "").strip(),
            "completion_mask_uri": str(payload.completion_mask_uri or "").strip(),
            "species": str(payload.species or source.get("species_name") or "").strip() or None,
            "prompt": str(payload.prompt or INPAINT_DEFAULT_PROMPT).strip(),
            "negative_prompt": str(payload.negative_prompt or INPAINT_DEFAULT_NEGATIVE_PROMPT).strip(),
            "inpaint": params,
        }
    else:
        params = _params_dict(payload.params)
        input_state = {}
    existing = _find_active_job(
        db,
        dataset_id=dataset_id,
        source_item_id=str(item.id) if item is not None else str(input_state.get("original_image_uri") or ""),
        reference_asset_id=str(reference.get("asset_id") or "") if reference else "",
        params=params,
        mode=mode,
        input_state=input_state,
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
        mode=mode,
        input_state=input_state,
    )
    run = PipelineRun(
        run_id=run_id,
        source_batch_id=item.batch_id if item is not None else None,
        source_image_id=item.image_id if item is not None else None,
        pipeline_type=PIPELINE_TYPE,
        status="PENDING",
        current_stage="queued",
        stage_json=json.dumps(state, ensure_ascii=False),
        model_version=model_id,
    )
    db.add(run)
    adapters.record_operation(
        db,
        "CREATE_PORTRAIT_RUN",
        "PIPELINE_RUN",
        run_id,
        detail={
            "dataset_id": dataset_id,
            "source_item_id": item.id if item is not None else None,
            "reference_asset_id": reference.get("asset_id") if reference else None,
            "model": model_id,
            "mode": mode,
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
    request = state.get("request") if isinstance(state.get("request"), dict) else {}
    mode = str(request.get("mode") or DUAL_IP_MODE)
    experiment = state.get("experiment")
    if not isinstance(experiment, dict):
        experiment = _experiment_metadata(
            request.get("inpaint") if mode == INPAINT_MODE else request.get("params"),
            mode=mode,
        )
    model_id, model_label = _model_for_mode(mode)
    result_species = str(request.get("species") or source.get("species_name") or source.get("species_id") or "").strip() or None
    metadata = {
        "mode": mode,
        "model": model_label,
        "model_id": model_id,
        "params": request.get("inpaint") if mode == INPAINT_MODE else request.get("params") or DEFAULT_PARAMS,
        "adapter_config": experiment.get("adapter_config", {}),
        "generation": experiment.get("generation", {}),
        "strength": experiment.get("strength"),
        "steps": experiment.get("steps"),
        "seed": experiment.get("seed"),
        "mask_type": experiment.get("mask_type"),
        "prompt": request.get("prompt") if mode == INPAINT_MODE else None,
        "negative_prompt": request.get("negative_prompt") if mode == INPAINT_MODE else None,
        "run_id": run.run_id,
        "species_id": source.get("species_id"),
        "species_name": result_species,
        "reference_asset_id": reference.get("asset_id"),
        "worker": state.get("worker"),
    }
    return {
        "run_id": run.run_id,
        "status": _status(run.status),
        "source_image": (
            f"/api/platform/portrait/results/{run.run_id}/media/original"
            if mode == INPAINT_MODE
            else _public_source(source).get("image_url")
        ),
        "reference_image": _public_reference(reference).get("url") if reference else None,
        "generated_image": _asset_media_url(str(result["asset_id"])) if result.get("asset_id") else None,
        "fish_mask_image": f"/api/platform/portrait/results/{run.run_id}/media/fish-mask" if mode == INPAINT_MODE else None,
        "completion_mask_image": f"/api/platform/portrait/results/{run.run_id}/media/completion-mask" if mode == INPAINT_MODE else None,
        "metadata": metadata,
    }


@router.get("/portrait/results/{run_id}/media/{kind}")
def portrait_result_media(run_id: str, kind: str, db: Session = Depends(get_db)) -> Response:
    """Serve the private V2 source/mask artifacts through the console API."""

    if kind not in {"original", "fish-mask", "completion-mask", "generated"}:
        raise HTTPException(status_code=404, detail="资源不存在")
    run = db.get(PipelineRun, run_id)
    if run is None or run.pipeline_type != PIPELINE_TYPE:
        raise HTTPException(status_code=404, detail="Fish Portrait 结果不存在")
    state = _json(run.stage_json, {})
    state = state if isinstance(state, dict) else {}
    request = state.get("request") if isinstance(state.get("request"), dict) else {}
    result = state.get("result") if isinstance(state.get("result"), dict) else {}
    uri = {
        "original": request.get("original_image_uri") or (state.get("source") or {}).get("uri"),
        "fish-mask": request.get("fish_mask_uri"),
        "completion-mask": request.get("completion_mask_uri"),
        "generated": result.get("generated_uri"),
    }.get(kind)
    if not uri:
        raise HTTPException(status_code=404, detail="资源不存在")
    try:
        content, media_type = _read_managed_uri(str(uri))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="资源不存在") from exc
    except Exception as exc:
        logger.exception("Fish Portrait media read failed run_id=%s kind=%s", run_id, kind)
        raise HTTPException(status_code=503, detail="资源暂时不可用") from exc
    return Response(
        content=content,
        media_type=media_type,
        headers={"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"},
    )


__all__ = [
    "PortraitJobCreate",
    "PortraitParams",
    "PortraitInpaintParams",
    "INPAINT_MODE",
    "PIPELINE_TYPE",
    "_execute_portrait_job",
    "router",
]

