from __future__ import annotations

import mimetypes
from collections import defaultdict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.bulk_review import BulkReviewApply, BulkReviewItem, api_bulk_apply
from app.crop_review import CropReviewUpdate, update_crop_review
from app.db import get_db
from app.models import DatasetVersion, ImageAsset
from app.platform.services import adapters
from app.training_api import TrainingCreate, queue_training_run
from app.frozen_crop_bridge import _read_uri


router = APIRouter(prefix="/api/platform", tags=["platform-api"])


class ReviewSelection(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=100)
    batch_id: str | None = Field(default=None, max_length=128)


class ReviewSpeciesSelection(ReviewSelection):
    species: str = Field(min_length=1, max_length=128)


class ReviewBBoxSelection(ReviewSelection):
    accepted_bbox: list[float] = Field(min_length=4, max_length=4)
    species: str | None = Field(default=None, max_length=128)


class PlatformTrainingCreate(BaseModel):
    dataset_id: str | None = Field(default=None, max_length=128)
    dataset_version: str | None = Field(default=None, max_length=128)
    model_type: str = Field(default="classifier", max_length=64)
    model_family: str = Field(default="mobilenet_v3_small", max_length=128)
    epoch: int = Field(default=12, ge=1, le=100)
    epochs: int | None = Field(default=None, ge=1, le=100)
    batch_size: int = Field(default=16, ge=1, le=128)
    image_size: int = Field(default=224, ge=128, le=512)
    learning_rate: float = Field(default=0.001, gt=0, le=0.1)
    freeze: bool | None = None
    seed: int = 20260827


def _image_ref(value: str, batch_id: str | None) -> tuple[str | None, str]:
    value = str(value or "").strip()
    if batch_id and value.startswith(f"{batch_id}:"):
        return batch_id, value[len(batch_id) + 1 :]
    if not batch_id and ":" in value:
        possible_batch, image_id = value.split(":", 1)
        if possible_batch and image_id:
            return possible_batch, image_id
    return batch_id, value


def _resolve_images(db: Session, payload: ReviewSelection) -> list[ImageAsset]:
    resolved: list[ImageAsset] = []
    seen: set[tuple[str, str]] = set()
    for value in payload.ids:
        batch_id, image_id = _image_ref(value, payload.batch_id)
        if not batch_id:
            raise HTTPException(status_code=400, detail=f"无法确定图片所属批次：{value}")
        image = db.scalar(select(ImageAsset).where(ImageAsset.batch_id == batch_id, ImageAsset.image_id == image_id))
        if not image:
            raise HTTPException(status_code=404, detail=f"图片不存在：{value}")
        key = (image.batch_id, image.image_id)
        if key not in seen:
            seen.add(key)
            resolved.append(image)
    return resolved


def _group_images(images: list[ImageAsset]) -> dict[str, list[ImageAsset]]:
    grouped: dict[str, list[ImageAsset]] = defaultdict(list)
    for image in images:
        grouped[image.batch_id].append(image)
    return grouped


def _apply_review_items(db: Session, images: list[ImageAsset], *, status: str, species: str | None = None, bbox: list[float] | None = None) -> int:
    """Call the existing bulk-review state machine, grouped by batch."""

    changed = 0
    for batch_id, grouped in _group_images(images).items():
        items = []
        for image in grouped:
            values: dict[str, Any] = {
                "image_id": image.image_id,
                "review_status": status,
                "truth_species": species if species is not None else image.truth_species,
                "notes": image.notes,
            }
            if bbox is not None:
                values["accepted_bbox"] = bbox
            items.append(BulkReviewItem(**values))
        result = api_bulk_apply(BulkReviewApply(batch_id=batch_id, items=items), db)
        changed += int(result.get("updated", 0))
    return changed


@router.get("/dashboard")
def platform_dashboard(db: Session = Depends(get_db)) -> dict[str, Any]:
    return adapters.dashboard(db)


@router.get("/datasets")
def platform_datasets(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    return adapters.datasets(db)


@router.get("/datasets/{dataset_id}")
def platform_dataset_detail(dataset_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    result = adapters.dataset_detail(db, dataset_id)
    if result is None:
        raise HTTPException(status_code=404, detail="数据集不存在")
    return result


@router.get("/datasets/{dataset_id}/clean-report")
def platform_clean_report(dataset_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    result = adapters.clean_report(db, dataset_id)
    if result is None:
        raise HTTPException(status_code=404, detail="数据集不存在")
    return result


@router.get("/datasets/{dataset_id}/manifest")
def platform_dataset_manifest(dataset_id: str, db: Session = Depends(get_db)) -> Response:
    """Download the registered manifest through the controlled Platform API."""

    dataset = db.get(DatasetVersion, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")
    try:
        content, _ = _read_uri(dataset.manifest_uri)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Manifest 不存在") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Manifest 暂时不可用") from exc
    media_type = mimetypes.guess_type(str(dataset.manifest_uri or ""))[0] or "application/octet-stream"
    extension = ".json" if media_type == "application/json" else ".csv" if media_type == "text/csv" else ".manifest"
    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="dataset-manifest{extension}"',
            "Cache-Control": "private, max-age=300",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/review/items")
def platform_review_items(
    status: str | None = Query(default=None, max_length=32),
    batch_id: str | None = Query(default=None, max_length=128),
    species: str | None = Query(default=None, max_length=128),
    issue: str | None = Query(default=None, max_length=32),
    q: str | None = Query(default=None, max_length=256),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=30, ge=1, le=100),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    return adapters.review_items(
        db,
        status=status,
        batch_id=batch_id,
        species=species,
        issue=issue,
        q=q,
        page=page,
        page_size=page_size,
    )


@router.get("/review/queue")
def platform_review_queue(db: Session = Depends(get_db)) -> dict[str, Any]:
    return adapters.review_queue(db)


@router.post("/review/batch-confirm")
def platform_batch_confirm(payload: ReviewSelection, db: Session = Depends(get_db)) -> dict[str, Any]:
    images = _resolve_images(db, payload)
    try:
        changed = _apply_review_items(db, images, status="approved")
        adapters.record_operation(db, "BATCH_CONFIRM", "review", str(len(images)), detail={"updated": changed})
        db.commit()
        return {"updated": changed, "status": "approved"}
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail="批量确认失败，请检查真实鱼种和 accepted_bbox") from exc


@router.post("/review/species")
def platform_species_update(payload: ReviewSpeciesSelection, db: Session = Depends(get_db)) -> dict[str, Any]:
    images = _resolve_images(db, payload)
    try:
        # Preserve already-approved items through the same existing gate.
        pending = [image for image in images if image.review_status != "approved"]
        approved = [image for image in images if image.review_status == "approved"]
        changed = _apply_review_items(db, pending, status="pending", species=payload.species) if pending else 0
        if approved:
            changed += _apply_review_items(db, approved, status="approved", species=payload.species)
        adapters.record_operation(db, "SPECIES_UPDATE", "review", str(len(images)), detail={"species": payload.species, "updated": changed})
        db.commit()
        return {"updated": changed, "species": payload.species}
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail="鱼种修改失败") from exc


@router.post("/review/bbox")
def platform_bbox_update(payload: ReviewBBoxSelection, db: Session = Depends(get_db)) -> dict[str, Any]:
    images = _resolve_images(db, payload)
    # BBox writes use the existing crop-review bridge so candidate_bbox remains
    # diagnostic and only an explicit accepted_bbox becomes training truth.
    try:
        for image in images:
            species = payload.species or image.truth_species or image.claimed_species
            update_crop_review(
                image.batch_id,
                image.image_id,
                CropReviewUpdate(decision="ACCEPTED", accepted_bbox=payload.accepted_bbox, species_name=species),
                db,
            )
        adapters.record_operation(db, "BBOX_UPDATE", "review", str(len(images)), detail={"updated": len(images)})
        db.commit()
        return {"updated": len(images), "bbox": payload.accepted_bbox}
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail="BBox 修改失败，请先确认鱼种和坐标") from exc


@router.post("/review/batch-delete")
def platform_batch_delete(payload: ReviewSelection, db: Session = Depends(get_db)) -> dict[str, Any]:
    images = _resolve_images(db, payload)
    try:
        changed = _apply_review_items(db, images, status="rejected")
        adapters.record_operation(db, "BATCH_REJECT", "review", str(len(images)), detail={"updated": changed, "soft_delete": True})
        db.commit()
        return {"updated": changed, "soft_delete": True, "status": "rejected"}
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail="批量删除失败") from exc


@router.get("/training/jobs")
def platform_training_jobs(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    return adapters.training_jobs(db)


@router.post("/training/create")
def platform_training_create(payload: PlatformTrainingCreate, db: Session = Depends(get_db)) -> dict[str, Any]:
    dataset_version = (payload.dataset_version or payload.dataset_id or "").strip()
    if not dataset_version:
        raise HTTPException(status_code=400, detail="请选择数据集")
    model_type = payload.model_type.strip().lower()
    pipeline_type = "CROP_CLASSIFIER_V1" if model_type in {"classifier", "分类模型", "分类"} else "WHOLE_IMAGE_V1"
    try:
        result = queue_training_run(
            db,
            TrainingCreate(
                dataset_version=dataset_version,
                model_family=payload.model_family,
                epochs=payload.epochs or payload.epoch,
                batch_size=payload.batch_size,
                image_size=payload.image_size,
                learning_rate=payload.learning_rate,
                seed=payload.seed,
                pipeline_type=pipeline_type,
            ),
        )
        adapters.record_operation(db, "TRAINING_CREATE", "training", result.get("run_id"), detail={"dataset": dataset_version, "model_type": model_type})
        db.commit()
        return {"task_id": result.get("run_id"), "model_version": result.get("model_version"), "status": result.get("status")}
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/training/{run_id}/report")
def platform_training_report(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    result = adapters.training_report(db, run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    return result


@router.get("/models")
def platform_models(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    return adapters.models(db)


@router.get("/models/{model_id}/evaluation")
def platform_model_evaluation(model_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    result = adapters.evaluation(db, model_id)
    if result is None:
        raise HTTPException(status_code=404, detail="模型不存在")
    return result


@router.get("/pipelines")
def platform_pipelines(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    return adapters.pipelines(db)


@router.get("/pipelines/{pipeline_id}")
def platform_pipeline_detail(pipeline_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    result = adapters.pipeline_detail(db, pipeline_id)
    if result is None:
        raise HTTPException(status_code=404, detail="流水线任务不存在")
    return result


@router.get("/assets")
def platform_assets(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    return adapters.assets(db)


@router.get("/assets/{asset_id}")
def platform_asset_detail(asset_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    result = adapters.asset_detail(db, asset_id)
    if result is None:
        raise HTTPException(status_code=404, detail="鱼体资产不存在")
    return result


@router.get("/assets/{asset_id}/media/{kind}")
def platform_asset_media(asset_id: str, kind: str, db: Session = Depends(get_db)) -> Response:
    if kind not in {"original", "mask", "transparent", "sticker"}:
        raise HTTPException(status_code=404, detail="资源不存在")
    uri = adapters.asset_uri(db, asset_id, kind)
    if uri is None:
        raise HTTPException(status_code=404, detail="资源不存在")
    try:
        content, _ = _read_uri(uri)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="资源不存在") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="资源暂时不可用") from exc
    media_type = mimetypes.guess_type(str(uri))[0] or "application/octet-stream"
    return Response(content=content, media_type=media_type, headers={"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"})


@router.get("/knowledge")
def platform_knowledge(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    return adapters.knowledge(db)


@router.get("/knowledge/{species_id}")
def platform_knowledge_detail(species_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    result = adapters.knowledge_detail(db, species_id)
    if result is None:
        raise HTTPException(status_code=404, detail="鱼种知识不存在")
    return result


@router.get("/habitat")
def platform_habitat() -> list[dict[str, Any]]:
    return adapters.habitat()


@router.get("/system/tasks")
def platform_tasks(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    return adapters.tasks(db)


@router.get("/system/logs")
def platform_logs(limit: int = Query(default=100, ge=1, le=200), db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    return adapters.logs(db, limit)


__all__ = ["router"]
