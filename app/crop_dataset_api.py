from __future__ import annotations

import csv
import inspect
import os
import re
from pathlib import Path
from typing import Any, Mapping

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db import get_db
from app.factory import get_bucket_name
from trainer.crop_dataset_pipeline import (
    CROP_DATASET_VERSION,
    build_reviewed_crop_dataset_from_db,
    freeze_crop_dataset,
)
from trainer.crop_dataset_validator import CropDatasetValidationError, validate_crop_dataset


router = APIRouter(prefix="/api/crop-datasets", tags=["crop-datasets"])


class CropDatasetBuildRequest(BaseModel):
    batch_id: str | None = Field(default=None, max_length=128)
    source_type: str = Field(default="RAW_BATCH", max_length=32)
    source_dataset: str | None = Field(default=None, max_length=128)
    dataset_version: str = Field(default=CROP_DATASET_VERSION, min_length=4, max_length=128)
    pipeline: str = Field(default="CROP_CLASSIFIER_V1", max_length=64)
    pipeline_type: str | None = Field(default=None, max_length=64)
    expand_ratio: float = Field(default=0.15, ge=0.0, le=1.0)
    limit: int = Field(default=5000, ge=1, le=100000)
    freeze: bool = False
    git_commit: str | None = Field(default=None, max_length=128)


class CropDatasetFreezeRequest(BaseModel):
    confirm: bool = False
    git_commit: str | None = Field(default=None, max_length=128)


def _safe_version(value: str) -> str:
    value = value.strip()
    if not re.fullmatch(r"DS_[A-Za-z0-9_.-]{1,120}", value):
        raise HTTPException(status_code=400, detail="dataset_version 格式不合法")
    return value


def _safe_batch(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    batch_id = value.strip()
    if not re.fullmatch(r"BATCH_[A-Za-z0-9_.-]{1,120}", batch_id):
        raise HTTPException(status_code=400, detail={"error": "INVALID_BATCH_ID", "reason": "batch_id 格式不合法"})
    return batch_id


def _safe_source_type(value: str) -> str:
    source_type = (value or "RAW_BATCH").strip().upper()
    if source_type not in {"RAW_BATCH", "FROZEN_DATASET"}:
        raise HTTPException(status_code=400, detail={"error": "SOURCE_TYPE_NOT_SUPPORTED", "source_type": source_type})
    return source_type


def _safe_pipeline(value: str) -> str:
    pipeline = (value or "").strip().upper()
    if pipeline != "CROP_CLASSIFIER_V1":
        raise HTTPException(
            status_code=400,
            detail={"error": "PIPELINE_NOT_SUPPORTED", "pipeline": pipeline, "reason": "Crop Dataset 只支持 CROP_CLASSIFIER_V1"},
        )
    return pipeline


def _invoke_builder(db: Session, root: Path, payload: CropDatasetBuildRequest, version: str):
    """Call old test doubles and the new batch-scoped builder safely."""

    kwargs = {
        "dataset_version": version,
        "pipeline": payload.pipeline,
        "expand_ratio": payload.expand_ratio,
        "limit": payload.limit,
        "batch_id": payload.batch_id,
        "source_dataset": payload.source_dataset,
    }
    builder = build_reviewed_crop_dataset_from_db
    try:
        parameters = inspect.signature(builder).parameters
        kwargs = {key: value for key, value in kwargs.items() if key in parameters or any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())}
    except (TypeError, ValueError):
        kwargs.pop("batch_id", None)
    return builder(db, root, **kwargs)


def _staging_root(dataset_version: str) -> Path:
    version = _safe_version(dataset_version)
    base = Path(os.getenv("CROP_DATASET_STAGING_ROOT", "var/crop_datasets"))
    # Accept both the configured parent (the documented form) and an already
    # versioned root supplied by older deployments.  Appending the version to
    # the latter was another source of hard-to-diagnose duplicate paths.
    if base.name == version:
        return base
    return base / version


def _staging_manifest(root: Path) -> Path:
    """Return the sole canonical staging manifest location."""

    return root / "metadata" / "crop_manifest.csv"


def _rollback(db: Session | None) -> None:
    if db is None:
        return
    try:
        db.rollback()
    except Exception:
        # Preserve the original build error if a lightweight test double (or
        # a disconnected session) cannot roll back.
        return


def _registered_state(db: Session | None, dataset_version: str) -> str | None:
    """Read a persisted Freeze state without making staging assumptions."""

    if db is None:
        return None
    try:
        from app.models import DatasetVersion

        dataset = db.get(DatasetVersion, dataset_version)
    except Exception:
        return None
    if dataset is None:
        return None
    status = str(getattr(dataset, "status", "") or "").strip().upper()
    if status in {"READY_FOR_TRAINING", "FROZEN"}:
        return "READY_FOR_TRAINING"
    if status == "VALIDATED":
        return "VALIDATED"
    return None


def _freeze_marker_exists(root: Path) -> bool:
    # Local Freeze writes both markers.  In a deployed build the immutable
    # marker may only be in GCS, in which case _registered_state is the guard.
    return (root / "dataset.json").is_file() or (root / "metadata" / "dataset.json").is_file()


def _staging_state(
    root: Path,
    dataset_version: str,
    db: Session | None,
    validation: Mapping[str, Any] | None = None,
) -> str:
    """Classify a staging tree for the safe-resume decision.

    ``VALIDATED`` and ``READY_FOR_TRAINING`` are immutable operator gates.
    An existing manifest with a failed validation remains rebuildable and is
    reported as ``STAGING``.
    """

    registered = _registered_state(db, dataset_version)
    if registered:
        return registered
    if _freeze_marker_exists(root):
        return "READY_FOR_TRAINING"
    manifest = _staging_manifest(root)
    if not manifest.is_file():
        return "NEW"
    if validation is None:
        validation = validate_crop_dataset(root, require_metadata=True, check_source_image=True)
    if validation.get("valid"):
        return "VALIDATED"
    return "STAGING"


def _manifest_classes(manifest_path: Path) -> list[str]:
    if not manifest_path.is_file():
        return []
    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = csv.DictReader(handle)
            values = {
                str(row.get("species_key") or row.get("species_name") or "").strip()
                for row in rows
            }
    except (OSError, csv.Error):
        return []
    return sorted(value for value in values if value)


def _manifest_source_batches(manifest_path: Path) -> set[str]:
    if not manifest_path.is_file():
        return set()
    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            return {
                str(row.get("source_batch") or "").strip()
                for row in csv.DictReader(handle)
                if str(row.get("source_batch") or "").strip()
            }
    except (OSError, csv.Error):
        return set()


def _build_validation(root: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the canonical manifest after a builder attempt.

    The builder report also carries build failures that may not be visible in
    the materialized CSV (for example an accepted source that could not be
    downloaded), so retain an explicitly failed builder report while using a
    fresh validator pass for all successful builds.
    """

    validation = validate_crop_dataset(root, require_metadata=True, check_source_image=True)
    reported = report.get("validation")
    if isinstance(reported, Mapping) and reported.get("valid") is False:
        return dict(reported)
    return validation


@router.post("/build")
def build_crop_dataset_endpoint(payload: CropDatasetBuildRequest, db: Session = Depends(get_db)) -> dict:
    """Build reviewed crops; optionally publish/register them when explicitly frozen."""

    version = _safe_version(payload.dataset_version)
    source_type = _safe_source_type(payload.source_type)
    requested_pipeline = payload.pipeline_type or payload.pipeline
    _safe_pipeline(requested_pipeline)
    if payload.pipeline_type and payload.pipeline.strip().upper() != payload.pipeline_type.strip().upper():
        raise HTTPException(status_code=400, detail={"error": "PIPELINE_MISMATCH", "reason": "pipeline 与 pipeline_type 不一致"})
    payload.batch_id = _safe_batch(payload.batch_id)
    if source_type == "FROZEN_DATASET":
        if not payload.source_dataset:
            raise HTTPException(status_code=400, detail={"error": "SOURCE_DATASET_REQUIRED"})
        payload.source_dataset = _safe_version(payload.source_dataset)
        if payload.batch_id:
            raise HTTPException(status_code=400, detail={"error": "SOURCE_SELECTOR_CONFLICT", "reason": "Frozen Dataset 不接受 batch_id"})
    elif payload.source_dataset:
        raise HTTPException(status_code=400, detail={"error": "SOURCE_DATASET_ONLY_FOR_FROZEN"})
    root = _staging_root(version)
    manifest_path = _staging_manifest(root)
    existing_validation = (
        validate_crop_dataset(root, require_metadata=True, check_source_image=True)
        if manifest_path.is_file()
        else None
    )
    existing_state = _staging_state(root, version, db, existing_validation)
    if existing_state in {"VALIDATED", "READY_FOR_TRAINING"}:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "CROP_DATASET_NOT_REBUILDABLE",
                "state": existing_state,
                "dataset_version": version,
                "manifest_path": str(manifest_path),
                "message": "validated or frozen Crop Dataset cannot be overwritten",
            },
        )
    if source_type == "FROZEN_DATASET":
        from app.frozen_crop_bridge import crop_readiness

        readiness = crop_readiness(db, payload.source_dataset)
        if not readiness.get("ground_truth_confirmed"):
            raise HTTPException(status_code=400, detail={"error": "FROZEN_DATASET_NOT_READY", "readiness": readiness})
        if not readiness.get("crop_ready"):
            raise HTTPException(
                status_code=400,
                detail={"error": "BBOX_REVIEW_INCOMPLETE", "source_dataset": payload.source_dataset, "readiness": readiness},
            )
    resuming_failed_staging = manifest_path.is_file()
    try:
        report = _invoke_builder(db, root, payload, version)
        if payload.batch_id and int(report.get("accepted", 0) or 0) == 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "NO_ACCEPTED_BBOX_ASSETS",
                    "batch_id": payload.batch_id,
                    "reason": "指定 Batch 没有 ACCEPTED/TRAINING_READY 的 accepted_bbox 资产；candidate_bbox 不可直接训练",
                },
            )
        if payload.batch_id:
            source_batches = set(report.get("source_batches") or [])
            manifest_batches = _manifest_source_batches(manifest_path)
            if source_batches and source_batches != {payload.batch_id}:
                raise HTTPException(
                    status_code=400,
                    detail={"error": "BATCH_SCOPE_VIOLATION", "batch_id": payload.batch_id, "source_batches": sorted(source_batches)},
                )
            if manifest_batches != {payload.batch_id}:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "BATCH_SCOPE_VIOLATION",
                        "batch_id": payload.batch_id,
                        "source_batches": sorted(manifest_batches),
                        "reason": "manifest 每一行必须保留指定 source_batch",
                    },
                )
        validation = _build_validation(root, report)
        if not validation.get("valid"):
            raise CropDatasetValidationError(validation)
        classes = _manifest_classes(manifest_path)
        result: dict = {
            "dataset_version": version,
            "batch_id": payload.batch_id,
            "source_type": source_type,
            "source_dataset": payload.source_dataset,
            "pipeline": "CROP_CLASSIFIER_V1",
            "state": "VALIDATED",
            "status": "BUILT_NOT_FROZEN",
            "staging_root": str(root),
            "manifest_path": str(manifest_path),
            "rows": int(validation.get("rows", report.get("written", 0)) or 0),
            "valid_rows": int(validation.get("valid_rows", report.get("written", 0)) or 0),
            "classes": classes,
            "image_count": report.get("written", 0),
            "class_count": len(classes) if classes else report.get("class_count", 0),
            "validation": validation,
            "safety": report.get("safety"),
            "freeze_required": True,
            "resumed_failed_staging": resuming_failed_staging,
        }
        if payload.freeze:
            result["freeze"] = freeze_crop_dataset(
                root,
                dataset_version=version,
                bucket_name=get_bucket_name(),
                db=db,
                git_commit=(payload.git_commit or os.getenv("APP_GIT_COMMIT", "unknown")).strip() or "unknown",
                source_batches=report.get("source_batches") or ([payload.batch_id] if payload.batch_id else []),
            )
            result["state"] = "READY_FOR_TRAINING"
            result["status"] = "READY_FOR_TRAINING"
        return result
    except CropDatasetValidationError as exc:
        _rollback(db)
        raise HTTPException(status_code=400, detail={"error": "CROP_DATASET_INVALID", "report": exc.report}) from exc
    except HTTPException:
        _rollback(db)
        raise
    except Exception as exc:
        _rollback(db)
        raise HTTPException(status_code=400, detail={"error": "CROP_DATASET_BUILD_FAILED", "reason": str(exc)}) from exc


@router.get("/sources")
def crop_dataset_sources(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Return Batch readiness without mutating review or dataset state."""

    from sqlalchemy import func, select

    from app.models import Batch, BatchCropReview, ImageAsset

    batches = db.scalars(select(Batch).order_by(Batch.created_at.desc())).all()
    result: list[dict[str, Any]] = []
    for batch in batches:
        total = int(db.scalar(select(func.count()).select_from(ImageAsset).where(ImageAsset.batch_id == batch.batch_id)) or 0)
        review_rows = db.execute(
            select(BatchCropReview.status, func.count())
            .where(BatchCropReview.batch_id == batch.batch_id)
            .group_by(BatchCropReview.status)
        ).all()
        counts = {str(status): int(count) for status, count in review_rows}
        accepted = counts.get("ACCEPTED", 0) + counts.get("TRAINING_READY", 0)
        valid_accepted = int(
            db.scalar(
                select(func.count())
                .select_from(BatchCropReview)
                .where(
                    BatchCropReview.batch_id == batch.batch_id,
                    BatchCropReview.status.in_({"ACCEPTED", "TRAINING_READY"}),
                    BatchCropReview.accepted_bbox_json.is_not(None),
                    (BatchCropReview.species_key.is_not(None) | BatchCropReview.species_name.is_not(None)),
                )
            )
            or 0
        )
        missing_species = max(accepted - int(
            db.scalar(
                select(func.count())
                .select_from(BatchCropReview)
                .where(
                    BatchCropReview.batch_id == batch.batch_id,
                    BatchCropReview.status.in_({"ACCEPTED", "TRAINING_READY"}),
                    BatchCropReview.species_key.is_not(None) | BatchCropReview.species_name.is_not(None),
                )
            )
            or 0
        ), 0)
        if valid_accepted:
            readiness = "READY_TO_BUILD"
        elif total:
            readiness = "REVIEW_REQUIRED"
        else:
            readiness = "NO_IMAGES"
        result.append(
            {
                "batch_id": batch.batch_id,
                "source": batch.source,
                "status": batch.status,
                "image_count": total,
                "accepted_bbox_count": accepted,
                "valid_accepted_bbox_count": valid_accepted,
                "missing_species_count": missing_species,
                "review_required_count": max(total - counts.get("ACCEPTED", 0) - counts.get("TRAINING_READY", 0) - counts.get("REJECTED", 0), 0),
                "rejected_count": counts.get("REJECTED", 0),
                "counts": counts,
                "readiness": readiness,
                "buildable": valid_accepted > 0,
                "review_url": f"/crop-review?batch_id={batch.batch_id}",
            }
        )
    from app.frozen_crop_bridge import crop_readiness
    from app.models import DatasetVersion

    frozen: list[dict[str, Any]] = []
    for dataset in db.scalars(select(DatasetVersion).where(DatasetVersion.status == "FROZEN").order_by(DatasetVersion.created_at.desc())).all():
        readiness = crop_readiness(db, dataset.dataset_version)
        readiness["source_dataset"] = dataset.dataset_version
        readiness["review_url"] = f"/crop-review?dataset_version={dataset.dataset_version}"
        frozen.append(readiness)
    return {
        "items": result,
        "frozen_datasets": frozen,
        "count": len(result),
        "contract": {"pipeline": "CROP_CLASSIFIER_V1", "source": "accepted_bbox", "source_types": ["RAW_BATCH", "FROZEN_DATASET"]},
    }


@router.get("/{dataset_version}/validation")
def validate_crop_dataset_endpoint(dataset_version: str, db: Session = Depends(get_db)) -> dict:
    version = _safe_version(dataset_version)
    root = _staging_root(version)
    manifest_path = _staging_manifest(root)
    validation = validate_crop_dataset(root, require_metadata=True, check_source_image=True)
    return {
        "dataset_version": version,
        "state": _staging_state(root, version, db, validation),
        "manifest_path": str(manifest_path),
        "validation": validation,
    }


@router.get("/{dataset_version}/status")
def crop_dataset_status_endpoint(dataset_version: str) -> dict:
    """Human-friendly status alias retained for console polling."""

    return validate_crop_dataset_endpoint(dataset_version)


@router.post("/{dataset_version}/freeze")
def freeze_crop_dataset_endpoint(
    dataset_version: str,
    payload: CropDatasetFreezeRequest,
    db: Session = Depends(get_db),
) -> dict:
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="Freeze 需要 confirm=true")
    version = _safe_version(dataset_version)
    root = _staging_root(version)
    try:
        return freeze_crop_dataset(
            root,
            dataset_version=version,
            bucket_name=get_bucket_name(),
            db=db,
            git_commit=(payload.git_commit or os.getenv("APP_GIT_COMMIT", "unknown")).strip() or "unknown",
        )
    except CropDatasetValidationError as exc:
        _rollback(db)
        raise HTTPException(status_code=400, detail={"error": "CROP_DATASET_INVALID", "report": exc.report}) from exc
    except Exception as exc:
        _rollback(db)
        raise HTTPException(status_code=400, detail={"error": "CROP_DATASET_FREEZE_FAILED", "reason": str(exc)}) from exc


__all__ = ["router"]
