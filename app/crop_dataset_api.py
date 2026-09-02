from __future__ import annotations

import csv
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
    dataset_version: str = Field(default=CROP_DATASET_VERSION, min_length=4, max_length=128)
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
    if manifest.is_file():
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
    resuming_failed_staging = manifest_path.is_file()
    try:
        report = build_reviewed_crop_dataset_from_db(
            db,
            root,
            dataset_version=version,
            expand_ratio=payload.expand_ratio,
            limit=payload.limit,
        )
        validation = _build_validation(root, report)
        if not validation.get("valid"):
            raise CropDatasetValidationError(validation)
        classes = _manifest_classes(manifest_path)
        result: dict = {
            "dataset_version": version,
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


@router.get("/{dataset_version}/validation")
def validate_crop_dataset_endpoint(dataset_version: str) -> dict:
    version = _safe_version(dataset_version)
    root = _staging_root(version)
    manifest_path = _staging_manifest(root)
    validation = validate_crop_dataset(root, require_metadata=True, check_source_image=True)
    return {
        "dataset_version": version,
        "state": _staging_state(root, version, None, validation),
        "manifest_path": str(manifest_path),
        "validation": validation,
    }


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
